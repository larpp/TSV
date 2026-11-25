"""Training adapter to official Timestral mask/decoding APIs; no new weights."""
import math
import torch


def model_class(base):
    class MaskedForecast(base):
        def __init__(self,*args,policy=None,**kwargs):
            self.csv_covariates=policy
            super().__init__(*args,**kwargs)

        def quantiles(self,x):
            p=self.csv_covariates
            if p is None and torch.isfinite(x).all():return super().quantiles(x)
            if x.ndim!=3 or torch.isinf(x).any():raise ValueError('CSV inputs must be finite or NaN, never infinite')
            v=p['target_count'] if p else x.shape[-1]
            inputs,observed=self.forecaster._as_context(x[...,:v].transpose(1,2).contiguous(),None)
            if not observed.any(dim=-1).all():raise ValueError('Every sensor requires an observed context value')
            if x.shape[1]>self.model.config.context_length:raise ValueError('Context exceeds model limit')
            past=known=None
            if p:
                g=p['columns'];t=p['context'];h=p['horizon']
                npast=len(g['past_covariates']);ncatpast=len(g['categorical_past_covariates'])
                nknown=len(g['known_covariates']);ncatknown=len(g['categorical_known_covariates'])
                cp=npast+ncatpast;ck=nknown+ncatknown;chunks=math.ceil(h/t)
                if x.shape[1:]!=(t,v+cp+ck+ck*chunks):raise ValueError('CSV covariate dimensions differ from checkpoint')
                if not self.forecaster.supports_covariates or (ck and not self.forecaster.supports_known_covariates):
                    raise ValueError('Official checkpoint lacks requested covariate support')
                past_all=x[...,v:v+cp].transpose(1,2)
                known_past=x[...,v+cp:v+cp+ck];tail=x[...,v+cp+ck:]
                future=tail.reshape(len(x),t,chunks,ck).permute(0,2,1,3).reshape(len(x),chunks*t,ck)[:,:h] if ck else tail[:,:0]
                known_all=torch.cat([known_past,future],dim=1).transpose(1,2) if ck else None
                past=past_all[:,:npast] if npast else None
                known=known_all[:,:nknown] if nknown else None
                if ncatpast or ncatknown:
                    target=self.forecaster._encoding_target(inputs,observed)
                    cpv=self.forecaster._encode_categorical(past_all[:,npast:] if ncatpast else None,target)
                    ckv=self.forecaster._encode_categorical(known_all[:,nknown:] if ncatknown else None,target)
                    if cpv is not None:past=torch.cat([past,cpv],dim=1) if past is not None else cpv
                    if ckv is not None:known=torch.cat([known,ckv],dim=1) if known is not None else ckv
            rows,mask,roles,future,future_mask,count=self.forecaster._assemble_rows(inputs,observed,self.horizon,past,known)
            q=self.forecaster._block_ar_decode(rows,mask,self.horizon,roles,future,future_mask)
            result=q[:,:count].permute(0,2,1,3)
            if result.shape!=(len(x),self.horizon,v,len(self.levels)) or not torch.isfinite(result).all():
                raise ValueError('Invalid masked Timestral prediction')
            return result
    return MaskedForecast


def masked_losses(q,y,levels,median_index):
    mask=torch.isfinite(y)
    if torch.isinf(y).any() or not mask.any():raise ValueError('No observed targets, or infinite target')
    # Index BEFORE subtraction: NaN*0 is still NaN and poisons gradients.
    prediction=q[mask];truth=y[mask]
    error=truth[:,None]-prediction
    levels=torch.as_tensor(levels,device=q.device,dtype=q.dtype)
    pinball=torch.maximum(levels*error,(levels-1)*error).mean()
    mse=torch.nn.functional.mse_loss(prediction[:,median_index],truth)
    return pinball,mse


def train_batch(model,x,y,optimizer):
    optimizer.zero_grad(set_to_none=True)
    q=model.quantiles(x)
    loss,mse=masked_losses(q,y,model.levels,model.median_index)
    if not torch.isfinite(loss) or not torch.isfinite(mse):raise ValueError('Nonfinite masked loss')
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0,error_if_nonfinite=True)
    optimizer.step()
    return dict(pinball=float(loss.detach().cpu()),median_scaled_mse=float(mse.detach().cpu()))
