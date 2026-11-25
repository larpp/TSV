"""Paired C/D wrapper. Only predicted process features enter the future query."""
from pathlib import Path
import copy
import numpy as np
import torch
from torch import nn
from ..contracts import digest,file_sha
from ..encoding import FeatureCodec
from ..model import StepAwareForecast,injected_decode,restore_trainability,source_identity
from ..identity import source_identity as execution_identity
from .renderer import Renderer
from .patterns import validate_bank


def injection_coefficient(epoch,warmup=3):
    return min(1.,max(0.,(epoch-warmup-1)/2.))


class Forecast(StepAwareForecast):
    def future_additions(self,features,dropout_mask,rows):
        if set(features)!={'categories','numeric','pad'}:raise ValueError('Future input allows process fields only')
        ids=features['categories'];numbers=features['numeric'];pad=features['pad']
        if ids.shape[1]!=32 or numbers.shape[:2]!=ids.shape[:2] or pad.shape!=ids.shape[:2]:
            raise ValueError('Ordered future H30 + two pad positions required')
        h = self.data_config['horizon']
        if not pad[:,h:].all():raise ValueError('Future beyond the configured horizon is forbidden')
        encoder=self.process.encoder
        if self.future_mode=='reverse':
            ids=torch.cat([ids[:,:h].flip(1),ids[:,h:]],1)
            numbers=torch.cat([numbers[:,:h].flip(1),numbers[:,h:]],1)
            pad=torch.cat([pad[:,:h].flip(1),pad[:,h:]],1)
        elif self.future_mode not in ('actual','off'):raise ValueError('Unknown future ablation')
        # Same state/ordered-patch encoder as the past, deliberately OMIT set_mlp entirely.
        z=encoder.categorical(ids,dropout_mask)+encoder.numeric(numbers)
        z=z.masked_fill(pad[...,None],0)
        patch=torch.nn.functional.gelu(encoder.patch(z.reshape(len(z),1,-1)))
        delta=self.future_projection(patch)*(~pad).any(1)[:,None,None]
        if self.future_mode=='off':delta=delta*0
        n=self.target_count;b=len(z);dim=delta.shape[-1]
        target=delta[:,None,:,:].expand(b,n,1,dim)
        return torch.cat([target,delta.new_zeros(b,rows-n,1,dim)],1).reshape(b*rows,1,dim)

    def __init__(self,profile,artifact,data_config,scaler,bank,enable_future):
        super().__init__(profile,artifact,data_config,scaler)
        self.bank=copy.deepcopy(validate_bank(bank));self.enable_future=bool(enable_future);self.coefficient=0.
        self.codec=FeatureCodec(profile,artifact);self.renderer=Renderer(artifact,bank,data_config['horizon']);self.future_mode='actual'
        with torch.random.fork_rng(devices=[]):
            self.future_projection=nn.Linear(profile['state_embedding_dim'],self.backbone.model.config.d_model)
            nn.init.zeros_(self.future_projection.weight);nn.init.zeros_(self.future_projection.bias)

    def render(self,batch,proc):
        device=batch['x'].device;features=[];plans=[]
        remaining=proc['remaining_seconds'].detach().cpu().tolist()
        sigma=proc['sigma'].detach().cpu().tolist();probs=proc['probabilities'].detach().cpu().tolist()
        for i,entry in enumerate(batch['causal']):
            first=True
            def predict(state,candidates):
                nonlocal first
                if first:
                    first=False
                    return dict(remaining=remaining[i],sigma=sigma[i],probabilities=probs[i][:len(candidates)])
                # Recondition heads on PLANNED state with the SAME observed past sensor summary.
                with torch.no_grad():
                    f=self.codec.encode([state],np.full((1,5),np.nan,np.float32))
                    ids=torch.as_tensor(f['categories'],device=device);num=torch.as_tensor(f['numeric'],device=device)
                    drop=proc['encoded']['dropout_mask'][i:i+1]
                    encoded=self.process.encoder.categorical(ids,drop)+self.process.encoder.numeric(num)
                    # Same Head input layout as P/C; future Set values remain explicitly missing.
                    encoded=encoded+self.process.encoder.set_mlp(torch.zeros(1,10,device=device,dtype=num.dtype))
                    cats=self.codec.categories(candidates).reshape(1,-1,9)
                    if not candidates:return dict(remaining=float(self.renderer.horizon),sigma=None,probabilities=[])
                    ce=self.process.encoder.categorical(torch.as_tensor(cats,device=device),drop)
                    out=self.process.heads(encoded,num,proc['summary'][i:i+1].detach(),ce,torch.ones(1,len(candidates),dtype=torch.bool,device=device))
                    return dict(remaining=float(out['remaining_seconds'][0]),sigma=float(out['sigma'][0]),probabilities=out['probabilities'][0].cpu().tolist())
            plan=self.renderer.render(entry['current'],entry['recent'],predict);plans.append(plan)
            encoded=self.codec.encode(plan['states']+[None]*(32-self.renderer.horizon),np.full((32,5),np.nan,np.float32))
            features.append({k:encoded[k] for k in ('categories','numeric','pad')})
        tensor={k:torch.as_tensor(np.stack([f[k] for f in features]),device=device) for k in features[0]}
        return tensor,plans

    def forward(self,batch,dropout_mask=None):
        x=batch['x'];n=self.target_count
        if 'oracle_future' in batch:raise ValueError('Oracle future inputs are forbidden in C/D')
        rows,mask,roles,known,known_mask,count = self.native_rows(x)
        proc=self.process(batch,dropout_mask)
        if count!=n:raise ValueError('Target order differs')
        b,v,t=rows.shape;groups=torch.arange(b,device=x.device).repeat_interleave(v)
        ctx,static=self.token_additions(proc['encoded'],v);delta=torch.zeros_like(static);plans=[]
        if self.enable_future:
            rendered,plans=self.render(batch,proc)
            delta=self.future_additions(rendered,proc['encoded']['dropout_mask'],v)*self.coefficient
        q=injected_decode(self.backbone.model,rows.reshape(b*v,t),mask.reshape(b*v,t),groups,roles,ctx,static+delta,known,known_mask)
        h = self.data_config['horizon']
        q=q[:,:h].reshape(b,v,h,len(self.levels))[:,:n].permute(0,2,1,3)
        if not torch.isfinite(q).all():raise ValueError('Nonfinite forecast')
        return dict(quantiles=q,prediction=q[...,self.median_index],process=proc,context_delta=ctx,
                    future_static_delta=static,rendered_delta=delta,rendered_plans=plans)

    def checkpoint(self,**kwargs):
        state=super().checkpoint(**kwargs)
        state.update(format='stepaware-cd-renderer-v1',bank=self.bank,enable_future=self.enable_future,
                     coefficient=self.coefficient,future_renderer=self.enable_future,oracle_only=False,execution_code=execution_identity())
        return state

    @classmethod
    def restore(cls,state,device='cpu', *, context=None, horizon=None):
        if (state.get('format')!='stepaware-cd-renderer-v1' or state.get('oracle_only') is not False
            or state.get('execution_code')!=execution_identity() or state.get('code')!=source_identity()
            or state.get('contract_sha256')!=digest([state['profile'],state['artifact'],state['data_config'],state['scaler']])):
            raise ValueError('C/D artifact/code/contract mismatch')
        # Check the original saved contract above before applying evaluation-only lengths.
        config = copy.deepcopy(state['data_config'])
        for name, value in [('context', context), ('horizon', horizon)]:
            if value is not None:
                config[name] = config['normalization_policy'][name] = value
                if 'native_covariates' in config:
                    config['native_covariates'][name] = value
        model=cls(state['profile'],state['artifact'],config,state['scaler'],state['bank'],state['enable_future']).to(device)
        if model.backbone.pretrained_source!=state['source']:raise ValueError('Official source changed')
        model.backbone.load_state_dict(state['backbone_state'],strict=True)
        if set(state['adapter_state'])!={k for k in model.state_dict() if not k.startswith('backbone.')}:raise ValueError('Adapter keys differ')
        merged=model.state_dict();merged.update(state['adapter_state'])
        if any(not torch.isfinite(v).all() for v in merged.values() if v.is_floating_point()):raise ValueError('Nonfinite checkpoint')
        model.load_state_dict(merged,strict=True);restore_trainability(model,state['trainable_parameters'])
        coefficient=state['coefficient']
        if not np.isfinite(coefficient) or not 0<=coefficient<=1:raise ValueError('Invalid future coefficient')
        if state['future_renderer']!=model.enable_future:raise ValueError('Renderer flag mismatch')
        model.coefficient=coefficient;return model
