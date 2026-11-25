"""Gradient-enabled official Timestral inference primitive for local adaptation.

This adapter is project code, not Mistral Forge or an official fine-tuning recipe.
No change is made to the upstream code or pretrained architecture.
"""
import math

import torch
from torch import nn

from .model_runtime import HERE, official_model_class, verify_pretrained


class TimestralForecast(nn.Module):
    def __init__(self, horizon, tune_mode='full', past_step_channels=0):
        super().__init__()
        if horizon < 1 or tune_mode not in ('head','full'): raise ValueError('Invalid Timestral configuration')
        load_model, Forecaster, self.pad_to_patch = official_model_class()
        self.pretrained_source = verify_pretrained()
        # Official loader uses strict=True and rebuilds RoPE frequencies after moving.
        self.model = load_model(HERE/self.pretrained_source['local_directory'], device='cpu', dtype=torch.float32)
        self.horizon = horizon
        if type(past_step_channels) is not int or past_step_channels not in (0,1,2):
            raise ValueError('Expected zero, one or two past Step channels')
        self.past_step_channels = past_step_channels
        if past_step_channels and not self.forecaster_supports_steps():
            raise ValueError('This checkpoint does not support past covariates')
        self.tune_mode = tune_mode
        self.forecaster = Forecaster(self.model,quantile_real_cap_k=0)
        self.model.requires_grad_(tune_mode == 'full')
        if tune_mode == 'head':
            self.model.head.requires_grad_(True)
            self.model.transformer.out_norm.requires_grad_(True)
        self.levels = tuple(self.model.config.quantile_knots)
        self.median_index = self.levels.index(0.5)
        if math.ceil(horizon/self.model.config.patch_size) > self.model.config.block_size:
            raise ValueError('This training adapter supports one decode block only (up to1024steps)')

    def train(self, mode=True):
        super().train(mode)
        if self.tune_mode == 'head':
            self.model.eval()
            self.model.head.train(mode)
            self.model.transformer.out_norm.train(mode)
        return self

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn,recurse=recurse)
        # The official loader rebuilds this after its device move; our parent
        # adapter can be moved again by the caller, so preserve the same rule.
        if hasattr(self,'model'):
            self.model.transformer.rope.rebuild_inv_freq()
        return result

    def quantiles(self, x):
        if x.ndim != 3 or x.shape[1] < 1:
            raise ValueError('Expected finite [batch, context, sensors] inputs')
        width = self.past_step_channels
        if width and x.shape[-1] <= width: raise ValueError('Missing target channels')
        target = x[...,:-width] if width else x
        if not torch.isfinite(target).all(): raise ValueError('Nonfinite target input')
        b, length, sensors = target.shape
        if length > self.model.config.context_length:
            raise ValueError('Input exceeds pretrained maximum context; refusing silent truncation')
        if width:
            past = x[...,-width:].transpose(1,2).contiguous()
            if torch.isinf(past).any(): raise ValueError('Infinite past Step code')
            inputs = target.transpose(1,2).contiguous()
            # These official primitives preserve gradients, unlike the public
            # forecast() method decorated with torch.inference_mode(). Native
            # multivariate categorical covariates use raw numeric codes as well.
            rows,mask,roles,future,future_mask,n_target = self.forecaster._assemble_rows(
                inputs,torch.ones_like(inputs,dtype=torch.bool),self.horizon,past,None)
            q = self.forecaster._block_ar_decode(rows,mask,self.horizon,roles,future,future_mask)
            q = q[:,:n_target].permute(0,2,1,3)
            if q.shape != (b,self.horizon,sensors,len(self.levels)) or not torch.isfinite(q).all():
                raise ValueError('Invalid Step-conditioned forecast')
            return q
        # All sensors in one Window attend jointly; distinct Windows never attend.
        context = x.transpose(1,2).contiguous()
        observed = torch.ones_like(context,dtype=torch.bool)
        context, observed = self.pad_to_patch(context,observed,self.model.config.patch_size)
        context = context.reshape(b*sensors,-1)
        observed = observed.reshape_as(context)
        groups = torch.arange(b,device=x.device).repeat_interleave(sensors)
        patches = math.ceil(self.horizon/self.model.config.patch_size)
        raw = self.model.predict_block(context,observed,groups,patches)
        # Mandatory for horizons >32: use the official advancing prediction frame.
        refined = self.forecaster._refine_block(raw,context,observed,None)
        q = refined[:,:self.horizon].reshape(b,sensors,self.horizon,len(self.levels)).permute(0,2,1,3)
        if q.shape != (b,self.horizon,sensors,len(self.levels)) or not torch.isfinite(q).all():
            raise ValueError('Invalid Timestral quantile forecast')
        return q

    def forecaster_supports_steps(self):
        return self.model.config.covariate_enabled

    def forward(self,x):
        return self.quantiles(x)[...,self.median_index]

    def delta_state(self):
        return {name:p.detach().cpu().clone() for name,p in self.named_parameters() if p.requires_grad}

    def load_delta(self,state):
        expected = {name for name,p in self.named_parameters() if p.requires_grad}
        if set(state) != expected: raise ValueError('Checkpoint parameter keys differ from tuning policy')
        params = dict(self.named_parameters())
        with torch.no_grad():
            for name,value in state.items():
                if value.shape != params[name].shape or not torch.isfinite(value).all():
                    raise ValueError('Invalid checkpoint parameter: '+name)
                params[name].copy_(value)

    def counts(self):
        return dict(total_parameters=sum(p.numel() for p in self.parameters()),
                    trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),
                    tune_mode=self.tune_mode,precision='float32',point_forecast='median',
                    joint_sensor_attention=True)


def pinball_loss(quantiles, target, levels):
    error = target.unsqueeze(-1) - quantiles
    levels = torch.as_tensor(levels,device=quantiles.device,dtype=quantiles.dtype)
    return torch.maximum(levels*error,(levels-1)*error).mean()


def train_batch(model,x,y,optimizer):
    optimizer.zero_grad(set_to_none=True)
    q = model.quantiles(x)
    loss = pinball_loss(q,y,model.levels)
    mse = torch.nn.functional.mse_loss(q[...,model.median_index],y)
    if not torch.isfinite(loss) or not torch.isfinite(mse): raise ValueError('Nonfinite training loss')
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0,error_if_nonfinite=True)
    optimizer.step()
    return dict(pinball=float(loss.detach().cpu()),median_scaled_mse=float(mse.detach().cpu()))
