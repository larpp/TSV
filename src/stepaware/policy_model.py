"""Isolated A0/A1/A2 numerical policies; official parameter structure is preserved."""
import copy
import math
import types

import torch

POLICY_KEY = 'normalization_policy'
VARIANTS = ('A0', 'A1-005', 'A1-010', 'A2')


def make_policy(variant, sensors, context=128, horizon=30):
    if context < 1 or not 1 <= horizon <= 32:
        raise ValueError('Expected positive context and a single forecast patch (1..32 steps)')
    if variant not in VARIANTS:
        raise ValueError('Unknown comparison policy')
    floor = {'A0':1e-6, 'A1-005':.05, 'A1-010':.1, 'A2':1e-6}[variant]
    return dict(schema_version=1, variant=variant, target_sensors=list(sensors),
                target_scale_floors=[floor]*len(sensors), floor_units='Train-global standardized target units',
                loss_space='internal_asinh' if variant=='A2' else 'train_standardized',
                minimum_scale=1e-6, context=context, horizon=horizon, patch_size=32,
                target_context_scaling='all causal patch frames and matching inverse use the same floor',
                covariate_scaling='official unchanged', refine='single future patch; H>32 rejected',
                label_transform='observed labels only; float64 affine/asinh then output dtype; detached past-only frame')


def validate_policy(policy):
    if not isinstance(policy,dict):raise ValueError('Missing saved normalization policy')
    expected=make_policy(policy.get('variant'),policy.get('target_sensors',[]), policy['context'], policy['horizon'])
    if not expected['target_sensors'] or policy!=expected:
        raise ValueError('Normalization policy differs from this versioned A0/A1/A2 contract')
    return policy


def internal_pinball(internal, target, loc, scale, levels):
    """Direct pre-inverse loss. Never invert rounded output back to normalized space."""
    valid=torch.isfinite(target)
    if torch.isinf(target).any() or not valid.any():raise ValueError('No finite targets or infinite target')
    locations=loc[:,None,:].expand_as(target)[valid].detach().double()
    scales=scale[:,None,:].expand_as(target)[valid].detach().double()
    truth=torch.asinh((target[valid].detach().double()-locations)/scales).to(internal.dtype)
    prediction=internal[valid]
    error=truth[:,None]-prediction
    knots=torch.as_tensor(levels,dtype=internal.dtype,device=internal.device)
    return torch.maximum(knots*error,(knots-1)*error).mean()


def policy_model_class(masked_base):
    class PolicyForecast(masked_base):
        def __init__(self,*args,normalization_policy,policy=None,**kwargs):
            self.normalization_policy=copy.deepcopy(validate_policy(normalization_policy))
            self.target_count=len(self.normalization_policy['target_sensors'])
            self.row_count = self.target_count
            self.input_width = self.target_count
            if policy:
                from experiment_common.timestral_covariates.data import validate_policy as validate_covariates
                validate_covariates(policy)
                if (policy['target_count'], policy['context'], policy['horizon']) != (self.target_count, normalization_policy['context'], normalization_policy['horizon']):
                    raise ValueError('Covariate and normalization policy shapes differ')
                columns = policy['columns']
                self.row_count += sum(len(names) for names in columns.values())
                # CSV packs the known horizon into a second block of columns.
                self.input_width = self.row_count + math.ceil(policy['horizon'] / policy['context']) * (len(columns['known_covariates']) + len(columns['categorical_known_covariates']))
            super().__init__(*args,policy=policy,**kwargs)
            c=self.model.config
            if self.horizon != normalization_policy['horizon'] or c.patch_size!=32 or not c.use_asinh or c.minimum_scale!=1e-6 or c.stabilize_with_global:
                raise ValueError('Policies require the pinned P32, H30, asinh, unstabilized 28M model')
            if self.normalization_policy['variant'].startswith('A1'):
                original_scale=self.model.scale
                floors=self.normalization_policy['target_scale_floors']
                nrows=self.row_count
                # Instance-local behavior only: no edits or new parameters in official source.
                def scale_with_floor(network,inputs,observed):
                    from timestral.scaler import apply_affine_asinh_masked
                    if inputs.shape[0]%nrows:raise ValueError('Packed target/covariate row grouping differs')
                    scaled,loc,scale=original_scale(inputs,observed)
                    per_row=inputs.new_tensor(floors+[1e-6]*(nrows-len(floors))).repeat(inputs.shape[0]//nrows)
                    corrected=torch.maximum(scale,per_row[:,None])
                    scaled=apply_affine_asinh_masked(inputs,loc,corrected,observed,True)
                    return scaled,loc,corrected
                self.model.scale=types.MethodType(scale_with_floor,self.model)

        def quantiles(self,x):
            if x.shape[1] != self.normalization_policy['context']: raise ValueError('Context differs from policy')
            expected=self.input_width
            if x.shape[2]!=expected:raise ValueError('Target/past-covariate width differs from saved policy')
            return super().quantiles(x)

        def target_frame(self,x):
            from timestral.scaler import causal_patch_loc_scale
            rows=x[...,:self.target_count].transpose(1,2).contiguous().flatten(0,1)
            observed=torch.isfinite(rows)
            loc,scale=causal_patch_loc_scale(torch.nan_to_num(rows),observed,32,1e-6,False)
            loc=loc[:,-1].reshape(len(x),self.target_count)
            scale=scale[:,-1].reshape_as(loc)
            floor=x.new_tensor(self.normalization_policy['target_scale_floors'])
            return loc.detach(),torch.maximum(scale,floor[None,:]).detach()

        def loss_and_metrics(self,x,y):
            from experiment_common.timestral_csv.model import masked_losses
            if self.normalization_policy['loss_space']=='train_standardized':
                quantiles=self.quantiles(x)
                loss,mse=masked_losses(quantiles,y,self.levels,self.median_index)
                return loss,mse
            captured=[]
            handle=self.model.head.register_forward_hook(lambda module,inputs,output:captured.append(output))
            try:
                quantiles=self.quantiles(x)
            finally:
                handle.remove()
            if len(captured)!=1:raise ValueError('Expected a single official Head invocation at H30')
            raw=captured[0]
            # Native covariates decode Context+future tokens, target-only decodes future only.
            # In both cases the FINAL 32 samples are the one forecast patch.
            internal=raw[:,-32:][:,:self.horizon].reshape(len(x),self.row_count,self.horizon,len(self.levels))
            internal=internal[:,:self.target_count].permute(0,2,1,3)
            loc,scale=self.target_frame(x)
            loss=internal_pinball(internal,y,loc,scale,self.levels)
            valid=torch.isfinite(y)
            mse=torch.nn.functional.mse_loss(quantiles[...,self.median_index][valid],y[valid])
            return loss,mse

    return PolicyForecast


def train_batch(model,x,y,optimizer):
    optimizer.zero_grad(set_to_none=True)
    loss,mse=model.loss_and_metrics(x,y)
    if not torch.isfinite(loss) or not torch.isfinite(mse):raise ValueError('Nonfinite policy loss/metric')
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    optimizer.step()
    return dict(pinball=float(loss.detach()),median_scaled_mse=float(mse.detach()))
