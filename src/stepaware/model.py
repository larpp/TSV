"""Explicit local token injection; official parameter shapes, masks and head are reused."""
import copy
from pathlib import Path

import torch
from torch import nn

from .contracts import MODEL, ROOT, HERE, digest, file_sha, load_profile
from .metadata import validate_metadata
from .encoding import ProcessEncoder, SensorSummary, ProcessHeads, auxiliary_losses

from experiment_common.timestral.timestral_model import TimestralForecast
from experiment_common.timestral_csv.model import model_class as masked_class, masked_losses
from .policy_model import policy_model_class, validate_policy
from .identity import source_identity

def baseline_class():
    return policy_model_class(masked_class(TimestralForecast))


class ProcessSystem(nn.Module):
    """P can train these modules without constructing or executing Timestral."""
    def __init__(self, profile, artifact, sensor_count):
        super().__init__();validate_metadata(artifact,profile)
        self.encoder=ProcessEncoder(profile,artifact)
        self.summary=SensorSummary(sensor_count,profile['state_embedding_dim'])
        self.heads=ProcessHeads(profile,profile['state_embedding_dim'])
        self.sensor_count=sensor_count
        self.profile=copy.deepcopy(profile);self.artifact=copy.deepcopy(artifact)

    def forward(self, batch, dropout_mask=None):
        encoded=self.encoder(batch['process'],dropout_mask)
        summary=self.summary(batch['x'][...,:self.sensor_count])
        candidates=self.encoder.categorical(batch['candidate_categories'],encoded['dropout_mask'])
        output=self.heads(encoded['seconds'][:,-1],batch['process']['numeric'][:,-1],summary,
                          candidates,batch['candidate_mask'])
        output.update(encoded=encoded,summary=summary)
        output['remaining_available']=output['available']
        return output

    def losses(self, batch, output=None):
        output=self(batch) if output is None else output;l=batch['labels']
        values=auxiliary_losses(output,l['next_label'],l['remaining_seconds'],l['next_valid'],l['remaining_valid'])
        values['position_residual']=self.encoder.residual_penalty()
        values['total']=values['next']+values['remaining']+1e-4*values['position_residual']
        return values

    def checkpoint(self, sensors, scaler, *, training=None, optimizer=None):
        if len(sensors)!=self.sensor_count:raise ValueError('P sensor order/count differs')
        contract=dict(profile=self.profile,artifact=self.artifact,sensors=list(sensors),scaler=scaler)
        return dict(format='stepaware-p-v1',contract=contract,contract_sha256=digest(contract),code=source_identity(),
                    state={k:v.detach().cpu().clone() for k,v in self.state_dict().items()},
                    trainable_parameters=[k for k,v in self.named_parameters() if v.requires_grad],
                    training=training or {},optimizer=optimizer.state_dict() if optimizer else None)

    @classmethod
    def restore(cls,state,device='cpu'):
        if state.get('format')!='stepaware-p-v1' or state['code']!=source_identity() or state['contract_sha256']!=digest(state['contract']):
            raise ValueError('P checkpoint contract/code differs')
        c=state['contract'];model=cls(c['profile'],c['artifact'],len(c['sensors'])).to(device)
        if any(not torch.isfinite(v).all() for v in state['state'].values()):raise ValueError('Nonfinite P checkpoint')
        model.load_state_dict(state['state'],strict=True)
        restore_trainability(model,state['trainable_parameters'])
        return model


def restore_trainability(model,names):
    parameters=dict(model.named_parameters())
    if len(names)!=len(set(names)) or set(names)-set(parameters):raise ValueError('Invalid trainable parameter names')
    for name,value in parameters.items():value.requires_grad_(name in names)


def injected_decode(network, inputs, observed, groups, roles, context_delta, future_delta,
                    known_future=None, known_future_mask=None):
    """Local equivalent of official P32/H30 block with explicit post-projection additions.

    No hooks, no monkey-patching the shared network, only user-declared known covariates can supply future values.
    Native past rows receive zero additions. The official H30 refine is identity.
    """
    from timestral.model import ROLE_TARGET, ROLE_KNOWN_COV
    from timestral.masks import (compute_enc_key_validity,build_forecast_time_mask,
        build_two_stream_variate_mask,group_first_observed_patch,build_forecast_positions,
        build_two_stream_time_mask,build_two_stream_positions)
    from timestral.scaler import denormalize_quantiles
    c=network.config;rows,t=inputs.shape;ps=c.patch_size;ll=t//ps;h=1
    if t % ps or ps != 32: raise ValueError('Expected patch-aligned context and P32')
    if context_delta.shape!=(rows,ll,c.d_model) or future_delta.shape!=(rows,h,c.d_model):
        raise ValueError('Token injection row/patch alignment mismatch')
    target_rows=torch.ones(rows,dtype=torch.bool,device=inputs.device) if roles is None else roles==ROLE_TARGET
    if bool((context_delta[~target_rows]!=0).any()) or bool((future_delta[~target_rows]!=0).any()):
        raise ValueError('Native Covariate rows must not receive process additions')
    if bool(target_rows.all()):
        scaled,loc,scale=network.scale(inputs,observed)
        x_enc=network.representation(scaled,observed)+context_delta
        x_dec=network.representation.forecast_tokens(rows,h)+future_delta
        obs_per=observed.unflatten(-1,(ll,ps)).any(-1)
        valid=compute_enc_key_validity(obs_per,c.attend_missing_patches)
        tm=build_forecast_time_mask(ll,h,valid,inputs.device)
        vm=build_two_stream_variate_mask(groups,valid,h)
        ids,inverse=torch.unique(groups,return_inverse=True)
        first=group_first_observed_patch(obs_per,inverse,ids.numel())[inverse]
        positions=build_forecast_positions(ll,h,first)
        q=network._run_two_stream(x_enc,x_dec,tm,vm,positions)
    else:
        n=ll+h
        if known_future is None:
            known_future = torch.zeros_like(inputs[:, :ps])
            known_future_mask = torch.zeros_like(observed[:, :ps])
        full = torch.cat([inputs, known_future], -1)
        mask = torch.cat([observed, known_future_mask], -1)
        known = roles == ROLE_KNOWN_COV
        scaled,loc,scale=network.scale(full,mask)
        if known.any():
            scaled, loc, scale = network._known_cov_scaling(full, mask, scaled, loc, scale, known)
        x_enc=network.representation(scaled,mask)
        x_dec=network.representation.forecast_tokens(rows,n)
        # Only observed Context encoder tokens and the final future query are modified.
        x_enc=x_enc+torch.cat([context_delta,torch.zeros_like(future_delta)],1)
        x_dec=x_dec+torch.cat([torch.zeros_like(context_delta),future_delta],1)
        obs_per=mask.unflatten(-1,(n,ps)).any(-1)
        valid=compute_enc_key_validity(obs_per,c.attend_missing_patches)
        starts=torch.full((rows,),ll,device=inputs.device,dtype=torch.long)
        tm=build_two_stream_time_mask(starts,n,c.block_size,valid,bidirectional_rows=known)
        vm=build_two_stream_variate_mask(groups,valid,n,target_rows=target_rows,bidirectional_rows=known)
        ids,inverse=torch.unique(groups,return_inverse=True)
        first=group_first_observed_patch(obs_per,inverse,ids.numel())[inverse]
        positions=build_two_stream_positions(n,first)
        q=network._run_two_stream(x_enc,x_dec,tm,vm,positions)[:,-ps:]
    anchor_loc=loc[:,(ll-1)*ps:ll*ps][:,:1,None]
    anchor_scale=scale[:,(ll-1)*ps:ll*ps][:,:1,None]
    return denormalize_quantiles(q,anchor_loc,anchor_scale,c.use_asinh)


class StepAwareForecast(nn.Module):
    def __init__(self, profile, artifact, data_config, scaler):
        super().__init__();self.profile=copy.deepcopy(profile);self.artifact=copy.deepcopy(artifact)
        self.data_config=copy.deepcopy(data_config);self.scaler=copy.deepcopy(scaler)
        validate_metadata(artifact,profile)
        policy=validate_policy(data_config['normalization_policy'])
        if policy['variant']!='A1-005':raise ValueError('C/P requires the selected A1-005 policy')
        if profile['inject_future']:raise ValueError('Future renderer is not implemented')
        self.targets=list(data_config['sensors']);self.target_count=len(self.targets)
        if (policy['context'], policy['horizon']) != (data_config['context'], data_config['horizon']):
            raise ValueError('Window and normalization contracts differ')
        if len(scaler['mean'])!=self.target_count or len(scaler['std'])!=self.target_count:
            raise ValueError('Scaler target count differs')
        if not torch.isfinite(torch.tensor([scaler['mean'],scaler['std']])).all() or min(scaler['std'])<=0:
            raise ValueError('Invalid Train scaler')
        if policy['target_sensors']!=self.targets:raise ValueError('Target order differs')
        native=data_config.get('native_covariates')
        self.backbone=baseline_class()(horizon=data_config['horizon'],tune_mode='full',normalization_policy=policy,policy=native)
        if data_config['context'] > self.backbone.forecaster.max_context:
            raise ValueError('Context exceeds pretrained backbone capacity')
        self.process=ProcessSystem(profile,artifact,self.target_count)
        d=self.backbone.model.config.d_model;de=profile['state_embedding_dim']
        self.context_projection=nn.Linear(de,d)
        # Static Recipe known at t0 is allowed in future queries; no future Step/Loop is rendered.
        self.recipe_projection=nn.Linear(de,d)
        self.sensor_embedding=nn.Embedding(self.target_count,d)
        for layer in (self.context_projection,self.recipe_projection):
            nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.sensor_embedding.weight)
        self.levels=self.backbone.levels;self.median_index=self.backbone.median_index

    def native_rows(self, x):
        """Unpack CSV context and declared horizon values into official Timestral roles."""
        n = self.target_count
        if x.ndim != 3 or x.shape[1:] != (self.data_config['context'], self.backbone.input_width) or torch.isinf(x).any():
            raise ValueError('Input differs from the saved TSV target/covariate layout')
        forecaster = self.backbone.forecaster
        inputs, observed = forecaster._as_context(x[..., :n].transpose(1, 2).contiguous(), None)
        if not observed.any(-1).all():
            raise ValueError('Every target requires observed history')
        past = known = None
        policy = self.data_config.get('native_covariates')
        if policy:
            columns = policy['columns']
            npast = len(columns['past_covariates'])
            nknown = len(columns['known_covariates'])
            past_end = n + npast + len(columns['categorical_past_covariates'])
            known_end = past_end + nknown + len(columns['categorical_known_covariates'])
            past = x[..., n:past_end].transpose(1, 2) if past_end > n else None
            if known_end > past_end:
                history = x[..., past_end:known_end]
                tail = x[..., known_end:].reshape(len(x), x.shape[1], -1, history.shape[-1])
                horizon = tail.transpose(1, 2).reshape(len(x), -1, history.shape[-1])[:, :self.data_config['horizon']]
                known = torch.cat([history, horizon], dim=1).transpose(1, 2)
            if columns['categorical_past_covariates'] or columns['categorical_known_covariates']:
                target = forecaster._encoding_target(inputs, observed)
                if columns['categorical_past_covariates']:
                    encoded = forecaster._encode_categorical(past[:, npast:], target)
                    past = torch.cat([past[:, :npast], encoded], dim=1)
                if columns['categorical_known_covariates']:
                    encoded = forecaster._encode_categorical(known[:, nknown:], target)
                    known = torch.cat([known[:, :nknown], encoded], dim=1)
        return forecaster._assemble_rows(inputs, observed, self.data_config['horizon'], past, known)

    def token_additions(self, encoded, rows):
        batch,patches,_=encoded['patches'].shape;n=self.target_count;dim=self.sensor_embedding.embedding_dim
        ctx=self.context_projection(encoded['patches']) if self.profile['inject_context'] else encoded['patches'].new_zeros(batch,patches,dim)
        identity=self.sensor_embedding.weight if self.profile['sensor_embedding'] else torch.zeros_like(self.sensor_embedding.weight)
        ctx=ctx[:,None,:,:]+identity[None,:,None,:]
        ctx=ctx*encoded['patch_valid'][:,None,:,None]
        future=self.recipe_projection(encoded['recipe']) if self.profile['static_recipe_future'] else ctx.new_zeros(batch,dim)
        fut=future[:,None,None,:]+identity[None,:,None,:]
        # Batch-major, target rows first, then native past rows: same _assemble_rows contract.
        ctx=torch.cat([ctx,ctx.new_zeros(batch,rows-n,patches,dim)],1).reshape(batch*rows,patches,dim)
        fut=torch.cat([fut,fut.new_zeros(batch,rows-n,1,dim)],1).reshape(batch*rows,1,dim)
        return ctx,fut

    def forward(self,batch,dropout_mask=None):
        x=batch['x'];n=self.target_count
        if batch['process']['categories'].shape[:2]!=x.shape[:2]:raise ValueError('State and sensor time axes differ')
        rows,mask,roles,known,known_mask,count = self.native_rows(x)
        proc=self.process(batch,dropout_mask)
        if count!=n:raise ValueError('Target packing differs')
        b,v,t=rows.shape;groups=torch.arange(b,device=x.device).repeat_interleave(v)
        ctx,fut=self.token_additions(proc['encoded'],v)
        q=injected_decode(self.backbone.model,rows.reshape(b*v,t),mask.reshape(b*v,t),groups,roles,ctx,fut,known,known_mask)
        h = self.data_config['horizon']
        q=q[:,:h].reshape(b,v,h,len(self.levels))[:,:n].permute(0,2,1,3)
        if not torch.isfinite(q).all():raise ValueError('Nonfinite Step-aware prediction')
        return dict(quantiles=q,prediction=q[...,self.median_index],process=proc,
                    context_delta=ctx,future_static_delta=fut)

    def losses(self,batch,output=None):
        output=self(batch) if output is None else output
        forecast,mse=masked_losses(output['quantiles'],batch['y'],self.levels,self.median_index)
        aux=self.process.losses(batch,output['process'])
        return dict(forecast=forecast,mse=mse,**aux,
                    joint=forecast+.1*aux['next']+.1*aux['remaining']+1e-4*aux['position_residual'])

    def checkpoint(self, *, training=None, optimizer=None, scheduler=None, p_source=None):
        return dict(format='stepaware-cp-v1',profile=self.profile,artifact=self.artifact,data_config=self.data_config,
            scaler=self.scaler,source=self.backbone.pretrained_source,code=source_identity(),
            contract_sha256=digest([self.profile,self.artifact,self.data_config,self.scaler]),
            backbone_state={k:v.detach().cpu().clone() for k,v in self.backbone.state_dict().items()},
            adapter_state={k:v.detach().cpu().clone() for k,v in self.state_dict().items() if not k.startswith('backbone.')},
            trainable_parameters=[k for k,v in self.named_parameters() if v.requires_grad],
            training=training or {},p_source=p_source,optimizer=optimizer.state_dict() if optimizer else None,
            scheduler=scheduler.state_dict() if scheduler else None,future_renderer=False)

    @classmethod
    def restore(cls,state,device='cpu'):
        if state.get('format')!='stepaware-cp-v1' or state.get('future_renderer') or state['code']!=source_identity() or state['contract_sha256']!=digest([state['profile'],state['artifact'],state['data_config'],state['scaler']]):
            raise ValueError('Step-aware checkpoint format/code differs; baseline loaders cannot restore new modules')
        model=cls(state['profile'],state['artifact'],state['data_config'],state['scaler']).to(device)
        if model.backbone.pretrained_source!=state['source']:raise ValueError('Official source differs')
        model.backbone.load_state_dict(state['backbone_state'],strict=True)
        expected={k for k in model.state_dict() if not k.startswith('backbone.')}
        if set(state['adapter_state'])!=expected:raise ValueError('Adapter checkpoint keys differ')
        merged=model.state_dict();merged.update(state['adapter_state'])
        if any(not torch.isfinite(v).all() for v in merged.values() if v.is_floating_point()):raise ValueError('Nonfinite checkpoint')
        model.load_state_dict(merged,strict=True)
        restore_trainability(model,state['trainable_parameters'])
        return model

    def load_p_checkpoint(self,state):
        restored=ProcessSystem.restore(state)
        c=state['contract']
        if c!=dict(profile=self.profile,artifact=self.artifact,sensors=self.targets,scaler=self.scaler):
            raise ValueError('P/C input, sensor order or Train scaler differs')
        self.process.load_state_dict(restored.state_dict(),strict=True)
        return dict(contract_sha256=state['contract_sha256'],training=state['training'])
