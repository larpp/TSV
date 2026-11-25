"""Categorical identity and ordered second-level features bypass Timestral scaling."""
import math
import numpy as np
import torch
from torch import nn

from .contracts import CATEGORIES, NUMERIC
from .tracker import category_values, numeric_values
from .metadata import validate_metadata, candidate_states


class FeatureCodec:
    def __init__(self, profile, artifact):
        self.profile=profile; self.artifact=validate_metadata(artifact,profile)
        self.lookup={n:{v:i for i,v in enumerate(artifact['vocabulary'][n])} for n in CATEGORIES}

    def categories(self, states):
        return np.array([[self.lookup[n].get(category_values(s)[n],2) for n in CATEGORIES] if s is not None else [0]*len(CATEGORIES)
                         for s in states],dtype=np.int64)

    def encode(self, states, sets):
        values=np.asarray(sets,dtype=np.float32);mask=np.isfinite(values)
        if values.shape!=(len(states),5) or np.isinf(values).any():raise ValueError('Expected finite/NaN absolute Set values')
        sc=self.artifact['set_scaler']
        normalized=np.where(mask,(values-np.asarray(sc['mean']))/np.asarray(sc['std']),0).astype(np.float32)
        pad=np.array([s is None for s in states]);mask[pad]=False;normalized[pad]=0
        return dict(categories=self.categories(states),numeric=np.array([numeric_values(s) if s else [0.]*len(NUMERIC) for s in states],dtype=np.float32),
                    sets=normalized,set_mask=mask,pad=pad)

    def candidates(self, current):
        states=candidate_states(current,self.artifact)
        return states,self.categories(states).reshape(-1,len(CATEGORIES))


class ProcessEncoder(nn.Module):
    def __init__(self, profile, artifact):
        super().__init__();self.profile=profile;d=profile['state_embedding_dim']
        self.tables=nn.ModuleDict({n:nn.Embedding(len(artifact['vocabulary'][n]),d,padding_idx=0) for n in CATEGORIES})
        self.residual_names=('occurrence','main_position','sub_position')
        for n in self.residual_names:nn.init.zeros_(self.tables[n].weight)
        self.numeric=nn.Sequential(nn.Linear(len(NUMERIC),d),nn.GELU(),nn.Linear(d,d))
        self.set_mlp=nn.Sequential(nn.Linear(10,d),nn.GELU(),nn.Linear(d,d))
        self.patch=nn.Linear(32*d,d)

    def dropout_mask(self, batch, device, explicit=None):
        if explicit is not None:
            if explicit.shape!=(batch,):raise ValueError('Recipe dropout mask must be shared once per Window')
            return explicit.to(device=device,dtype=torch.bool)
        return torch.rand(batch,device=device)<self.profile['recipe_dropout_probability'] if self.training else torch.zeros(batch,device=device,dtype=torch.bool)

    def categorical(self, categories, drop):
        ids=categories.clone();shape=(len(drop),)+(1,)*(ids.ndim-2)
        condition=drop.reshape(shape)
        # Recipe UNKNOWN is learned; recipe-position residual is neutral under dropout.
        ids[...,0]=torch.where(condition,2,ids[...,0]);ids[...,3]=torch.where(condition,0,ids[...,3])
        value=sum(self.tables[n](ids[...,i]) for i,n in enumerate(CATEGORIES))
        return value.masked_fill((categories==0).all(-1)[...,None],0)

    def forward(self, batch, drop=None):
        ids=batch['categories'];drop=self.dropout_mask(len(ids),ids.device,drop)
        z=self.categorical(ids,drop)+self.numeric(batch['numeric'])
        z=z+self.set_mlp(torch.cat([batch['sets'],batch['set_mask'].to(z.dtype)],dim=-1))
        z=z.masked_fill(batch['pad'][...,None],0)
        seconds = z
        pad = batch['pad']
        left = (-z.shape[1]) % 32
        if left:
            z = torch.nn.functional.pad(z, (0, 0, left, 0))
            pad = torch.nn.functional.pad(pad, (left, 0), value=True)
        patches=torch.nn.functional.gelu(self.patch(z.reshape(len(z),-1,32*z.shape[-1])))
        valid=~pad.reshape(len(z),-1,32).all(-1)
        patches=patches.masked_fill(~valid[...,None],0)
        recipe_ids=torch.where(drop,2,ids[:,-1,0])
        recipe=self.tables['recipe'](recipe_ids)
        return dict(seconds=seconds,patches=patches,patch_valid=valid,recipe=recipe,dropout_mask=drop)

    def residual_penalty(self):
        return sum(self.tables[n].weight.square().mean() for n in self.residual_names)


class SensorSummary(nn.Module):
    """J: latest/mean/slope/coverage of observed past values, shared across sensors."""
    def __init__(self, sensor_count, dim=64):
        super().__init__();self.sensor_id=nn.Embedding(sensor_count,dim)
        self.mlp=nn.Sequential(nn.Linear(4+dim,dim),nn.GELU(),nn.Linear(dim,dim))
        self.sensor_count=sensor_count

    def forward(self, values):
        if values.ndim!=3 or values.shape[-1]!=self.sensor_count or torch.isinf(values).any():
            raise ValueError('Invalid sensor summary input')
        observed=torch.isfinite(values);x=torch.where(observed,values,0.)
        count=observed.sum(1);den=count.clamp_min(1);mean=x.sum(1)/den
        times=torch.arange(values.shape[1],device=x.device,dtype=x.dtype)[None,:,None]
        mean_t=(times*observed).sum(1)/den
        centered_t=times-mean_t[:,None,:]
        slope=((x-mean[:,None,:])*centered_t*observed).sum(1)/(centered_t.square()*observed).sum(1).clamp_min(1.)
        idx=torch.where(observed,times.long(),-1).amax(1)
        latest=x.gather(1,idx.clamp_min(0)[:,None,:]).squeeze(1)
        stats=torch.stack([latest,mean,slope,count/values.shape[1]],-1)
        sensor=self.sensor_id.weight[None,:,:].expand(len(x),-1,-1)
        encoded=self.mlp(torch.cat([stats,sensor],-1)).masked_fill((count==0)[...,None],0)
        return encoded.sum(1)/(count>0).sum(1).clamp_min(1)[:,None]


class ProcessHeads(nn.Module):
    def __init__(self, profile, dim=64):
        super().__init__();self.sigma_min=profile['duration_sigma_min']
        self.condition=nn.Sequential(nn.Linear(2*dim+len(NUMERIC),dim),nn.GELU(),nn.Linear(dim,dim))
        self.next_query=nn.Linear(dim,dim)
        self.remaining=nn.Linear(dim,2)

    def forward(self, current, numeric, summary, candidates, valid):
        condition=self.condition(torch.cat([current,numeric,summary],-1))
        logits=torch.einsum('bd,bnd->bn',self.next_query(condition),candidates)/math.sqrt(candidates.shape[-1])
        masked=logits.masked_fill(~valid,-torch.inf)
        available=valid.any(1)
        safe=torch.where(available[:,None],masked,torch.zeros_like(masked))
        probabilities=torch.softmax(safe,-1)*valid if safe.shape[1] else safe
        mu,raw_sigma=self.remaining(condition).unbind(-1)
        sigma=torch.nn.functional.softplus(raw_sigma)+self.sigma_min
        return dict(logits=masked,probabilities=probabilities,available=available,mu=mu,sigma=sigma,
                    remaining_seconds=mu.clamp(max=math.log(1e12)).exp().clamp_min(1.))


def auxiliary_losses(output, next_label, remaining, next_valid, remaining_valid):
    # Unknown / censored labels are excluded before arithmetic (including log).
    zero=output['mu'].sum()*0
    valid=next_valid & output['available']
    if valid.any():
        labels=next_label[valid]
        if (labels<0).any() or (labels>=output['logits'].shape[1]).any():raise ValueError('Invalid next-state label')
        if not torch.isfinite(output['logits'][valid].gather(1,labels[:,None])).all():raise ValueError('Label not a legal candidate')
        nxt=nn.functional.cross_entropy(output['logits'][valid],labels)
    else:nxt=zero
    v=remaining_valid & torch.isfinite(remaining) & (remaining>0)
    if v.any():
        log_r=remaining[v].clamp_min(1.).log();mu=output['mu'][v];sigma=output['sigma'][v]
        # Full log-normal NLL (log(y) term included), not elapsed-minus-total regression.
        rem=(log_r+sigma.log()+.5*((log_r-mu)/sigma).square()+.5*math.log(2*math.pi)).mean()
    else:rem=zero
    return dict(next=nxt,remaining=rem,next_count=int(valid.sum()),remaining_count=int(v.sum()))
