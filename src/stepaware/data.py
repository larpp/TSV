"""Attach causal metadata to the existing CSV windows without changing x/y alignment."""
import copy
from collections import OrderedDict

import numpy as np
import torch

from .contracts import CATEGORIES
from .encoding import FeatureCodec
from .metadata import read_process_rows
from .tracker import StateTracker, timestamp_us


def boundary_labels(states, maximum_gap_seconds=1.5):
    """Future observations are LABELS ONLY. A gap censors both next and duration labels."""
    n=len(states);macro=np.full(n,-1,dtype=np.int64);duration=macro.copy()
    for i in range(n-2,-1,-1):
        a,b=states[i],states[i+1]
        if a['run_id']!=b['run_id'] or a['quality']!='OBSERVED' or b['quality']!='OBSERVED' or a['terminal']:
            continue
        # A raw timestamp gap may exist even without an explicitly inserted missing row.
        if b['timestamp_us']-a['timestamp_us']>maximum_gap_seconds*1e6:continue
        change=a['macro_key']!=b['macro_key']
        macro[i]=i+1 if change else macro[i+1]
        loop_change=a['phase']=='SUB' and a['loop_valid'] and b['loop_valid'] and (a['k']!=b['k'] or a['K']!=b['K'])
        if a['phase']=='SUB' and not a['loop_valid']:continue
        # Changed settings are not a completed Loop duration label.
        if a['phase']=='SUB' and not change:
            if not b['loop_valid'] or a['K']!=b['K']:continue
            if a['k']!=b['k'] and b['k']!=a['k']+1:continue
        duration[i]=i+1 if change or loop_change else duration[i+1]
    return macro,duration


class ProcessTimeline:
    def __init__(self, rows, profile, artifact):
        self.profile=profile
        self.states=StateTracker(profile,artifact['schema'],start_observed=True).parse_run(rows)
        self.ticks=np.array([s['timestamp_us'] for s in self.states],dtype=np.int64)
        self.next_macro,self.duration_end=boundary_labels(self.states,profile['maximum_observation_gap_seconds'])
        self.sets=np.full((len(rows),5),np.nan,dtype=np.float32)
        for i,row in enumerate(rows):
            if str(row[profile['audit_columns']['row_observed']])!='1':continue
            for j,c in enumerate(profile['absolute_set_columns']):
                try:self.sets[i,j]=float(row[c])
                except (TypeError,ValueError):pass
        if np.isinf(self.sets).any():raise ValueError('Infinite Set observation')

    def context(self, requested_us):
        indices=np.searchsorted(self.ticks,requested_us,side='right')-1
        if (indices<0).any():raise ValueError('Insufficient process context: warming_up')
        states=[];sets=self.sets[indices].copy()
        for j,(i,t) in enumerate(zip(indices,requested_us)):
            s=dict(self.states[i]);age=(int(t)-s['timestamp_us'])/1e6
            s['elapsed']=dict(s['elapsed']);s['elapsed_known']=dict(s['elapsed_known'])
            s['timestamp_us']=int(t)
            if age>self.profile['maximum_observation_gap_seconds']:
                s.update(main_seq=None,main_id=None,subrecipe=None,sub_seq=None,sub_id=None,
                         phase='UNKNOWN',quality='MISSING',K_known=False,loop_valid=False,
                         occurrence_known=False,previous_loop_known=False,state_conflict=True,terminal=False)
                s['macro_key']=[s['recipe'],None,'UNKNOWN',None]
                s['elapsed_known']=dict.fromkeys(s['elapsed_known'],False);sets[j]=np.nan
            else:
                for k,known in s['elapsed_known'].items():
                    if known:s['elapsed'][k]+=age
            states.append(s)
        return states,sets,indices

    def labels(self, raw_index, origin_us, candidates, current):
        result=dict(next_label=-1,next_valid=False,remaining_seconds=float('nan'),remaining_valid=False)
        if current['terminal'] or current['quality']!='OBSERVED' or not candidates:return result
        nxt=int(self.next_macro[raw_index]);end=int(self.duration_end[raw_index])
        if nxt>=0 and self.ticks[nxt]>origin_us:
            wanted=self.states[nxt]['macro_key']
            for i,c in enumerate(candidates):
                if c['macro_key']==wanted:result.update(next_label=i,next_valid=True);break
        if end>=0 and self.ticks[end]>origin_us:
            result.update(remaining_seconds=max(1.,(int(self.ticks[end])-int(origin_us))/1e6),remaining_valid=True)
        return result


class ProcessWindows:
    """Small bounded metadata cache; CsvRunWindows still owns sensor/target masks and grids."""
    def __init__(self, base, location, profile, artifact, cache_runs=1):
        self.base=base;self.location=location;self.profile=profile;self.artifact=artifact
        self.codec=FeatureCodec(profile,artifact);self.cache=OrderedDict();self.cache_runs=cache_runs
        self.paths={r['run_id']:location.row_path(r) for r in location.rows}
        if base.config['context']!=128 or base.config['horizon']!=30:raise ValueError('C/P needs the frozen B window contract')
        if base.config['native_covariates']['columns']['past_covariates']!=profile['absolute_set_columns']:
            raise ValueError('Absolute Set and native past Set order must match')
        self.run_counts=base.run_counts;self.config=base.config;self.partition=base.partition

    def __len__(self):return len(self.base)

    def timeline(self, run_id):
        if run_id not in self.cache:
            value=ProcessTimeline(read_process_rows(self.paths[run_id],self.profile),self.profile,self.artifact)
            self.cache[run_id]=value
            while len(self.cache)>self.cache_runs:self.cache.popitem(last=False)
        self.cache.move_to_end(run_id)
        return self.cache[run_id]

    def __getitem__(self,index):
        x,y,_=self.base[index];segment,j=self.base.location(index)
        origin_us=int(segment['origin_us']+round(segment['grid'][segment['anchors'][j]]*1e6))
        requested=origin_us+np.arange(-127,1,dtype=np.int64)*1_000_000
        timeline=self.timeline(segment['run_id']);states,sets,indices=timeline.context(requested)
        # Both native and absolute paths refer to the same actual past observations.
        if not np.array_equal(sets,x[:,-5:],equal_nan=True):
            raise ValueError('Process/Set context is not aligned to the sensor CSV as-of grid')
        candidates,categories=self.codec.candidates(states[-1])
        labels=timeline.labels(indices[-1],origin_us,candidates,states[-1])
        return dict(x=x,y=y,process=self.codec.encode(states,sets),candidate_categories=categories,
                    candidates=candidates,current=states[-1],origin_us=origin_us,
                    labels=labels,index=int(index),source_input_us=timeline.ticks[indices])


def collate_windows(rows, device='cpu'):
    process={k:torch.as_tensor(np.stack([r['process'][k] for r in rows]),device=device) for k in rows[0]['process']}
    width=max(1,max(len(r['candidate_categories']) for r in rows))
    candidates=np.zeros((len(rows),width,len(CATEGORIES)),dtype=np.int64)
    valid=np.zeros((len(rows),width),dtype=bool)
    for i,r in enumerate(rows):
        n=len(r['candidate_categories']);candidates[i,:n]=r['candidate_categories'];valid[i,:n]=True
    return dict(x=torch.as_tensor(np.stack([r['x'] for r in rows]),device=device),
                y=torch.as_tensor(np.stack([r['y'] for r in rows]),device=device),process=process,
                candidate_categories=torch.as_tensor(candidates,device=device),candidate_mask=torch.as_tensor(valid,device=device),
                labels={k:torch.as_tensor([r['labels'][k] for r in rows],device=device) for k in rows[0]['labels']})
