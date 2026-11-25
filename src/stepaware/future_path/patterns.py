"""Streaming completed-loop observations; Train-only median pattern/transition bank."""
import copy
from collections import Counter,defaultdict
import numpy as np
from ..contracts import key,digest


def pattern_keys(s):
    return [key('position',s['recipe'],s['main_seq'],s['subrecipe']),
            key('occurrence',s['recipe'],s['main_id'],s['occurrence'],s['subrecipe']),
            key('shared',s['main_id'],s['occurrence'],s['subrecipe'])]


class LoopObserver:
    def __init__(self,max_gap=1.5):
        self.max_gap=max_gap;self.previous=None;self.parts=[];self.start=None;self.clean=False;self.recent=None
    def update(self,s):
        old=self.previous;completed=None
        contiguous=bool(old and s['run_id']==old['run_id'] and 0<(s['timestamp_us']-old['timestamp_us'])/1e6<=self.max_gap
                        and old['quality']=='OBSERVED' and s['quality']=='OBSERVED')
        same=bool(old and s['macro_key']==old['macro_key'])
        good=s['phase']=='SUB' and s['quality']=='OBSERVED' and s['loop_valid'] and s['sub_id'] is not None
        boundary=bool(old and old['phase']=='SUB' and (not same or s['k']!=old['k']))
        legal=bool(boundary and contiguous and old['loop_valid'] and
            ((same and s['loop_valid'] and s['K']==old['K'] and s['k']==old['k']+1) or
             (not same and s['phase']!='SUB' and old['k']==old['K'])))
        if legal and self.clean and self.parts:
            parts=copy.deepcopy(self.parts);parts[-1]['duration']=(s['timestamp_us']-parts[-1].pop('start'))/1e6
            if all(p['duration']>0 for p in parts):
                completed=dict(keys=pattern_keys(self.start),steps=parts,duration=sum(p['duration'] for p in parts),
                    completed_at_us=s['timestamp_us'],source='observed_completed_loop')
                self.recent=completed
        if not contiguous or not same or not good or (old and same and s['K']!=old['K']):self.recent=None
        if not good:
            self.parts=[];self.clean=False;self.start=None
        elif boundary or not self.parts or not same:
            self.start=copy.deepcopy(s);self.clean=bool(contiguous and s['elapsed_known']['loop'] and s['elapsed']['loop']==0)
            self.parts=[dict(seq=s['sub_seq'],id=s['sub_id'],start=s['timestamp_us'])]
        elif contiguous and s['K']==old['K']:
            if (s['sub_seq'],s['sub_id'])!=(old['sub_seq'],old['sub_id']):
                self.parts[-1]['duration']=(s['timestamp_us']-self.parts[-1].pop('start'))/1e6
                self.parts.append(dict(seq=s['sub_seq'],id=s['sub_id'],start=s['timestamp_us']))
        else:self.clean=False
        self.previous=copy.deepcopy(s)
        return completed


class BankBuilder:
    def __init__(self):self.patterns=defaultdict(list);self.edges=defaultdict(Counter);self.previous=None
    def add(self,s,completed):
        old=self.previous
        if old and old['run_id']==s['run_id'] and old['quality']==s['quality']=='OBSERVED' and 0<(s['timestamp_us']-old['timestamp_us'])<=1_500_000 and old['macro_key']!=s['macro_key']:
            self.edges[key(*old['macro_key'])][key(*s['macro_key'])]+=1
        self.previous=s
        if completed:
            for k in completed['keys']:self.patterns[k].append(completed['steps'])
    def finish(self,train_sources):
        patterns={}
        for k,records in self.patterns.items():
            signatures=[key(*[(p['seq'],p['id']) for p in record]) for record in records]
            chosen,count=Counter(signatures).most_common(1)[0]
            selected=[r for r,sig in zip(records,signatures) if sig==chosen]
            steps=[dict(seq=p['seq'],id=p['id'],duration=float(np.median([r[i]['duration'] for r in selected]))) for i,p in enumerate(selected[0])]
            patterns[k]=dict(steps=steps,duration=sum(p['duration'] for p in steps),source='train_median',
                             count=count,total_observed_loops=len(records),signature_count=len(set(signatures)))
        bank=dict(version=1,fit_partition='train',train_sources=train_sources,patterns=patterns,
                  edges={k:dict(v) for k,v in self.edges.items()})
        bank['identity']=digest(bank);return bank


def validate_bank(bank):
    b=copy.deepcopy(bank);identity=b.pop('identity',None)
    if b.get('fit_partition')!='train' or identity!=digest(b):raise ValueError('Invalid Train pattern bank')
    return bank


def choose_pattern(current,recent,bank):
    if recent and recent['completed_at_us']<=current['timestamp_us'] and recent['keys'][0]==pattern_keys(current)[0]:
        return recent
    for k in pattern_keys(current):
        p=bank['patterns'].get(k)
        if p:return p
    return None
