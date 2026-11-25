"""Past-only snapshots for the process renderer."""
import copy
import numpy as np
from ..training.runner import collate as c_collate
from .patterns import LoopObserver


def snapshots(timeline,segment,anchors):
    requested=segment['origin_us']+np.rint(segment['grid'][anchors]*1e6).astype(np.int64)
    states,_,indices=timeline.context(requested);observer=LoopObserver();cursor=0;result=[]
    for state,index in zip(states,indices):
        while cursor<=index:
            observer.update(timeline.states[cursor]);cursor+=1
        recent=copy.deepcopy(observer.recent) if state['quality']=='OBSERVED' else None
        result.append(dict(current=state,recent=recent))
    return result


class RenderWindows:
    def __init__(self,prepared,records):
        self.prepared=prepared
        for n in ('base','config','partition','run_counts','artifact','profile','identity'):setattr(self,n,getattr(prepared,n))
        self.records=[{int(a):s for a,s in zip(r['anchors'],r['snapshots'])} for r in records]
    def __len__(self):return len(self.prepared)
    def info(self,index):return self.prepared.info(index)
    def sampling_weights(self):return self.prepared.sampling_weights()
    def __getitem__(self,index):
        row=self.prepared[index];segment,j,_=self.prepared.locate(index)
        i=int(np.searchsorted(self.base.offsets,index,side='right')-1)
        row['causal']=copy.deepcopy(self.records[i][int(segment['anchors'][j])]);return row


def collate(rows):
    batch=c_collate(rows);batch['causal']=[r['causal'] for r in rows];return batch
