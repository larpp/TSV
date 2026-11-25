"""Versioned per-Run memory maps; keep the B sampler without reparsing each Window."""
import json
from pathlib import Path
import numpy as np

from ..contracts import HERE,ROOT,digest,file_sha,key
from ..data import ProcessTimeline
from ..encoding import FeatureCodec
from ..metadata import read_process_rows


class PreparedWindows:
    def __init__(self,base,location,profile,artifact,cache_root,log=None):
        self.base=base;self.config=base.config;self.partition=base.partition;self.run_counts=base.run_counts
        self.artifact=artifact;self.profile=profile;self.arrays=[]
        identity=dict(manifest=location.fingerprint,partition=base.partition,config=base.config,
            segments=[dict(run_id=int(s['run_id']),origin_us=int(s['origin_us']),rows=len(s['grid']),
                anchors=digest(s['anchors'].tolist())) for s in base.segments],
            artifact=artifact['identity'],profile=profile,scaler=dict(mean=base.mean.tolist(),std=base.std.tolist()),
            code={str(p.relative_to(ROOT)):file_sha(p) for p in [HERE/'tracker.py',HERE/'data.py',HERE/'encoding.py',
                HERE/'metadata.py',HERE/'contracts.py',Path(__file__),ROOT/'src/experiment_common/timestral_csv/data.py',
                ROOT/'src/experiment_common/tips_data.py']})
        self.cache=Path(cache_root)/digest(identity)[:24];self.cache.mkdir(parents=True,exist_ok=True)
        self.identity=identity;codec=FeatureCodec(profile,artifact)
        paths={r['run_id']:location.row_path(r) for r in location.rows}
        self.width=max(sum(c['recipe']==r for c in artifact['candidates']) for r in {c['recipe'] for c in artifact['candidates']})
        for i,segment in enumerate(base.segments):
            folder=self.cache/f'segment-{i:04d}';manifest=folder/'complete.json'
            if not manifest.exists():
                folder.mkdir(exist_ok=True)
                timeline=ProcessTimeline(read_process_rows(paths[segment['run_id']],profile),profile,artifact)
                requested=segment['origin_us']+np.rint(segment['grid']*1e6).astype(np.int64)
                states,sets,indices=timeline.context(requested)
                # Cross-check the independent process Set path for overlapping native past inputs.
                native = base.config.get('native_covariates', {}).get('columns', {})
                past = native.get('past_covariates', [])
                for j, name in enumerate(profile['absolute_set_columns']):
                    if name in past and not np.array_equal(sets[:, j], segment['cov_past'][:, past.index(name)], equal_nan=True):
                        raise ValueError(f'Cached Set grid differs for {name}')
                arrays=codec.encode(states,sets);arrays['categories']=arrays['categories'].astype(np.int32)
                n=len(segment['anchors']);arrays.update(candidates=np.zeros((n,self.width,9),np.int32),candidate_count=np.zeros(n,np.int32),
                    next_label=np.full(n,-1,np.int64),next_valid=np.zeros(n,bool),remaining_seconds=np.full(n,np.nan,np.float32),
                    remaining_valid=np.zeros(n,bool),macro=np.zeros(n,np.int32),phase=np.zeros(n,np.int8),last_loop=np.zeros(n,bool))
                macro_ids={key(*c['macro_key']):j for j,c in enumerate(artifact['candidates'])};candidate_cache={}
                for j,anchor in enumerate(segment['anchors']):
                    state=states[anchor]
                    signature=key(state['macro_key'],state['main_id'],state['occurrence_known'],state['counters'],state['quality'],state['terminal'])
                    if signature not in candidate_cache:candidate_cache[signature]=codec.candidates(state)
                    candidates,categories=candidate_cache[signature];count=len(candidates)
                    arrays['candidates'][j,:count]=categories;arrays['candidate_count'][j]=count
                    labels=timeline.labels(indices[anchor],int(requested[anchor]),candidates,state)
                    for k,v in labels.items():arrays[k][j]=v
                    arrays['macro'][j]=macro_ids.get(key(*state['macro_key']),-1)
                    arrays['phase'][j]=['UNKNOWN','MAIN','SUB','MAIN_AFTER_SUB'].index(state['phase'])
                    arrays['last_loop'][j]=bool(state['loop_valid'] and state['k']==state['K'])
                for name,array in arrays.items():
                    with (folder/(name+'.npy.tmp')).open('wb') as f:np.save(f,array,allow_pickle=False)
                    (folder/(name+'.npy.tmp')).replace(folder/(name+'.npy'))
                info=dict(identity=digest(identity),files={name:file_sha(folder/(name+'.npy')) for name in arrays})
                tmp=folder/'complete.json.tmp';tmp.write_text(json.dumps(info),encoding='utf-8');tmp.replace(manifest)
            info=json.loads(manifest.read_text(encoding='utf-8'))
            if info['identity']!=digest(identity):raise ValueError('Process cache identity differs')
            for name,h in info['files'].items():
                if file_sha(folder/(name+'.npy'))!=h:raise ValueError('Process cache file changed; rebuild explicitly: '+str(folder))
            self.arrays.append({name:np.load(folder/(name+'.npy'),mmap_mode='r',allow_pickle=False) for name in info['files']})
            if log and (i==0 or (i+1)%10==0 or i+1==len(base.segments)):
                log.emit('cache_progress',f'{base.partition} process cache {i+1}/{len(base.segments)} Runs')

    def __len__(self):return len(self.base)
    def sampling_weights(self):return self.base.sampling_weights()
    def locate(self,index):
        segment,j=self.base.location(int(index))
        i=int(np.searchsorted(self.base.offsets,int(index),side='right')-1)
        return segment,j,self.arrays[i]
    def info(self,index):
        segment,j,a=self.locate(index);macro=int(a['macro'][j])
        c=self.artifact['candidates'][macro] if macro>=0 else None
        return dict(run_id=int(segment['run_id']),macro=macro,recipe=c['recipe'] if c else '<UNKNOWN>',
                    phase=['UNKNOWN','MAIN','SUB','MAIN_AFTER_SUB'][int(a['phase'][j])],last_loop=bool(a['last_loop'][j]))
    def __getitem__(self,index):
        x,y,_=self.base[int(index)];segment,j,a=self.locate(index);end=int(segment['anchors'][j])+1
        features={k:np.asarray(a[k][end-self.config['context']:end]) for k in ('categories','numeric','sets','set_mask','pad')}
        features['categories']=features['categories'].astype(np.int64)
        n=int(a['candidate_count'][j])
        return dict(x=x,y=y,process=features,candidate_categories=np.asarray(a['candidates'][j,:n],dtype=np.int64),index=int(index),
                    labels={k:a[k][j].item() for k in ('next_label','next_valid','remaining_seconds','remaining_valid')})

    def label_rows(self):
        for index in range(len(self)):
            _,j,a=self.locate(index)
            yield index,self.info(index),{k:a[k][j].item() for k in ('next_label','next_valid','remaining_seconds','remaining_valid')},a['candidates'][j,:int(a['candidate_count'][j])]
