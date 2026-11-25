"""Manifest-bound CSV Runs, causal one-second grids, and NaN observation masks."""
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv

from experiment_common.tips_data import ROOT, RunWindows, contained, sha, numeric
from experiment_common.sensor_selection import resolve, source_columns, DICTIONARY, DEFAULT_PROFILE
from experiment_common.sensor_states import StateEncoder
from experiment_common.timestral_covariates.data import KEY, GROUPS, token, numerical, validate_policy

FEATURE = 'tips-csv-masked-grid-1'


class CsvLocation:
    kind = 'csv_partition'
    def __init__(self, directory, role):
        directory = Path(directory).resolve()
        self.directory = directory / role if (directory/'manifest.json').exists() else directory
        self.source = self.split_root = self.directory.parent
        if self.directory.name != role:
            raise ValueError('CSV partition must match its train/valid/test role')
        manifest = json.loads((self.source/'manifest.json').read_text(encoding='utf-8'))
        if manifest['identity']['policy']['version'] != 'mt-nit-csv-time-gaps-2':
            raise ValueError('Expected timestamp-gap-preserving CSV dataset policy v2')
        self.manifest = manifest
        self.rows = [dict(r, run_id=int(r['run_id'].removeprefix('run_'))) for r in manifest['selected'] if r['split']==role]
        if not self.rows: raise ValueError('Empty CSV partition')
        self.fingerprint = dict(csv_manifest_sha256=sha(self.source/'manifest.json'), source=str(self.source))
        actual = {p.resolve() for p in self.directory.glob('*.csv')}
        expected = {self.row_path(r) for r in self.rows}
        if actual != expected: raise ValueError('CSV files differ from manifest membership')
        for r in self.rows:
            if sha(self.row_path(r)) != manifest['files'][r['csv_path']]: raise ValueError('CSV hash changed: '+r['csv_path'])

    def row_path(self, row):
        path = contained(self.source, row['csv_path'])
        if path.parent != self.directory: raise ValueError('CSV outside selected partition')
        return path

    def snapshot(self):
        return dict(directory=str(self.directory),kind=self.kind,fingerprint=self.fingerprint,
                    run_ids=[r['run_id'] for r in self.rows],
                    source_hashes=[r['source_sha256'] for r in self.rows])


def is_csv_location(path):
    if path is None: return False
    path=Path(path)
    manifest=path/'manifest.json' if (path/'manifest.json').exists() else path.parent/'manifest.json'
    if not manifest.exists(): return False
    return json.loads(manifest.read_text(encoding='utf-8')).get('identity',{}).get('policy',{}).get('version')=='mt-nit-csv-time-gaps-2'


def selection(location, profile_path):
    path=Path(profile_path or DEFAULT_PROFILE)
    profile=json.loads(path.read_text(encoding='utf-8-sig'))
    result=resolve(profile,json.loads(DICTIONARY.read_text(encoding='utf-8-sig')))
    aliases=defaultdict(set)
    for row in location.rows:
        with location.row_path(row).open(encoding='utf-8-sig',newline='') as f: header=next(csv.reader(f))
        for name,physical in source_columns(header,result['selected_sensors']).items(): aliases[name].add(physical)
    result.update(mode='profile',profile_snapshot=profile,profile_sha256=sha(path),dictionary_sha256=sha(DICTIONARY),
                  train_schemas_checked=len(location.rows),source_columns={n:sorted(v) for n,v in aliases.items()})
    return result


def masked_windows(times, values, config):
    """Never compress gaps or use future observations in the context.

    Labels retain the existing first-at/after alignment, including inserted NaN
    rows (never search past them for a later nonmissing label).
    """
    c,h,stride=config['context'],config['horizon'],config['stride']
    grid=np.arange(int(np.floor(times[-1]))+1,dtype=np.float64)
    asof=np.searchsorted(times,grid,side='right')-1
    inputs=values[asof].astype(np.float32).copy()
    inputs[grid-times[asof]>config['max_age_seconds']]=np.nan
    anchors=np.arange(c-1,len(grid)-h,stride,dtype=np.int64)
    requested=grid[anchors,None]+np.arange(1,h+1)[None,:]
    targets=np.searchsorted(times,requested,side='left')
    valid=targets<len(times)
    targets=np.minimum(targets,len(times)-1)
    valid &= times[targets]-requested<=config['target_tolerance_seconds']
    # Each channel needs at least one observed context value; no invented level.
    counts=np.vstack([np.zeros((1,values.shape[1]),dtype=np.int64),np.cumsum(np.isfinite(inputs),axis=0)])
    context_ok=(counts[anchors+1]-counts[anchors-c+1]>0).all(axis=1)
    target_ok=(np.isfinite(values)[targets]&valid[...,None]).any(axis=(1,2))
    keep=context_ok&target_ok
    audit=dict(candidate_windows=len(anchors),excluded_input_windows=int((~context_ok).sum()),
               excluded_target_windows=int((context_ok&~target_ok).sum()),windows=int(keep.sum()))
    return dict(times=times,grid=grid,input_values=inputs,values=values.astype(np.float32),
                anchors=anchors[keep],targets=targets[keep],target_valid=valid[keep]),audit


class CsvRunWindows(RunWindows):
    def __init__(self, split_dir, partition, config, root=ROOT, location=None, **kwargs):
        if config.get('feature_version')!=FEATURE: raise ValueError('CSV needs the saved masked CSV preprocessing contract; retrain or use original Parquet data')
        location=location or CsvLocation(Path(split_dir)/partition,partition)
        self.config=config; self.partition=partition; self.fingerprint=location.fingerprint
        self.segments=[]; self.offsets=[0]; self.audit=[]; self.run_counts=Counter(); self.covariate_audit=[]
        self.mean=np.zeros(len(config['sensors']),dtype=np.float32); self.std=np.ones_like(self.mean)
        encoder=StateEncoder(config,infer=partition=='train' and 'open_close_encoding' not in config)
        policy=validate_policy(config[KEY]) if KEY in config else None
        groups=policy['columns'] if policy else {k:[] for k in GROUPS}
        names=list(dict.fromkeys(n for g in GROUPS for n in groups[g]))
        cats=groups['categorical_past_covariates']+groups['categorical_known_covariates']
        vocab={n:set() for n in cats}
        for row in location.rows:
            path=location.row_path(row)
            with path.open(encoding='utf-8-sig',newline='') as f: header=next(csv.reader(f))
            columns=list(dict.fromkeys(['Timestamp','_row_observed','_context_valid','ProcessRecipeStepSeqNo']+config['sensors']+names))
            mapped=source_columns(header,columns)
            table=pacsv.read_csv(path,convert_options=pacsv.ConvertOptions(include_columns=list(mapped.values()),
                                column_types={v:pa.string() for v in mapped.values()},strings_can_be_null=False))
            raw={n:table[mapped[n]].to_pylist() for n in columns}
            ticks=np.asarray(raw['Timestamp'],dtype='datetime64[us]').astype(np.int64)
            if len(ticks)!=row['output_rows'] or np.any(np.diff(ticks)<=0): raise ValueError('Invalid CSV Run length/time order')
            observed=np.asarray(raw['_row_observed'])=='1'
            steps=numeric(raw['ProcessRecipeStepSeqNo'])
            selected=np.isin(steps,config['main_steps'])
            # Break at explicitly excluded observed Steps, not missing time rows.
            allowed=selected|~observed
            boundaries=np.r_[0,np.flatnonzero(allowed[1:]!=allowed[:-1])+1,len(ticks)]
            values=[]
            for name in config['sensors']:
                col=raw[name]; number=numeric(col)
                missing=np.array([token(v) is None for v in col])|~observed
                finite_values=encoder.column(name,[v for v,m in zip(col,missing) if not m],number[~missing])
                encoded=np.full(len(col),np.nan);encoded[~missing]=finite_values
                if np.any(~np.isfinite(encoded[~missing])): raise ValueError('Invalid sensor value: '+name)
                values.append(encoded)
            values=np.column_stack(values)
            audit=dict(split=partition,run_id=row['run_id'],source_rows=len(ticks),observed_rows=int(observed.sum()),
                       missing_sensor_cells=int((~np.isfinite(values)).sum()),windows=0,candidate_windows=0,
                       excluded_input_windows=0,excluded_target_windows=0,skipped_step_segments=0)
            for first,end in zip(boundaries[:-1],boundaries[1:]):
                if not allowed[first]:audit['skipped_step_segments']+=1;continue
                times=(ticks[first:end]-ticks[first])/1e6
                data,counts=masked_windows(times,values[first:end],config)
                for k,v in counts.items():audit[k]+=v
                if not counts['windows']:continue
                data.update(run_id=row['run_id'],origin_us=int(ticks[first]),main_step=-1,
                            main_steps=np.where(np.isfinite(steps[first:end]),steps[first:end],-1).astype(np.int16),
                            continuous_segment=len(self.segments)+1)
                if policy:
                    data['cov_raw']={n:raw[n][first:end] for n in names}
                    # Invalid Loop metadata cannot be used as valid loop covariates.
                    for name in names:
                        if name.startswith('SubRecipeLoop'):
                            data['cov_raw'][name]=[v if ok=='1' else '' for v,ok in zip(data['cov_raw'][name],raw['_context_valid'][first:end])]
                    for name in cats:vocab[name].update(t for v in data['cov_raw'][name] if (t:=token(v)) is not None)
                self.segments.append(data);self.offsets.append(self.offsets[-1]+counts['windows'])
                self.run_counts[row['run_id']]+=counts['windows']
            self.audit.append(audit)
        encoder.freeze()
        if not len(self):raise ValueError('No eligible CSV windows: need observed context per sensor and at least one observed target')
        if policy:
            if partition=='train' and not policy['vocabulary']:policy['vocabulary']={n:sorted(v) for n,v in vocab.items()}
            if set(policy['vocabulary'])!=set(cats):raise ValueError('Missing frozen Train covariate vocabulary')
            for s in self.segments:
                values={}
                for n in names:
                    if n in cats:
                        lookup={v:float(i) for i,v in enumerate(policy['vocabulary'][n])}
                        values[n]=np.array([lookup.get(token(v),np.nan) for v in s['cov_raw'][n]],dtype=np.float32)
                    else:values[n]=numerical(s['cov_raw'][n],n)
                asof=np.searchsorted(s['times'],s['grid'],side='right')-1
                for role,cols in [('past',groups['past_covariates']+groups['categorical_past_covariates']),('known',groups['known_covariates']+groups['categorical_known_covariates'])]:
                    src=np.column_stack([values[n] for n in cols]) if cols else np.empty((len(s['times']),0),dtype=np.float32)
                    s['cov_'+role]=src[asof];s['cov_'+role][s['grid']-s['times'][asof]>config['max_age_seconds']]=np.nan
                    if role=='known':s['cov_known_source']=src
                del s['cov_raw']
                self.covariate_audit.append(dict(run_id=s['run_id'],missing={n:int(np.isnan(v).sum()) for n,v in values.items()}))

    def __getitem__(self,index):
        x,y,i=super().__getitem__(index)
        s,j=self.location(index);y=y.copy();y[~s['target_valid'][j]]=np.nan
        if KEY not in self.config:return x,y,i
        import math
        end=s['anchors'][j]+1;start=end-self.config['context']
        past=s['cov_past'][start:end];known=s['cov_known'][start:end];width=known.shape[1]
        chunks=math.ceil(self.config['horizon']/len(x))
        tail=np.full((len(x),chunks*width),np.nan,dtype=np.float32)
        if width:
            requests=s['grid'][end-1]+np.arange(1,self.config['horizon']+1)
            asof=np.searchsorted(s['times'],requests,side='right')-1
            future=s['cov_known_source'][asof].copy()
            future[requests-s['times'][asof]>self.config['max_age_seconds']]=np.nan
            for block in range(chunks):
                piece=future[block*len(x):(block+1)*len(x)]
                tail[:len(piece),block*width:(block+1)*width]=piece
        return np.concatenate([x,past,known,tail],axis=-1),y,i
