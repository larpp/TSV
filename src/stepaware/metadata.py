"""Fit vocabulary, candidate metadata and absolute Set statistics on Train only."""
import copy
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv

from .contracts import CATEGORIES, RESERVED, digest, file_sha, key
from .tracker import StateTracker, category_values, parse_fields


def read_process_rows(path, profile):
    columns=list(dict.fromkeys(list(profile['process_columns'].values())+
                              list(profile['audit_columns'].values())+profile['absolute_set_columns']))
    table=pacsv.read_csv(path, convert_options=pacsv.ConvertOptions(
        include_columns=columns, column_types={c:pa.string() for c in columns}, strings_can_be_null=False))
    arrays={n:table[n].to_pylist() for n in columns}
    return [dict(zip(columns, values)) for values in zip(*(arrays[n] for n in columns))]


def fit_metadata(location, profile, progress=None):
    if location.directory.name != 'train':
        raise ValueError('Process vocabulary and Set scaler must be fitted on Train only')
    schema=dict(start_keys=[],end_keys=[],substep_keys=[])
    starts=set();ends=set();subs=set();counts=np.zeros(5);sums=np.zeros(5);squares=np.zeros(5)
    sources={};cached=[]
    # First pass identifies stale-entry field combinations from TRAIN evidence only.
    for i, row in enumerate(location.rows):
        path=location.row_path(row);rows=read_process_rows(path,profile)
        observed=[r for r in rows if r[profile['audit_columns']['row_observed']]=='1']
        if not observed:continue
        first=parse_fields(observed[0],profile);last=parse_fields(observed[-1],profile)
        if (first['main_name'] or '').upper()=='START':
            starts.add(key(first['main_recipe'],first['main_step_seq'],first['main_step_id']))
        if (last['main_name'] or '').upper()=='END':
            ends.add(key(last['main_recipe'],last['main_step_seq'],last['main_step_id']))
        for raw in observed:
            d=parse_fields(raw,profile)
            if d['execution_recipe'] and d['execution_recipe'].startswith('SubRecipe/'):
                if (d['substep_seq'],d['substep_id']) != (d['main_step_seq'],d['main_step_id']) and d['substep_seq'] is not None and d['substep_id'] is not None:
                    subs.add(key(d['execution_recipe'],d['substep_seq'],d['substep_id']))
            for j,c in enumerate(profile['absolute_set_columns']):
                try:v=float(raw[c])
                except (ValueError,TypeError):continue
                if np.isfinite(v):counts[j]+=1;sums[j]+=v;squares[j]+=v*v
        sources[row['csv_path']]=file_sha(path)
        if progress:progress('schema',i+1,len(location.rows))
    if (counts==0).any():raise ValueError('Set column has no observed Train values')
    mean=sums/counts;std=np.sqrt(np.maximum(0.,squares/counts-mean*mean));std[std<1e-8]=1.
    schema.update(start_keys=sorted(starts),end_keys=sorted(ends),substep_keys=sorted(subs))
    vocab={n:set() for n in CATEGORIES};candidates={}
    for i,row in enumerate(location.rows):
        tracker=StateTracker(profile,schema,start_observed=True)
        for raw in read_process_rows(location.row_path(row),profile):
            s=tracker.update(raw)
            for n,v in category_values(s).items():vocab[n].add(v)
            if s['quality']=='OBSERVED' and s['main_seq'] is not None and s['phase']!='UNKNOWN':
                k=key(*s['macro_key'])
                candidate=dict(recipe=s['recipe'],main_seq=s['main_seq'],main_id=s['main_id'],phase=s['phase'],
                               subrecipe=s['subrecipe'],terminal=s['terminal'],macro_key=s['macro_key'])
                if k in candidates and candidates[k]!=candidate:raise ValueError('Ambiguous Train macro metadata')
                candidates[k]=candidate
        if progress:progress('vocabulary',i+1,len(location.rows))
    # Semantic quality and phase enums are not learned IDs; unseen raw keys stay UNKNOWN.
    vocab['quality'].update(['OBSERVED','UNKNOWN','MISSING'])
    vocab['phase'].update(['MAIN','SUB','MAIN_AFTER_SUB'])
    artifact=dict(version=1,fit_partition='train',schema=schema,
        vocabulary={n:RESERVED+sorted(v-set(RESERVED)) for n,v in vocab.items()},
        candidates=[candidates[k] for k in sorted(candidates)],
        set_scaler=dict(mean=mean.tolist(),std=std.tolist(),observations=counts.astype(int).tolist(),fit_partition='train'),
        train_sources=sources,profile_sha256=digest(profile))
    artifact['identity']=digest(artifact)
    return artifact


def validate_metadata(artifact, profile):
    a=copy.deepcopy(artifact);identity=a.pop('identity',None)
    if identity!=digest(a) or artifact['fit_partition']!='train' or artifact['profile_sha256']!=digest(profile):
        raise ValueError('Process metadata identity/Train/profile differs')
    for n in CATEGORIES:
        v=artifact['vocabulary'][n]
        if v[:3]!=RESERVED or len(v)!=len(set(v)):raise ValueError('Invalid category vocabulary')
    return artifact


def candidate_states(current, artifact):
    """Recipe candidates and occurrence codes depend only on the current tracker snapshot."""
    if current['terminal'] or current['quality']=='MISSING' or current['phase']=='UNKNOWN':return []
    if not any(c['macro_key']==current['macro_key'] and c['main_id']==current['main_id']
               for c in artifact['candidates']):return []
    result=[]
    for c in artifact['candidates']:
        if c['recipe']!=current['recipe'] or c['macro_key']==current['macro_key']:continue
        s=dict(current,**c,quality='OBSERVED',sub_seq=None,sub_id=None)
        if current['occurrence_known']:
            count=current['counters'].get(str(c['main_id']),0)
            same_entry=c['main_seq']==current['main_seq'] and c['main_id']==current['main_id']
            s['occurrence']=count if same_entry else count+1
        else:s['occurrence']=None
        result.append(s)
    return result
