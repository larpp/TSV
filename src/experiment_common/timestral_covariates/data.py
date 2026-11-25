"""Named native covariates on the existing immutable, causal Run windows."""
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from experiment_common.sensor_selection import source_columns
from experiment_common.tips_data import ROOT, StateRunWindows, contained, read_csv

KEY = 'native_covariates'
GROUPS = ('past_covariates', 'known_covariates',
          'categorical_past_covariates', 'categorical_known_covariates')


def columns_from_args(args):
    result = {key: list(getattr(args, key, None) or []) for key in GROUPS}
    seen = set()
    for key, names in result.items():
        for name in names:
            if not isinstance(name, str) or not name.strip() or name != name.strip():
                raise ValueError('Covariate column names must be nonempty with no surrounding whitespace')
            if name in seen:
                raise ValueError('Duplicate Covariate column across inputs: ' + name)
            seen.add(name)
    return result


def new_policy(columns, config):
    return dict(schema_version=1, columns=columns, context=config['context'],
                horizon=config['horizon'], target_count=len(config['sensors']),
                vocabulary={}, fit_partition='train',
                missing='NaN; unseen held-out categories are missing, never refitted',
                alignment='last source observation at or before each grid timestamp',
                known_future_source='recorded future values of user-declared known columns; offline availability assumption')


def validate_policy(policy):
    expected = {'schema_version', 'columns', 'context', 'horizon', 'target_count',
                'vocabulary', 'fit_partition', 'missing', 'alignment', 'known_future_source'}
    if not isinstance(policy, dict) or set(policy) != expected or type(policy['schema_version']) is not int or policy['schema_version'] != 1:
        raise ValueError('Invalid native Covariate policy')
    if set(policy['columns']) != set(GROUPS) or policy['fit_partition'] != 'train':
        raise ValueError('Invalid native Covariate input groups')
    names = []
    for group in GROUPS:
        values = policy['columns'][group]
        if not isinstance(values, list) or any(not isinstance(v,str) or not v.strip() or v != v.strip() for v in values):
            raise ValueError('Invalid Covariate column list')
        names += values
    if not names or len(names) != len(set(names)):
        raise ValueError('Empty or duplicate Covariate policy')
    if any(type(policy[k]) is not int or policy[k] < 1 for k in ('context','horizon','target_count')):
        raise ValueError('Invalid Covariate input dimensions')
    categories = policy['columns']['categorical_past_covariates'] + policy['columns']['categorical_known_covariates']
    if set(policy['vocabulary']) - set(categories):
        raise ValueError('Unexpected categorical vocabulary')
    for values in policy['vocabulary'].values():
        if not isinstance(values,list) or any(not isinstance(v,str) or not v for v in values) or len(values) != len(set(values)):
            raise ValueError('Invalid frozen categorical vocabulary')
    return policy


def missing(value):
    return value is None or str(value).strip().lower() in ('', 'nan', 'null', 'none', 'n/a')


def token(value):
    return None if missing(value) else str(value).strip()


def numerical(values, name):
    result = np.full(len(values), np.nan, dtype=np.float32)
    for i, value in enumerate(values):
        if missing(value):
            continue
        try:
            result[i] = float(value)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(f'Numeric Covariate {name!r} contains {value!r}; use a categorical input for strings') from exc
        if not np.isfinite(result[i]):
            raise ValueError('Nonfinite numeric Covariate: ' + name)
    return result


class CovariateWindows(StateRunWindows):
    """First V columns stay identical; extra columns transport native covariates.

    Known future is packed into extra channels, not target/time rows, keeping
    persistence, metrics, scaler, sampling and existing training code unchanged.
    Only explicitly named known columns may read beyond the forecast origin.
    """
    def __init__(self, split_dir, partition, config, root=ROOT, location=None):
        self.policy = validate_policy(config[KEY])
        super().__init__(split_dir, partition, config, root=root, location=location)
        p = self.policy
        if (p['context'],p['horizon'],p['target_count']) != (config['context'],config['horizon'],len(config['sensors'])):
            raise ValueError('Covariate policy and target window dimensions differ')
        groups = p['columns']
        all_names = [n for g in GROUPS for n in groups[g]]
        categories = groups['categorical_past_covariates'] + groups['categorical_known_covariates']
        records = location.rows if location is not None else read_csv(Path(split_dir)/partition/'runs.csv')
        metadata = {}
        for row in records:
            path = location.row_path(row) if location is not None else contained(root,row['parquet_path'])
            mapped = source_columns(pq.read_schema(path).names, all_names)
            table = pq.read_table(path,columns=list(dict.fromkeys(['_timestamp']+list(mapped.values()))))
            ticks = table['_timestamp'].to_numpy().astype('datetime64[us]').astype(np.int64)
            raw = {n:table[mapped[n]].to_pylist() for n in all_names}
            metadata[int(row['run_id'])] = ticks,raw
        # Fit only on retained, selected Train segments; Valid/Test never extend it.
        if partition == 'train' and not p['vocabulary']:
            vocabulary = {n:set() for n in categories}
            for segment in self.segments:
                ticks,raw = metadata[segment['run_id']]
                first = int(np.searchsorted(ticks,segment['origin_us']))
                end = first+len(segment['times'])
                for name in categories:
                    vocabulary[name].update(t for v in raw[name][first:end] if (t:=token(v)) is not None)
            p['vocabulary'] = {n:sorted(values) for n,values in vocabulary.items()}
        if set(p['vocabulary']) != set(categories):
            raise ValueError('Missing frozen Train categorical vocabulary')
        encoded = {}
        self.covariate_audit = []
        for rid,(ticks,raw) in metadata.items():
            values = {}
            for name in all_names:
                if name in categories:
                    lookup = {v:float(i) for i,v in enumerate(p['vocabulary'][name])}
                    values[name] = np.array([lookup.get(token(v),np.nan) for v in raw[name]],dtype=np.float32)
                else:
                    values[name] = numerical(raw[name],name)
            encoded[rid] = ticks,values
        past = groups['past_covariates'] + groups['categorical_past_covariates']
        known = groups['known_covariates'] + groups['categorical_known_covariates']
        for segment in self.segments:
            ticks,values = encoded[segment['run_id']]
            first = int(np.searchsorted(ticks,segment['origin_us']))
            count = len(segment['times'])
            times = (ticks[first:first+count]-segment['origin_us'])/1e6
            if not np.array_equal(times,segment['times']):
                raise ValueError('Covariate and target source rows differ')
            indices = np.searchsorted(times,segment['grid'],side='right')-1
            for role,names in (('past',past),('known',known)):
                source = np.column_stack([values[n][first:first+count] for n in names]) if names else np.empty((count,0),dtype=np.float32)
                segment['cov_'+role] = source[indices]
                if role == 'known': segment['cov_known_source'] = source
            self.covariate_audit.append(dict(run_id=segment['run_id'],origin_us=segment['origin_us'],
                columns={n:int(np.isnan(values[n][first:first+count]).sum()) for n in all_names}))

    def __getitem__(self,index):
        x,y,i = super().__getitem__(index)
        segment,j = self.location(index)
        end = segment['anchors'][j]+1; start = end-self.config['context']
        past = segment['cov_past'][start:end]
        known_past = segment['cov_known'][start:end]
        width = known_past.shape[1]
        if width:
            requests = segment['grid'][end-1]+np.arange(1,self.config['horizon']+1)*self.config['grid_seconds']
            asof = np.searchsorted(segment['times'],requests,side='right')-1
            future = segment['cov_known_source'][asof].copy()
            stale = (asof < 0) | (requests-segment['times'][asof] > self.config['max_age_seconds'])
            future[stale] = np.nan
            chunks = math.ceil(len(future)/len(x))
            tail = np.full((len(x),chunks*width),np.nan,dtype=np.float32)
            for block in range(chunks):
                piece = future[block*len(x):(block+1)*len(x)]
                tail[:len(piece),block*width:(block+1)*width] = piece
        else:
            tail = np.empty((len(x),0),dtype=np.float32)
        return np.concatenate([x,past,known_past,tail],axis=-1),y,i
