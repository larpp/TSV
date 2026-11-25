"""Point-in-time windows on immutable TIPS Run partitions; no future interpolation."""
from bisect import bisect_right
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT = 'etc/data/hd02_splits/d1c591fbc818b3b2'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def contained(root, relative):
    p = Path(relative)
    if p.is_absolute() or '..' in p.parts: raise ValueError('Expected safe relative path')
    resolved = (Path(root) / p).resolve()
    if not resolved.is_relative_to(Path(root).resolve()): raise ValueError('Path escapes root')
    return resolved


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))


def verify_split(split_dir, root=ROOT):
    split_dir = Path(split_dir).resolve()
    m = json.loads((split_dir / 'manifest.json').read_text(encoding='utf-8'))
    for name, entry in m['files'].items():
        if sha(contained(split_dir, name)) != entry['sha256']: raise ValueError('Split artifact changed: ' + name)
    source = contained(root, m['identity']['source'])
    if sha(source / 'manifest.json') != m['identity']['source_manifest_sha256']:
        raise ValueError('Source manifest changed')
    source_m = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    assignments = read_csv(split_dir / 'assignments.csv')
    ids = [int(r['run_id']) for r in assignments]
    if len(ids) != len(set(ids)): raise ValueError('Run appears in multiple partitions')
    for name in ['train', 'valid', 'test', 'reserved_transfer']:
        expected = [r for r in assignments if r['split'] == name]
        if read_csv(split_dir / name / 'runs.csv') != expected: raise ValueError('Partition membership mismatch')
    return {'split_manifest_sha256': sha(split_dir / 'manifest.json'),
            'source_manifest_sha256': m['identity']['source_manifest_sha256'],
            'source': source.relative_to(root).as_posix()}, source, source_m


def numeric(values):
    def convert(x):
        try: return float(x)
        except (TypeError, ValueError): return np.nan
    return np.array([convert(x) for x in values], dtype=np.float64)


def segment_windows(times, values, context, horizon, stride, grid_seconds=1.0,
                    max_age=3.0, target_tolerance=1.5):
    """Grid input uses last value already observed; labels use actual future observations.

    Relative float seconds avoid epoch precision loss. Input length L means L grid
    points spanning (L-1)*grid_seconds; prediction length H means +1..H grid steps.
    """
    if min(context, horizon, stride) < 1 or min(grid_seconds, max_age, target_tolerance) <= 0:
        raise ValueError('Window settings must be positive')
    if len(times) != len(values) or not len(times) or np.any(np.diff(times) <= 0):
        raise ValueError('Invalid segment time order/length')
    grid = np.arange(int(np.floor((times[-1] - times[0]) / grid_seconds)) + 1) * grid_seconds + times[0]
    asof = np.searchsorted(times, grid, side='right') - 1
    causal = values[asof]
    input_bad = (~np.isfinite(causal).all(axis=1)) | (grid - times[asof] > max_age)
    cumulative_bad = np.r_[0, np.cumsum(input_bad)]
    anchors = np.arange(context - 1, len(grid), stride)
    # Future availability is checked only for constructing/evaluating labels,
    # never for computing the causal input grid or its normalization.
    input_ok = cumulative_bad[anchors + 1] - cumulative_bad[anchors - context + 1] == 0
    requests = grid[anchors, None] + np.arange(1, horizon + 1)[None, :] * grid_seconds
    target = np.searchsorted(times, requests, side='left')
    inside = target < len(times)
    clipped = np.minimum(target, len(times) - 1)
    target_ok = (inside & (times[clipped] - requests <= target_tolerance)
                 & np.isfinite(values).all(axis=1)[clipped]).all(axis=1)
    keep = input_ok & target_ok
    audit = dict(candidate_windows=len(anchors), excluded_input_windows=int((~input_ok).sum()),
                 excluded_target_windows=int((input_ok & ~target_ok).sum()), windows=int(keep.sum()))
    return dict(grid=grid, input_values=causal.astype(np.float32),
                values=values.astype(np.float32), times=times, anchors=anchors[keep],
                targets=target[keep]), audit


class RunWindows(Dataset):
    def __init__(self, split_dir, partition, config, root=ROOT, detect_open_close=False, location=None):
        if partition not in ['train', 'valid', 'test']: raise ValueError('Reserved data are not a training/evaluation partition')
        if detect_open_close and partition != 'train': raise ValueError('Sensor types may be inferred from Train only')
        encoder = None
        if detect_open_close or 'open_close_encoding' in config:
            from .sensor_states import StateEncoder
            encoder = StateEncoder(config, infer=detect_open_close)
        self.config = config; self.partition = partition
        self.segments = []; self.offsets = [0]; self.audit = []
        self.mean = np.zeros(len(config['sensors']), dtype=np.float32)
        self.std = np.ones(len(config['sensors']), dtype=np.float32)
        if location is None:
            fingerprint, source, source_m = verify_split(split_dir, root)
            rows = read_csv(Path(split_dir) / partition / 'runs.csv')
        else:
            fingerprint,source,source_m = location.fingerprint,location.source,location.source_manifest
            rows = location.rows
        self.fingerprint = fingerprint
        self.run_counts = Counter()
        for row in rows:
            rid = int(row['run_id'])
            path = location.row_path(row) if location is not None else contained(root, row['parquet_path'])
            rel = path.relative_to(source).as_posix()
            if sha(path) != source_m['files'][rel]['sha256']: raise ValueError('Run Parquet changed')
            from .sensor_selection import source_columns
            schema = pq.read_schema(path)
            mapped = source_columns(schema.names, config['sensors'])
            run_scope = config.get('window_scope', 'main_step') == 'run'
            required = ['_run_id', '_timestamp', '_continuous_segment', 'ProcessRecipeStepSeqNo'] + list(mapped.values())
            if run_scope: required += ['_original_row_index', '_cycle_id']
            missing = set(required) - set(schema.names)
            if missing: raise ValueError(f'Run {rid} missing columns: {sorted(missing)}')
            table = pq.read_table(path, columns=required)
            if len(table) != int(row['rows_after']): raise ValueError('Run row count mismatch')
            run_ids = table['_run_id'].to_numpy()
            if np.any(run_ids != rid): raise ValueError('Parquet Run identity mismatch')
            ticks = table['_timestamp'].to_numpy().astype('datetime64[us]').astype(np.int64)
            if np.any(np.diff(ticks) <= 0): raise ValueError('Run timestamps must strictly increase')
            segments = table['_continuous_segment'].to_numpy()
            steps = numeric(table['ProcessRecipeStepSeqNo'].to_pylist())
            if not np.isfinite(steps).all() or np.any(steps != np.floor(steps)) or np.any((steps < 0) | (steps > 255)):
                raise ValueError('Invalid Main Step sequence number')
            if encoder is None:
                values = np.column_stack([numeric(table[mapped[c]].to_pylist()) for c in config['sensors']])
            else:
                columns = []
                for name in config['sensors']:
                    raw = table[mapped[name]].to_pylist()
                    columns.append(encoder.column(name, raw, numeric(raw)))
                values = np.column_stack(columns)
            if run_scope:
                original = table['_original_row_index'].to_numpy()
                if np.any(np.diff(original) <= 0): raise ValueError('Invalid original row order')
                selected = np.isin(steps, config['main_steps'])
                # _continuous_segment also splits at every Main Step. Rebuild only
                # physical continuity so genuine Step/Subcycle transitions remain.
                cuts = ((np.diff(original) != 1) | (np.diff(ticks) > 3_000_000)
                        | (selected[1:] != selected[:-1]))
                boundaries = np.r_[0, np.flatnonzero(cuts) + 1, len(table)]
                cycle_ids = table['_cycle_id'].to_pylist()
            else:
                boundaries = np.r_[0, np.flatnonzero(segments[1:] != segments[:-1]) + 1, len(table)]
            audit = dict(split=partition, run_id=rid, source_rows=len(table), selected_rows=0,
                         skipped_step_segments=0, too_short_segments=0, candidate_windows=0,
                         excluded_input_windows=0, excluded_target_windows=0, windows=0)
            if run_scope:
                audit.update(observed_main_steps=sorted(set(map(int, steps))), subcycle_rows=0,
                             cross_main_step_windows=0, subcycle_target_windows=0)
            for first, end in zip(boundaries[:-1], boundaries[1:]):
                if not run_scope and np.any(steps[first:end] != steps[first]):
                    raise ValueError('Unidentified/changing Main Step inside a continuous segment')
                step = int(steps[first])
                if config['main_steps'] and step not in config['main_steps']:
                    audit['skipped_step_segments'] += 1; continue
                audit['selected_rows'] += int(end - first)
                if run_scope: audit['subcycle_rows'] += sum(c is not None for c in cycle_ids[first:end])
                times = (ticks[first:end] - ticks[first]) / 1e6
                if np.any(np.diff(times) > 3.0): raise ValueError('Unexpected >3s gap inside a continuous segment')
                data, counts = segment_windows(times, values[first:end], config['context'], config['horizon'],
                    config['stride'], config['grid_seconds'], config['max_age_seconds'], config['target_tolerance_seconds'])
                for k, v in counts.items(): audit[k] += v
                if not counts['windows']:
                    if end - first < config['context'] or times[-1] < (config['context'] - 1 + config['horizon']) * config['grid_seconds']:
                        audit['too_short_segments'] += 1
                    continue
                data.update(run_id=rid, main_step=step, continuous_segment=int(segments[first]), origin_us=int(ticks[first]))
                if run_scope:
                    data['main_steps'] = steps[first:end].astype(np.int16)
                    subcycle = np.array([c is not None for c in cycle_ids[first:end]])
                    # Count whether any original Step boundary occurs across input+target.
                    changes = np.r_[0, np.cumsum(np.diff(data['main_steps']) != 0)]
                    input_first = data['anchors'] - config['context'] + 1
                    input_source = np.searchsorted(times, data['grid'][input_first], side='right') - 1
                    audit['cross_main_step_windows'] += int(np.sum(changes[data['targets'][:, -1]] != changes[input_source]))
                    audit['subcycle_target_windows'] += int(subcycle[data['targets']].any(axis=1).sum())
                self.segments.append(data)
                self.offsets.append(self.offsets[-1] + counts['windows'])
                self.run_counts[rid] += counts['windows']
            self.audit.append(audit)
        if encoder is not None: encoder.freeze()
        if not len(self):
            raise ValueError(f'No eligible {partition} Windows for selected sensors/Steps/context/horizon; inspect data or shorten context')

    def __len__(self): return self.offsets[-1]

    def location(self, index):
        s = bisect_right(self.offsets, int(index)) - 1
        return self.segments[s], int(index) - self.offsets[s]

    def __getitem__(self, index):
        s, j = self.location(index); end = s['anchors'][j] + 1
        x = s['input_values'][end - self.config['context']:end]
        y = s['values'][s['targets'][j]]
        return ((x - self.mean) / self.std, (y - self.mean) / self.std, int(index))

    def fit_scaler(self):
        if self.partition != 'train': raise ValueError('Only train may fit preprocessing')
        # Count each original training observation once, not each overlapping Window.
        values = np.concatenate([s['values'] for s in self.segments]).astype(np.float64)
        finite = np.isfinite(values)
        count = finite.sum(axis=0)
        if np.any(count == 0): raise ValueError('A sensor has no finite training observations')
        clean = np.where(finite, values, 0.0)
        mean = clean.sum(axis=0) / count
        variance = np.where(finite, (values - mean) ** 2, 0.0).sum(axis=0) / count
        std = np.sqrt(variance); std[std < 1e-8] = 1.0
        return dict(mean=mean.tolist(), std=std.tolist(), observations=count.tolist(), fit_partition='train')

    def set_scaler(self, state):
        self.mean = np.asarray(state['mean'], dtype=np.float32)
        self.std = np.asarray(state['std'], dtype=np.float32)
        if self.mean.shape != (len(self.config['sensors']),) or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or np.any(self.std <= 0):
            raise ValueError('Invalid frozen scaler')

    def sampling_weights(self):
        return np.concatenate([np.full(len(s['anchors']), 1.0 / self.run_counts[s['run_id']], dtype=np.float64) for s in self.segments])


class StateRunWindows(RunWindows):
    """Timestral opt-in: infer types on Train, reuse frozen types on Valid/Test."""
    def __init__(self, split_dir, partition, config, root=ROOT, location=None):
        super().__init__(split_dir, partition, config, root,
                         detect_open_close=partition == 'train' and 'open_close_encoding' not in config,
                         location=location)
