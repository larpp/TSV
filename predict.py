"""Forecast one origin from one preprocessed Run. Future sensor observations are unnecessary."""
import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from experiment_common.experiment_utils import device_for
from experiment_common.sensor_selection import source_columns
from experiment_common.sensor_states import StateEncoder, state_sensors
from experiment_common.timestral_covariates.data import numerical, token
from experiment_common.tips_data import numeric, sha
from stepaware.tracker import timestamp_us
from stepaware.training.runner import move
from tsv.compatibility import restore_forecast
from tsv.evaluate import CausalInputs
from tsv.inputs import REFERENCE, REFERENCE_SHA, default_checkpoint
from scripts.prepare_weights import add_mode, prepare_backbone, prepare_checkpoint, prepare_metadata


def read_history(path, model, at, mappings):
    profile, config = model.profile, model.data_config
    roles = {**profile['process_columns'], **profile['audit_columns']}
    aliases = {}
    for mapping in mappings:
        role, sep, name = mapping.partition('=')
        if not sep or role not in roles or not name:
            raise ValueError(f'Expected a process role and CSV name, got {mapping!r}')
        aliases[roles[role]] = name
    names = list(roles.values()) + profile['absolute_set_columns'] + model.targets
    for group in config.get('native_covariates', {}).get('columns', {}).values():
        names.extend(group)
    names = list(dict.fromkeys(names))
    physical = [aliases.get(n, n) for n in names]
    if len(set(physical)) != len(physical):
        raise ValueError('Ambiguous CSV column mapping')
    rows = []
    run, previous = None, None
    selected = set(config['main_steps'])
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        header = source_columns(reader.fieldnames, physical)
        columns = {n: header[p] for n, p in zip(names, physical)}
        for entry in reader:
            tick = timestamp_us(entry[columns['Timestamp']])
            if at is not None and tick > at:
                break
            if previous is not None and tick <= previous:
                raise ValueError('Expected strictly increasing timestamps')
            previous = tick
            row = {n: entry[p] for n, p in columns.items()}
            if not row['_run_id'] or run is not None and run != row['_run_id']:
                raise ValueError('Expected a single Run')
            run = row['_run_id']
            if row['_row_observed'] == '1' and float(row['ProcessRecipeStepSeqNo']) not in selected:
                rows.clear()
            else:
                rows.append(row)
    if not rows:
        raise ValueError('No selected history at the requested origin')
    return rows


def covariate(values, name, vocabulary):
    if name not in vocabulary:
        return numerical(values, name)
    lookup = {v: i for i, v in enumerate(vocabulary[name])}
    return np.asarray([lookup.get(token(v), np.nan) for v in values], dtype=np.float32)


def make_inputs(rows, model, at, schedule):
    config = model.data_config
    ticks = np.asarray([timestamp_us(r['Timestamp']) for r in rows], dtype=np.int64)
    origin = ticks[-1] if at is None else at
    anchor = int((origin - ticks[0]) // 1_000_000)
    if anchor + 1 < config['context'] or origin - ticks[-1] > config['max_age_seconds'] * 1e6:
        raise ValueError('Insufficient recent history for this forecast origin')
    origin = int(ticks[0]) + anchor * 1_000_000
    encoder = StateEncoder(config)
    observed = np.array([r['_row_observed'] == '1' for r in rows])
    values = np.full((len(rows), len(model.targets)), np.nan, dtype=np.float32)
    for j, name in enumerate(model.targets):
        v = [r[name] for r in rows]
        kept = observed & np.array([token(x) is not None for x in v])
        col = [x for x, good in zip(v, kept) if good]
        values[kept, j] = encoder.column(name, col, numeric(col))
        if not np.isfinite(values[kept, j]).all():
            raise ValueError(f'Invalid measurement in {name}')
    segment = dict(origin_us=ticks[0], times=(ticks - ticks[0]) / 1e6, values=values)
    policy = config.get('native_covariates')
    if policy:
        grid = ticks[0] + np.arange(anchor + 1, dtype=np.int64) * 1_000_000
        ix = ticks.searchsorted(grid, side='right') - 1
        stale = grid - ticks[ix] > config['max_age_seconds'] * 1e6
        for role in ('past', 'known'):
            groups = policy['columns']
            cols = groups[role + '_covariates'] + groups['categorical_' + role + '_covariates']
            history = np.empty((len(grid), len(cols)), dtype=np.float32)
            for j, name in enumerate(cols):
                raw = [r[name] if not name.startswith('SubRecipeLoop') or r['_context_valid'] == '1' else '' for r in rows]
                history[:, j] = covariate(raw, name, policy['vocabulary'])[ix]
            history[stale] = np.nan
            if role == 'known' and cols:
                if schedule is None:
                    raise ValueError('This checkpoint requires --schedule with the preannounced future covariates')
                with schedule.open(encoding='utf-8-sig', newline='') as stream:
                    planned = list(csv.DictReader(stream))
                expected = [origin + i * 1_000_000 for i in range(1, config['horizon'] + 1)]
                if [timestamp_us(r['Timestamp']) for r in planned] != expected:
                    raise ValueError('Schedule timestamps must cover each forecast second in order')
                tail = np.column_stack([covariate([r[n] for r in planned], n, policy['vocabulary']) for n in cols])
                history = np.concatenate((history, tail))
            elif role == 'known':
                history = np.empty((len(grid) + config['horizon'], 0), dtype=np.float32)
            segment['cov_' + role] = history
    data = SimpleNamespace(config=config, mean=np.array(model.scaler['mean'], np.float32),
                           std=np.array(model.scaler['std'], np.float32))
    return CausalInputs(segment, rows, model, data).at(anchor), data, origin


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_mode(p)
    p.add_argument('--input', required=True, type=Path, help='One preprocessed Run CSV or its observed prefix')
    p.add_argument('--at', help='Optional naive ISO timestamp; uses the last 1-second grid point at or before it')
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--output', type=Path, default=ROOT / 'outputs/forecast.csv')
    p.add_argument('--column', action='append', default=[], metavar='ROLE=CSV_NAME')
    p.add_argument('--schedule', type=Path, help='Known-future checkpoints only: Timestamp and the declared columns')
    p.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    path = (args.checkpoint or default_checkpoint(ROOT / 'outputs')).resolve()
    prepare_metadata()
    prepare_checkpoint(path, args.pretrained)
    prepare_backbone(args.pretrained)
    checksum = sha(path)
    if path == REFERENCE and checksum != REFERENCE_SHA:
        raise ValueError('Packaged checkpoint hash mismatch')
    state = torch.load(path, map_location='cpu', weights_only=True)
    threshold = state['training'].get('state_threshold', 0.5)
    device = device_for(args.device)
    model = restore_forecast(state, device, checksum=checksum).eval()
    at = timestamp_us(args.at) if args.at else None
    rows = read_history(args.input, model, at, args.column)
    batch, data, origin = make_inputs(rows, model, at, args.schedule)
    raw = model(move(batch, device))['prediction'][0].cpu().numpy() * data.std + data.mean
    if not np.isfinite(raw).all():
        raise ValueError('Nonfinite sensor prediction')
    inputs = {args.input.resolve(), path}
    if args.schedule:
        inputs.add(args.schedule.resolve())
    if args.output.resolve() in inputs:
        raise ValueError('Use a separate CSV for the forecast')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    states = set(state_sensors(data.config))
    base = datetime(1970, 1, 1) + timedelta(microseconds=origin)
    with args.output.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['forecast_origin', 'forecast_time', 'lead_seconds', 'sensor', 'prediction', 'state'])
        for i, point in enumerate(raw, 1):
            for name, value in zip(model.targets, point):
                label = ('OPEN' if value > threshold else 'CLOSE') if name in states else ''
                writer.writerow([base.isoformat(), (base + timedelta(seconds=i)).isoformat(), i, name, float(value), label])
    print(f'{base.isoformat()} | {len(raw)} seconds x {len(model.targets)} sensors | {args.output.resolve()}')


if __name__ == '__main__':
    main()
