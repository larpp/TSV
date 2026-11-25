"""Canonical CSV views for both the sensor and process readers."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv

from stepaware.model import ProcessSystem  # initializes shared runtime imports
from stepaware.contracts import MODEL, ROOT, digest, file_sha, load_profile
from experiment_common.sensor_selection import DICTIONARY, resolve, source_columns
from experiment_common.step_selection import prepare_steps
from experiment_common.timestral_csv.data import CsvLocation, numeric
from experiment_common.timestral_covariates.data import GROUPS, columns_from_args

HOME = Path(__file__).resolve().parent
REFERENCE = MODEL / 'best.pt'
REFERENCE_SHA = 'ef03e6dad4422cf6205c609e35b7f81508730fb996c3c95891e8a0be6cc640bd'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def default_checkpoint(outputs):
    pointer = outputs / 'latest.json'
    if pointer.exists():
        return outputs / read_json(pointer)['checkpoint'].replace('\\', '/')
    return REFERENCE


def covariate_columns(args, saved=None):
    defaults = saved if saved is not None else read_json(HOME / 'data_config.json')['native_covariates']['columns']
    profile = load_profile()
    roles = {**profile['process_columns'], **profile['audit_columns']}
    aliases = {}
    for item in args.column:
        role, physical_name = item.split('=', 1)
        if role not in roles:
            raise ValueError(f'Unknown process role: {role}')
        aliases[physical_name] = roles[role]
    selected = {}
    for group in GROUPS:
        requested = getattr(args, group, None)
        if requested is not None:
            requested = [aliases.get(name, name) for name in requested]
        if saved is not None and requested is not None and requested != saved[group]:
            raise ValueError(f'{group} must match checkpoint columns and order: {saved[group]}')
        selected[group] = defaults[group] if requested is None else requested
    return columns_from_args(SimpleNamespace(**selected))


def selected_profiles(args, columns=None):
    profile = load_profile()
    sensors = resolve(read_json(args.sensor_profile), read_json(DICTIONARY))['selected_sensors']
    excluded = set(profile['absolute_set_columns'])
    for names in (columns or {}).values():
        excluded.update(names)
    sensors = [s for s in sensors if s not in excluded]
    if not sensors:
        raise ValueError('Select at least one forecast target in sensor_profile.json')
    return sensors, prepare_steps(args.step_profile)['selected_steps']


def location(directory, role, work, columns, steps):
    """Each piece retains its original Run ID. An excluded observed Step breaks the timeline."""
    directory = Path(directory).resolve()
    base = CsvLocation(directory, role)
    profile = load_profile()
    roles = {**profile['process_columns'], **profile['audit_columns']}
    columns = dict(pair.split('=', 1) for pair in columns)
    unknown = columns.keys() - roles.keys()
    if unknown:
        raise ValueError(f'Unknown process roles: {unknown}')
    aliases = {name: columns.get(key, name) for key, name in roles.items()}
    if len(set(aliases.values())) != len(aliases):
        raise ValueError('Each process role requires a distinct CSV column')
    output = work / 'csv' / role
    output.mkdir(parents=True, exist_ok=True)
    rows, paths = [], {}
    origins = {}
    for record in base.rows:
        path = base.row_path(record)
        with path.open(encoding='utf-8-sig', newline='') as fh:
            header = next(csv.reader(fh))
        physical = source_columns(header, list(aliases.values()))
        rename = {physical[v]: k for k, v in aliases.items()}
        canonical = [rename.get(name, name) for name in header]
        if len(set(s.strip() for s in canonical)) != len(canonical):
            raise ValueError(f'Ambiguous canonical header in {path}')
        fields = [physical[aliases[k]] for k in ('Timestamp', '_row_observed', 'ProcessRecipeStepSeqNo')]
        table = pacsv.read_csv(path, convert_options=pacsv.ConvertOptions(
            include_columns=fields, column_types={k: pa.string() for k in fields}))
        ticks = np.asarray(table[fields[0]].to_pylist(), dtype='datetime64[us]').astype(np.int64)
        if len(ticks) != record['output_rows'] or np.any(np.diff(ticks) <= 0):
            raise ValueError(f'Invalid source timeline in {path}')
        observed = np.asarray(table[fields[1]].to_pylist()) == '1'
        keep = ~observed | np.isin(numeric(table[fields[2]].to_pylist()), steps)
        bounds = np.r_[0, np.flatnonzero(keep[1:] != keep[:-1])+1, len(keep)]
        with path.open(encoding='utf-8-sig', newline='') as fh:
            reader = csv.reader(fh)
            next(reader)
            for i, (start, stop) in enumerate(zip(bounds, bounds[1:])):
                if not keep[start]:
                    for _ in range(stop-start):
                        next(reader)
                    continue
                target = path
                key = record['csv_path']
                if len(bounds) > 2 or header != canonical:
                    target = output / f'{path.stem}-part{i}.csv'
                    with target.open('w', encoding='utf-8', newline='') as out:
                        writer = csv.writer(out)
                        writer.writerow(canonical)
                        for _ in range(stop-start):
                            writer.writerow(next(reader))
                    key = f'{role}/{target.name}'
                rows.append(dict(record, csv_path=key, output_rows=int(stop-start)))
                paths[key] = target
                origins[record['run_id'], int(ticks[start])] = target
    return SimpleNamespace(directory=output, source=base.source, rows=rows,
                           row_path=lambda row: paths[row['csv_path']], segment_paths=origins,
                           fingerprint=base.fingerprint, original=base)


def window_options(parser, *, training):
    from experiment_common.sensor_states import validate_threshold

    def positive(value):
        value = int(value)
        if value < 1:
            raise ValueError('Expected a positive number of seconds')
        return value

    capacity = read_json(MODEL / 'pretrained' / 'config.json')['context_length']

    def context_length(value):
        value = positive(value)
        if value > capacity:
            raise ValueError(f'Context exceeds the backbone limit of {capacity}')
        return value

    parser.add_argument('--context', type=context_length, default=128 if training else None)
    parser.add_argument('--horizon', type=int, choices=range(1, 33), default=32 if training else None)
    parser.add_argument('--stride', type=positive, default=32 if training else None)
    parser.add_argument('--state-threshold', type=lambda v: validate_threshold(float(v)),
                        default=0.5 if training else None)


def column_options(parser):
    for group in GROUPS:
        parser.add_argument('--' + group.replace('_', '-'), nargs='*', default=None)
    parser.add_argument('--sensor-profile', type=Path, default=ROOT / 'config/sensor_profile.json')
    parser.add_argument('--step-profile', type=Path, default=ROOT / 'config/step_profile.json')
    parser.add_argument('--column', action='append', default=[])
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs')
