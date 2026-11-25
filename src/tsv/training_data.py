"""Fit preprocessing on selected Train observations and align Step/Loop with each CSV segment."""
import copy
from .inputs import HOME, read_json, selected_profiles, location, covariate_columns
from experiment_common.timestral_covariates.data import new_policy
from stepaware.contracts import digest, load_profile
from stepaware.metadata import fit_metadata, read_process_rows
from stepaware.data import ProcessTimeline
from stepaware.training.prepared_data import PreparedWindows
from stepaware.training.process_metrics import fit_baselines
from stepaware.future_path.data import RenderWindows, snapshots
from stepaware.future_path.patterns import BankBuilder, LoopObserver
from experiment_common.timestral_csv.data import CsvRunWindows
from experiment_common.experiment_utils import write_json
from stepaware.policy_model import make_policy


def prepare(args, output, log):
    columns = covariate_columns(args)
    sensors, steps = selected_profiles(args, columns)
    profile = dict(load_profile(), context=args.context, horizon=args.horizon)
    config = read_json(HOME / 'data_config.json')
    config.pop('open_close_encoding')
    config.update(sensors=sensors, main_steps=steps, context=args.context, horizon=args.horizon, stride=args.stride,
                  normalization_policy=make_policy('A1-005', sensors, args.context, args.horizon))
    if any(columns.values()):
        config['native_covariates'] = new_policy(columns, config)
        log.emit('covariates', f'Native covariate columns: {columns}')
        if columns['known_covariates'] or columns['categorical_known_covariates']:
            log.emit('known_future_assumption', config['native_covariates']['known_future_source'])
    else:
        config.pop('native_covariates')
    locations = {role: location(path, role, output, args.column, steps)
                 for role, path in [('train', args.train_data), ('valid', args.valid_data)]}
    sources = [{row['source_sha256'] for row in loc.original.rows} for loc in locations.values()]
    if sources[0] & sources[1]:
        raise ValueError('Train and Valid contain the same source Run')
    train = locations['train']
    artifact = fit_metadata(train, profile, lambda phase, i, n: log.emit('metadata', f'{phase} {i}/{n}') if i % 10 == 0 else None)
    prepared, rendered = {}, {}
    builder = BankBuilder()
    for role, loc in locations.items():
        base = CsvRunWindows(loc.source, role, config, location=loc)
        if role == 'train':
            scaler = base.fit_scaler()
        base.set_scaler(scaler)
        records, parts = [], []
        rows = {loc.row_path(r): r for r in loc.rows}
        for segment in base.segments:
            path = loc.segment_paths[segment['run_id'], segment['origin_us']]
            # One process timeline per contiguous selected segment. Global sampler weights use the original Run.
            single = copy.copy(base)
            single.segments = [segment]
            single.offsets = [0, len(segment['anchors'])]
            piece = copy.copy(loc)
            piece.rows = [rows[path]]
            block = PreparedWindows(single, piece, profile, artifact, output / 'cache')
            parts.append(block.arrays[0])
            timeline = ProcessTimeline(read_process_rows(path, profile), profile, artifact)
            if role == 'train':
                observer = LoopObserver()
                builder.previous = None
                for state in timeline.states:
                    builder.add(state, observer.update(state))
            records.append(dict(anchors=segment['anchors'], snapshots=snapshots(timeline, segment, segment['anchors'])))
        group = copy.copy(block)
        group.base, group.arrays, group.run_counts = base, parts, base.run_counts
        prepared[role] = group
        rendered[role] = RenderWindows(group, records)
    bank = builder.finish(artifact['train_sources'])
    reference = read_json(HOME / 'preprocessing.json')
    canonical = read_json(HOME / 'data_config.json')
    if (columns == canonical['native_covariates']['columns']
            and all(config[k] == canonical[k] for k in ('context', 'horizon', 'stride'))
            and sensors == canonical['sensors'] and steps == canonical['main_steps']
            and all(loc.fingerprint['csv_manifest_sha256'] == reference['manifest_sha256']
                    for loc in locations.values())):
        fitted = dict(profile=profile, artifact=artifact, scaler=scaler, data_config=config, bank=bank)
        for key, value in fitted.items():
            if digest(value) != reference['digests'][key]:
                raise ValueError(f'Reference dataset preprocessing differs from selected D-basic: {key}')
        log.emit('reproduction', 'Original schema, process vocabulary, Train scaler and renderer bank matched.')
    for name, data in [('data_config', config), ('scaler', scaler), ('pattern_bank', bank),
                       ('source_partitions', {k:v.original.snapshot() for k,v in locations.items()}),
                       ('process_metadata', artifact), ('process_profile', profile)]:
        write_json(output / (name + '.json'), data)
    return prepared, rendered, profile, artifact, config, scaler, bank, fit_baselines(prepared['train'])
