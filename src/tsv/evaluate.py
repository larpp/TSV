"""Test a D-basic checkpoint with batch-one causal replay."""
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch

from stepaware.contracts import MODEL, ROOT, file_sha
from stepaware.data import ProcessTimeline
from experiment_common.timestral_covariates.data import GROUPS
from .compatibility import restore_forecast
from stepaware.future_path.patterns import LoopObserver
from stepaware.metadata import read_process_rows
from stepaware.tracker import StateTracker
from stepaware.training.runner import move
from experiment_common.experiment_reporting import RunLog, new_output
from experiment_common.experiment_utils import device_for
from experiment_common.timestral_csv.data import CsvRunWindows
from .metrics import SensorMetrics

LATENCY_SCOPE = ('Batch 1, decoded rows through state tracking, context assembly, renderer, '
                 'model and original-unit CPU output. All origins including first use. '
                 'Excludes CSV decoding, checkpoint loading, scoring and UI.')


class CausalInputs:
    """Track this Run once, exposing only observations received by the origin."""
    def __init__(self, segment, rows, model, data):
        self.segment = segment
        self.rows = rows
        self.model = model
        self.data = data
        self.tracker = StateTracker(model.profile, model.artifact['schema'], start_observed=True)
        self.loops = LoopObserver()
        self.ticks = segment['origin_us'] + np.rint(segment['times'] * 1e6).astype(np.int64)
        if len(rows) != len(self.ticks):
            raise ValueError('Expected one complete timestamp-aligned segment per frozen Test Run')
        self.states = []
        self.sets = np.full((len(rows), len(model.profile['absolute_set_columns'])), np.nan, np.float32)

    def at(self, anchor):
        seg = self.segment
        context = self.data.config['context']
        horizon = self.data.config['horizon']
        grid = seg['origin_us'] + (anchor + np.arange(1-context, 1, dtype=np.int64)) * 1_000_000
        stop = self.ticks.searchsorted(grid[-1], side='right')
        profile = self.model.profile
        for i in range(len(self.states), stop):
            row = self.rows[i]
            state = self.tracker.update(row)
            if state['timestamp_us'] != self.ticks[i]:
                raise ValueError('Process and sensor timestamps differ')
            self.states.append(state)
            self.loops.update(state)
            if row[profile['audit_columns']['row_observed']] == '1':
                for j, name in enumerate(profile['absolute_set_columns']):
                    try:
                        self.sets[i, j] = float(row[name])
                    except (ValueError, TypeError):
                        pass
        first = self.ticks[:stop].searchsorted(grid[0], side='right') - 1
        if first < 0:
            raise ValueError('Insufficient history')
        history = SimpleNamespace(ticks=self.ticks[first:stop], states=self.states[first:stop],
                                  sets=self.sets[first:stop], profile=profile)
        states, sets, local = ProcessTimeline.context(history, grid)
        ix = first + local
        raw = seg['values'][ix].copy()
        raw[grid - self.ticks[ix] > 1e6 * profile['maximum_observation_gap_seconds']] = np.nan
        x = (raw - self.data.mean) / self.data.std
        if 'native_covariates' in self.data.config:
            past = seg['cov_past'][anchor+1-context:anchor+1]
            known = seg['cov_known'][anchor+1-context:anchor+1]
            # Offline replay treats only explicitly declared known columns as a schedule.
            blocks = (horizon + context - 1) // context
            width = known.shape[1]
            tail = np.full((blocks * context, width), np.nan, dtype=known.dtype)
            tail[:horizon] = seg['cov_known'][anchor+1:anchor+1+horizon]
            tail = tail.reshape(blocks, context, width).transpose(1, 0, 2).reshape(context, blocks*width)
            x = np.concatenate((x, past, known, tail), axis=1)
        encoded = self.model.codec.encode(states, sets)
        current = states[-1]
        _, candidates = self.model.codec.candidates(current)
        width = max(1, len(candidates))
        ids = np.zeros((1, width, candidates.shape[1]), np.int64)
        ids[0, :len(candidates)] = candidates
        return dict(x=torch.from_numpy(x[None]),
                    process={k: torch.from_numpy(v[None]) for k, v in encoded.items()},
                    candidate_categories=torch.from_numpy(ids),
                    candidate_mask=torch.from_numpy((np.arange(width) < len(candidates))[None]),
                    causal=[dict(current=current, recent=self.loops.recent if current['quality']=='OBSERVED' else None)])


@torch.inference_mode()
def evaluate(output, log, args):
    from .inputs import location, selected_profiles, covariate_columns, REFERENCE, REFERENCE_SHA
    checkpoint = args.checkpoint.resolve()
    checksum = file_sha(checkpoint)
    if checkpoint == REFERENCE and checksum != REFERENCE_SHA:
        raise ValueError('Released checkpoint hash mismatch')
    device = device_for(args.device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    expected = json.loads((MODEL / 'model_source.json').read_text(encoding='utf-8'))
    if state.get('source') != expected:
        raise ValueError(f'{MODEL.name} requires a D-basic checkpoint built from {expected["repo_id"]}')
    policy = state['data_config'].get('native_covariates')
    columns = covariate_columns(args, policy['columns'] if policy else {g: [] for g in GROUPS})
    threshold = args.state_threshold if args.state_threshold is not None else state.get('training', {}).get('state_threshold', .5)
    model = restore_forecast(state, device, checksum=checksum, context=args.context, horizon=args.horizon).eval()
    del state
    if not model.enable_future or (checkpoint == REFERENCE and model.coefficient != 1):
        raise ValueError('Unexpected renderer configuration')
    sensors, steps = selected_profiles(args, columns)
    if sensors != model.targets:
        raise ValueError('Checkpoint target schema differs from sensor profile. Use its training profile or retrain.')
    source = location(args.test_data, 'test', output, args.column, steps)
    config = dict(model.data_config, main_steps=steps)
    if args.stride is not None:
        config['stride'] = args.stride
    log.emit('window_settings', f"Context={config['context']}s | horizon={config['horizon']}s | "
             f"stride={config['stride']}s | OPEN when prediction > {threshold:g}")
    data = CsvRunWindows(source.source, 'test', config, location=source)
    data.set_scaler(model.scaler)
    groups = {'ALL': SensorMetrics(data, 1e-6, threshold)}
    latency = []
    by_run = {row['run_id']: row for row in source.rows}
    for n, segment in enumerate(data.segments):
        row = by_run[segment['run_id']]
        if segment['run_id'] != row['run_id']:
            raise ValueError('Unexpected Run order')
        recipe = row['recipe']
        if recipe not in groups:
            groups[recipe] = SensorMetrics(data, 1e-6, threshold)
        stream = CausalInputs(segment, read_process_rows(source.segment_paths[segment['run_id'], segment['origin_us']], model.profile), model, data)
        predictions, targets, indices = [], [], []
        for j, anchor in enumerate(segment['anchors']):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = perf_counter()
            batch = stream.at(int(anchor))
            scaled = model(move(batch, device))['prediction'].cpu().numpy()[0]
            raw = scaled * data.std + data.mean
            latency.append(1000 * (perf_counter() - start))
            if not np.isfinite(raw).all():
                raise ValueError('Nonfinite sensor prediction')
            index = data.offsets[n] + j
            x, y, _ = data[index]
            np.testing.assert_equal(x, batch['x'][0].numpy())
            predictions.append(scaled)
            targets.append(y)
            indices.append(index)
            if len(indices) == 32 or j + 1 == len(segment['anchors']):
                p, y = np.stack(predictions), np.stack(targets)
                for tracker in (groups['ALL'], groups[recipe]):
                    tracker.update(p, y, indices, data)
                predictions.clear()
                targets.clear()
                indices.clear()
        log.emit('run_completed', f"[{n+1}/{len(data.segments)}] Run {row['run_id']}  {recipe}  {j+1} windows")

    reports = {name: tracker.report() for name, tracker in groups.items()}
    delay = dict(mean_ms=float(np.mean(latency)), p50_ms=float(np.median(latency)),
                 p95_ms=float(np.percentile(latency, 95)), p99_ms=float(np.percentile(latency, 99)),
                 over_100ms=int(np.count_nonzero(np.array(latency)>100)), windows=len(latency), scope=LATENCY_SCOPE)
    result = dict(checkpoint=str(checkpoint), checkpoint_sha256=checksum,
                  manifest_sha256=source.fingerprint['csv_manifest_sha256'], renderer_coefficient=model.coefficient,
                  retrieval=False, device=torch.cuda.get_device_name() if device.type=='cuda' else 'CPU',
                  test_directory=str(args.test_data.resolve()), sensor_profile_sha256=file_sha(args.sensor_profile),
                  step_profile_sha256=file_sha(args.step_profile), main_steps=steps, process_columns=args.column,
                  context=config['context'], horizon=config['horizon'], stride=config['stride'], state_threshold=threshold, covariate_columns=columns, metrics=reports, latency=delay)
    (output/'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print(f"\n{MODEL.name} Test | {len(data.run_counts)} Runs | {len(latency):,} windows | {result['device']}")
    def fmt(value, places):
        return 'N/A' if value is None else f'{value:.{places}f}'
    for name, report in reports.items():
        mape = report['aggregate']['sensor_macro']['mape_percent']
        print(f"{name}: continuous MAPE={fmt(mape,4)}% | continuous Run-macro scaled MAE="
              f"{fmt(report['continuous_run_macro_scaled_mae'],6)} | binary macro F1={fmt(report['binary']['sensor_macro_f1'],4)}")
    print(f"Batch-1 latency: mean={delay['mean_ms']:.2f}ms | p50={delay['p50_ms']:.2f}ms | "
          f"p95={delay['p95_ms']:.2f}ms | p99={delay['p99_ms']:.2f}ms | >100ms={delay['over_100ms']}/{len(latency)}")
    print(LATENCY_SCOPE)
    print(f"Results: {output}")


def main():
    import argparse
    from .inputs import column_options, window_options
    parser = argparse.ArgumentParser()
    column_options(parser)
    window_options(parser, training=False)
    parser.add_argument('--test-data', type=Path, default=ROOT / 'data/test')
    parser.add_argument('--device', choices=['auto','cuda','cpu'], default='auto')
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    if args.checkpoint is None:
        from .inputs import default_checkpoint
        args.checkpoint = default_checkpoint(args.output)
    directory = new_output(args.output / 'test', 'test-')
    with RunLog(directory) as logger:
        logger.emit('started', f'Checkpoint: {args.checkpoint}')
        evaluate(directory, logger, args)
        logger.emit('completed', f'Results: {directory}')
