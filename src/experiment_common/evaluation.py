"""Model-independent sensor/Run/horizon metrics and evaluation."""
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader
from .experiment_reporting import Progress


class Metrics:
    def __init__(self, sensors, mape_epsilon=1e-6):
        self.sensors = sensors
        self.mape_epsilon=mape_epsilon
        self.groups = defaultdict(lambda: np.zeros(7, dtype=np.float64))
        self.horizons = defaultdict(lambda: np.zeros(7, dtype=np.float64))
        self.scaled_runs = defaultdict(lambda: np.zeros(2, dtype=np.float64))

    def update(self, prediction, target, indices, data):
        scaled = prediction - target
        error = scaled * data.std
        run_ids = []; steps = []; raw_targets=[]
        for i, index in enumerate(indices):
            s, j = data.location(int(index)); rid = s['run_id']
            target_steps = (s['main_steps'][s['targets'][j]] if 'main_steps' in s
                            else np.full(error.shape[1], s['main_step']))
            run_ids.append(rid); steps.append(target_steps)
            raw_targets.append(s['values'][s['targets'][j]] if 'values' in s
                else target[i] * data.std + getattr(data,'mean',np.zeros(len(self.sensors))))
        run_ids = np.asarray(run_ids); steps = np.asarray(steps)
        raw_targets=np.asarray(raw_targets,dtype=np.float64)
        eligible=np.abs(raw_targets)>self.mape_epsilon
        ape=np.zeros_like(raw_targets)
        np.divide(np.abs(error)*100, np.abs(raw_targets), out=ape, where=eligible)
        # Aggregate over a batch before splitting channels; whole-Run evaluation
        # otherwise repeats millions of tiny reductions for100+ sensors.
        for rid in np.unique(run_ids):
            selected = run_ids == rid
            self.scaled_runs[int(rid)] += [np.abs(scaled[selected]).mean(axis=(1, 2)).sum(), int(selected.sum())]
            for step in np.unique(steps[selected]):
                mask=(run_ids[:,None]==rid)&(steps==step)
                e = error[mask]; a=ape[mask]; ok=eligible[mask]
                sums = np.stack([np.abs(e).sum(axis=0), (e * e).sum(axis=0), e.sum(axis=0),
                                 np.full(len(self.sensors), len(e)),a.sum(axis=0),ok.sum(axis=0),
                                 np.where(ok,0,np.abs(e)).sum(axis=0)], axis=1)
                for channel, sensor in enumerate(self.sensors):
                    self.groups[(int(rid), int(step), sensor)] += sums[channel]
        horizon_sums = np.stack([np.abs(error).sum(axis=0), (error * error).sum(axis=0), error.sum(axis=0),
                                np.full(error.shape[1:], len(error)),ape.sum(axis=0),eligible.sum(axis=0),
                                np.where(eligible,0,np.abs(error)).sum(axis=0)], axis=-1)
        for channel, sensor in enumerate(self.sensors):
            for h in range(error.shape[1]):
                self.horizons[(sensor, h + 1)] += horizon_sums[h, channel]

    @staticmethod
    def scores(v):
        return dict(mae=float(v[0] / v[3]), rmse=float(np.sqrt(v[1] / v[3])),
                    bias=float(v[2] / v[3]), points=int(v[3]),
                    mape_percent=float(v[4]/v[5]) if v[5] else None,
                    mape_points=int(v[5]), mape_excluded_near_zero_points=int(v[3]-v[5]),
                    mape_coverage_percent=float(100*v[5]/v[3]),
                    near_zero_mae=float(v[6]/(v[3]-v[5])) if v[3]>v[5] else None)

    def report(self):
        per_sensor = {}
        for sensor in self.sensors:
            groups = {k: v for k, v in self.groups.items() if k[2] == sensor}
            by_run = defaultdict(lambda: np.zeros(7))
            for (rid, step, _), v in groups.items(): by_run[rid] += v
            run_mapes=[self.scores(v)['mape_percent'] for v in by_run.values() if v[5]]
            per_sensor[sensor] = dict(window_weighted=self.scores(sum(groups.values())),
                run_macro_mae=float(np.mean([self.scores(v)['mae'] for v in by_run.values()])),
                run_macro_rmse=float(np.mean([self.scores(v)['rmse'] for v in by_run.values()])),
                run_macro_mape_percent=float(np.mean(run_mapes)) if run_mapes else None)
        macro={}
        for metric in ['mae','rmse','bias','mape_percent']:
            values=[s['window_weighted'][metric] for s in per_sensor.values() if s['window_weighted'][metric] is not None]
            macro[metric]=float(np.mean(values)) if values else None
        totals=self.scores(sum(self.groups.values()))
        aggregate=dict(sensor_macro=macro, sensor_count=len(self.sensors),
            mape_eligible_sensors=sum(s['window_weighted']['mape_percent'] is not None for s in per_sensor.values()),
            mape_coverage_percent=totals['mape_coverage_percent'], all_target_points=totals,
            raw_mae_rmse_note='Averages mix sensor units; use normalized Run-macro MAE for comparable error.',
            mape_policy=dict(epsilon_raw_units=self.mape_epsilon, rule='exclude abs(actual raw target)<=epsilon; no denominator clipping',
                weighting='equal eligible sensor weight, each sensor pooled over overlapping Window target-points',
                near_zero='excluded from MAPE only; included in MAE/RMSE and near_zero_mae',official_policy_approved=False))
        return dict(run_macro_scaled_mae=float(np.mean([v[0] / v[1] for v in self.scaled_runs.values()])),
                    sensors=per_sensor, aggregate=aggregate,runs_evaluated=len(self.scaled_runs))


def evaluate(model, data, device, batch_size, max_batches=0, log=None, log_every=100, phase='valid', epoch=None, mape_epsilon=1e-6, observer=None):
    model.eval(); tracker = Metrics(data.config['sensors'],mape_epsilon); baseline = Metrics(data.config['sensors'],mape_epsilon)
    windows = 0
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0)
    total=min(len(loader),max_batches) if max_batches else len(loader)
    progress=Progress(log,phase,total,log_every,epoch) if log else None
    with torch.no_grad():
        for batch, (x, y, indices) in enumerate(loader):
            if max_batches and batch >= max_batches: break
            pred = model(x.to(device)).cpu().numpy(); target = y.numpy()
            if not np.isfinite(pred).all(): raise ValueError('Nonfinite prediction')
            if observer is not None: observer.update(pred, target, indices, data)
            tracker.update(pred, target, indices, data)
            # Model-local past covariates may be an input-only suffix.
            persistence = np.repeat(x[:, -1:, :len(data.config['sensors'])].numpy(), data.config['horizon'], axis=1)
            baseline.update(persistence, target, indices, data)
            windows += len(indices)
            if progress: progress.update(batch+1, windows=windows)
    if not windows: raise ValueError('No evaluated Windows')
    return dict(windows_evaluated=windows, windows_available=len(data),
                partial=windows < len(data), model=tracker.report(), persistence=baseline.report()), tracker
