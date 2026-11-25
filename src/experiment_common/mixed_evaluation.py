"""Opt-in state classification beside unchanged continuous forecast evaluation."""
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader

from .evaluation import Metrics, evaluate as numeric_evaluate
from .experiment_reporting import Progress, metric_line as numeric_metric_line
from .sensor_states import state_sensors, validate_threshold, apply_state_threshold


class _Columns:
    def __init__(self, values, columns): self.values = values; self.columns = columns
    def __getitem__(self, index): return self.values[index][..., self.columns]


class _ContinuousView:
    """Slice only requested Window values, never copy whole Run arrays."""
    def __init__(self, data, columns):
        self.data = data; self.columns = columns
        self.std = data.std[columns]; self.mean = data.mean[columns]

    def location(self, index):
        segment, j = self.data.location(index)
        result = dict(segment)
        if 'values' in result: result['values'] = _Columns(segment['values'], self.columns)
        return result, j


def _classification(counts):
    tn, fp, fn, tp = map(int, counts.reshape(-1))
    def ratio(a, b): return float(a / b) if b else 0.0
    po = ratio(tp, tp + fp); ro = ratio(tp, tp + fn)
    pc = ratio(tn, tn + fn); rc = ratio(tn, tn + fp)
    fo = ratio(2 * tp, 2 * tp + fp + fn); fc = ratio(2 * tn, 2 * tn + fn + fp)
    return dict(points=tn+fp+fn+tp, accuracy=ratio(tn+tp,tn+fp+fn+tp),
        open_precision=po, open_recall=ro, open_f1=fo, close_precision=pc,
        close_recall=rc, close_f1=fc, macro_f1=(fo+fc)/2,
        true_close_pred_close=tn, true_close_pred_open=fp,
        true_open_pred_close=fn, true_open_pred_open=tp,
        true_open_points=tp+fn, true_close_points=tn+fp,
        zero_division=0)


class MixedMetrics:
    def __init__(self, data, epsilon, threshold):
        self.data = data; self.threshold = validate_threshold(threshold)
        self.states = state_sensors(data.config); names = data.config['sensors']
        self.indices = [names.index(n) for n in self.states]
        self.continuous = [i for i,n in enumerate(names) if n not in self.states]
        self.all = Metrics(names, epsilon)
        self.numeric = Metrics([names[i] for i in self.continuous], epsilon) if self.continuous else None
        self.view = _ContinuousView(data, self.continuous)
        self.counts = {n: np.zeros((2,2), dtype=np.int64) for n in self.states}
        self.examples = []; self.example_counts = defaultdict(int)

    def update(self, prediction, target, indices, data):
        self.all.update(prediction, target, indices, data)
        if self.numeric:
            self.numeric.update(prediction[...,self.continuous],target[...,self.continuous],indices,self.view)
        raw = prediction * data.std + data.mean
        labels = apply_state_threshold(raw, data.config['sensors'], self.states, self.threshold)
        actual = []
        for index in indices:
            segment,j = data.location(int(index))
            actual.append(segment['values'][segment['targets'][j]])
        actual = np.asarray(actual)
        for column,name in zip(self.indices,self.states):
            y = actual[...,column]
            if not np.isfinite(y).all() or not np.isin(y,[0,1]).all():
                raise ValueError('Invalid frozen binary target: '+name)
            predicted = labels[...,column].astype(np.int64); truth = y.astype(np.int64)
            self.counts[name] += np.bincount((2*truth+predicted).reshape(-1),minlength=4).reshape(2,2)
            remaining = max(0,32-self.example_counts[name])
            for flat in range(min(remaining,truth.size)):
                row,h = np.unravel_index(flat,truth.shape)
                self.examples.append(dict(sensor=name, window_index=int(indices[row]),
                    run_id=int(data.location(int(indices[row]))[0]['run_id']),horizon_step=h+1,
                    actual=int(truth[row,h]), raw_prediction=float(raw[row,h,column]),
                    predicted=int(predicted[row,h]), predicted_state='OPEN' if predicted[row,h] else 'CLOSE',
                    threshold=self.threshold))
                self.example_counts[name] += 1

    @property
    def groups(self): return self.numeric.groups if self.numeric else {}
    @property
    def horizons(self): return self.numeric.horizons if self.numeric else {}
    scores = staticmethod(Metrics.scores)

    def report(self):
        all_regression = self.all.report()
        report = dict(all_regression)
        if self.numeric:
            continuous = self.numeric.report()
            report['aggregate'] = continuous['aggregate']
            report['continuous_run_macro_scaled_mae'] = continuous['run_macro_scaled_mae']
        else:
            report['aggregate'] = dict(sensor_macro=dict(mae=None,rmse=None,bias=None,mape_percent=None),
                sensor_count=0,mape_eligible_sensors=0,mape_coverage_percent=None,all_target_points=None,
                mape_policy=dict(rule='not applicable: no continuous sensors',official_policy_approved=False))
        report['aggregate'].update(sensor_count=len(self.data.config['sensors']),
            continuous_sensor_count=len(self.continuous),binary_sensor_count=len(self.states),
            regression_scope='continuous sensors only; normalized stopping MAE includes all raw predictions')
        for name in self.states:
            sensor = report['sensors'][name]
            sensor['target_type'] = 'open_close'
            sensor['run_macro_mape_percent'] = None
            for key in ['mape_percent','mape_points','mape_excluded_near_zero_points','mape_coverage_percent']:
                sensor['window_weighted'][key] = None
        results = {n:_classification(c) for n,c in self.counts.items()}
        report['binary'] = dict(threshold=self.threshold,comparison='raw prediction > threshold',
            mapping={'OPEN':1,'CLOSE':0},sensor_count=len(self.states),
            sensor_macro_f1=float(np.mean([v['macro_f1'] for v in results.values()])),sensors=results,
            point_weighted=_classification(sum(self.counts.values())),
            score_is_probability=False,scope='Window target points; overlapping targets may repeat')
        return report


def evaluate(model, data, device, batch_size, max_batches=0, log=None, log_every=100,
             phase='valid', epoch=None, mape_epsilon=1e-6, state_threshold=0.5, observer=None):
    validate_threshold(state_threshold)
    if not state_sensors(data.config):
        return numeric_evaluate(model,data,device,batch_size,max_batches,log,log_every,phase,epoch,mape_epsilon,observer)
    model.eval(); tracker = MixedMetrics(data,mape_epsilon,state_threshold)
    baseline = MixedMetrics(data,mape_epsilon,state_threshold)
    loader = DataLoader(data,batch_size=batch_size,shuffle=False,num_workers=0)
    total = min(len(loader),max_batches) if max_batches else len(loader)
    progress = Progress(log,phase,total,log_every,epoch) if log else None
    windows = 0
    with torch.no_grad():
        for batch,(x,y,indices) in enumerate(loader):
            if max_batches and batch >= max_batches: break
            pred = model(x.to(device)).cpu().numpy()
            if not np.isfinite(pred).all(): raise ValueError('Nonfinite prediction')
            if observer is not None: observer.update(pred, y.numpy(), indices, data)
            tracker.update(pred,y.numpy(),indices,data)
            baseline.update(np.repeat(x[:,-1:,:len(data.config['sensors'])].numpy(),data.config['horizon'],axis=1),y.numpy(),indices,data)
            windows += len(indices)
            if progress: progress.update(batch+1,windows=windows)
    if not windows: raise ValueError('No evaluated Windows')
    return dict(windows_evaluated=windows,windows_available=len(data),partial=windows<len(data),
                model=tracker.report(),persistence=baseline.report()),tracker


def metric_line(report, latency):
    if 'binary' not in report['model']: return numeric_metric_line(report,latency)
    a = report['model']['aggregate']; value = a['sensor_macro']['mape_percent']
    mape = 'N/A' if value is None else f'{value:.4f}%'
    return (f'Continuous sensor MAPE={mape}; binary macro F1={report["model"]["binary"]["sensor_macro_f1"]:.4f}; '
            f'state threshold={report["model"]["binary"]["threshold"]}; '
            f'Run-macro scaled MAE={report["model"]["run_macro_scaled_mae"]:.6f}; '
            f'prepared-Window latency mean={latency["mean_ms"]:.3f}ms (partial boundary)')
