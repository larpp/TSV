"""Run-local progress, plots and explicitly scoped offline Precision measurements."""
from contextlib import ExitStack, redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import traceback
import uuid

import numpy as np
import torch


def new_output(root, prefix=''):
    path = Path(root) / (prefix + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:12])
    path.mkdir(parents=True, exist_ok=False)
    return path


class Tee:
    def __init__(self, terminal, file): self.terminal = terminal; self.file = file
    def write(self, text):
        self.file.write(text); self.file.flush()
        return self.terminal.write(text)
    def flush(self): self.file.flush(); self.terminal.flush()
    def isatty(self): return False
    @property
    def encoding(self): return getattr(self.terminal, 'encoding', 'utf-8')


class RunLog:
    def __init__(self, output): self.output = Path(output); self.progress_events = []
    def __enter__(self):
        self.stack = ExitStack()
        console = self.stack.enter_context((self.output / 'console.log').open('x', encoding='utf-8'))
        self.events = self.stack.enter_context((self.output / 'progress.jsonl').open('x', encoding='utf-8'))
        self.stack.enter_context(redirect_stdout(Tee(sys.stdout, console)))
        self.stack.enter_context(redirect_stderr(Tee(sys.stderr, console)))
        return self
    def emit(self, event, message=None, **values):
        row = dict(time_utc=datetime.now(timezone.utc).isoformat(), event=event, **values)
        self.events.write(json.dumps(row, allow_nan=False) + '\n'); self.events.flush()
        if event == 'progress': self.progress_events.append(row)
        if message: print(message, flush=True)
    def __exit__(self, kind, error, tb):
        if error is not None:
            self.emit('failed', error_type=kind.__name__, reason=str(error))
            traceback.print_exception(kind, error, tb)
        self.stack.close()


class Progress:
    def __init__(self, log, phase, total, every=100, epoch=None):
        self.log=log; self.phase=phase; self.total=total; self.every=every; self.epoch=epoch
        self.started=time.perf_counter()
    def update(self, iteration, **values):
        if iteration not in (1, self.total) and iteration % self.every: return
        elapsed=time.perf_counter()-self.started
        eta=elapsed * (self.total-iteration) / iteration
        fields='; '.join(f'{k}={v:.6g}' for k,v in values.items() if isinstance(v,(float,int)))
        title=f'{self.phase}' + (f' epoch {self.epoch}' if self.epoch is not None else '')
        self.log.emit('progress', f'{title} {iteration}/{self.total} ({100*iteration/self.total:.1f}%) '
            f'elapsed={elapsed:.0f}s ETA={eta:.0f}s; {fields}', phase=self.phase, epoch=self.epoch,
            iteration=iteration, total_iterations=self.total, elapsed_seconds=elapsed, eta_seconds=eta, **values)


def benchmark_latency(model, data, device, repeats=100, warmup=10):
    """Batch1 prepared normalized CPU Window -> raw CPU forecast; no dataset IO timed."""
    if repeats<1 or warmup<0: raise ValueError('Invalid latency repetition/warmup settings')
    was_training=model.training
    model.eval()
    indices=np.linspace(0, len(data)-1, repeats, dtype=int)
    samples=[]
    def forecast(index):
        # Data lookup/preprocessing is deliberately outside this partial boundary.
        x,_,_=data[int(index)]
        if device.type=='cuda': torch.cuda.synchronize(device)
        start=time.perf_counter()
        tensor=torch.from_numpy(x[None]).to(device)
        prediction=model(tensor).detach().cpu().numpy()
        raw=prediction * data.std + data.mean
        if device.type=='cuda': torch.cuda.synchronize(device)
        elapsed=(time.perf_counter()-start)*1000
        if not np.isfinite(raw).all(): raise ValueError('Nonfinite benchmark forecast')
        return elapsed
    try:
        with torch.inference_mode():
            for _ in range(warmup): forecast(indices[0])
            for index in indices: samples.append(forecast(index))
    finally: model.train(was_training)
    return dict(mean_ms=float(np.mean(samples)), p95_ms=float(np.percentile(samples,95)),
        p99_ms=float(np.percentile(samples,99)), min_ms=float(min(samples)), max_ms=float(max(samples)),
        samples_ms=samples, input_window_indices=indices.tolist(), repetitions=repeats, warmup=warmup,
        batch_size=1, sensor_count=len(data.config['sensors']), context=data.config['context'],
        horizon=data.config['horizon'], device=str(device), torch_version=str(torch.__version__),
        device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
        boundary='prepared normalized CPU Window -> transfer + forward + CPU copy + raw-unit forecast',
        exclusions=['raw collection','Window assembly/normalization','queue waiting','process startup','DB/storage'],
        official_cycle_kpi_evaluated=False)


def precision_targets(report, latency):
    overall=report['model']['aggregate']
    mape=overall['sensor_macro']['mape_percent']
    return dict(mape=dict(value_percent=mape, threshold_percent=5.0,
            met_on_eligible_targets=None if mape is None else mape<=5.0,
            eligible_sensors=overall['mape_eligible_sensors'], total_sensors=overall['sensor_count'],
            coverage_percent=overall['mape_coverage_percent'], scope='sensor-macro; non-near-zero raw targets only'),
        latency=dict(value_ms=latency['mean_ms'], threshold_ms=100.0,
            met_for_prepared_window=latency['mean_ms']<=100.0, scope=latency['boundary']),
        official_kpi_status='not_evaluated: target/units/zero policy and Edge/end-to-end boundary not approved')


def metric_line(report, latency):
    a=report['model']['aggregate'];m=a['sensor_macro']['mape_percent']
    mape='N/A' if m is None else f'{m:.4f}%'
    return (f"Sensor-macro MAPE={mape} (target<=5%, coverage={a['mape_coverage_percent']:.2f}%, "
            f"eligible sensors={a['mape_eligible_sensors']}/{a['sensor_count']}); "
            f"sensor-macro raw MAE={a['sensor_macro']['mae']:.6g} RMSE={a['sensor_macro']['rmse']:.6g} (mixed units); "
            f"Run-macro scaled MAE={report['model']['run_macro_scaled_mae']:.6f}; "
            f"prepared-Window batch1 latency mean={latency['mean_ms']:.3f}ms "
            f"p95={latency['p95_ms']:.3f}ms p99={latency['p99_ms']:.3f}ms (target mean<=100ms; partial boundary)")


def pyplot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def plot_training(history, output, progress=()):
    plt=pyplot();epochs=[r['epoch'] for r in history]
    fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    axes[0,0].plot(epochs,[r['train_scaled_mse'] for r in history],label='Train normalized MSE')
    axes[0,1].plot(epochs,[r['validation']['model']['run_macro_scaled_mae'] for r in history],label='Valid normalized MAE')
    axes[0,1].plot(epochs,[r['validation']['persistence']['run_macro_scaled_mae'] for r in history],label='Persistence',alpha=.6)
    mape=[r['validation']['model'].get('aggregate',{}).get('sensor_macro',{}).get('mape_percent') for r in history]
    axes[1,0].plot(epochs,[np.nan if v is None else v for v in mape],label='Valid sensor-macro MAPE (%)')
    if all(v is None for v in mape):axes[1,0].text(.5,.5,'MAPE was not recorded in this historical run',transform=axes[1,0].transAxes,ha='center')
    axes[1,0].axhline(5,color='#b42318',linestyle='--',label='5% reference (eligible targets)')
    latency=[r.get('latency',{}).get('mean_ms',np.nan) for r in history]
    axes[1,1].plot(epochs,latency,label='Prepared-Window mean latency (ms)')
    if not np.isfinite(latency).any():axes[1,1].text(.5,.5,'Latency was not recorded in this historical run',transform=axes[1,1].transAxes,ha='center')
    axes[1,1].axhline(100,color='#b42318',linestyle='--',label='100ms reference (partial boundary)')
    for ax in axes.flat:ax.set_xlabel('Epoch');ax.grid(alpha=.2);ax.legend()
    fig.savefig(Path(output)/'learning_curves.png',dpi=160);plt.close(fig)
    points=[r for r in progress if r.get('phase')=='train' and 'loss_running' in r]
    if points:
        fig,ax=plt.subplots(figsize=(11,4),constrained_layout=True)
        ax.plot([r['global_iteration'] for r in points],[r['loss_running'] for r in points])
        ax.set(xlabel='Global training iteration',ylabel='Epoch running normalized MSE');ax.grid(alpha=.2)
        fig.savefig(Path(output)/'iteration_loss.png',dpi=160);plt.close(fig)


def plot_test(report, latency, output):
    plt=pyplot();fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    values=[report[k]['aggregate']['sensor_macro']['mape_percent'] for k in ['model','persistence']]
    axes[0,0].bar(['Model','Persistence'],[np.nan if v is None else v for v in values],color=['#1C499E','#59B72F'])
    axes[0,0].axhline(5,color='#b42318',linestyle='--');axes[0,0].set_title('Sensor-macro MAPE (%) on eligible targets')
    axes[0,1].hist(latency['samples_ms'],bins=min(20,len(latency['samples_ms'])),color='#1C499E')
    axes[0,1].axvline(100,color='#b42318',linestyle='--');axes[0,1].set_title('Prepared-Window batch1 latency (ms)')
    sensors=sorted(((n,s['window_weighted']['mape_percent']) for n,s in report['model']['sensors'].items()
                    if s['window_weighted']['mape_percent'] is not None),key=lambda x:x[1],reverse=True)[:15]
    if sensors:
        names,scores=zip(*sensors);axes[1,0].barh(names[::-1],scores[::-1],color='#1C499E')
    axes[1,0].set_title('Largest15 sensor MAPE (%)');axes[1,0].axvline(5,color='#b42318',linestyle='--')
    axes[1,1].bar(['Model','Persistence'],[report[k]['run_macro_scaled_mae'] for k in ['model','persistence']],color=['#1C499E','#59B72F'])
    axes[1,1].set_title('Run-macro normalized MAE')
    for ax in axes.flat:ax.grid(alpha=.2,axis='x')
    fig.savefig(Path(output)/'test_summary.png',dpi=160);plt.close(fig)
