"""Two-stage training for the selected D-basic model."""
import argparse
import random
from time import perf_counter
import uuid
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .inputs import column_options, read_json, window_options
from .training_data import prepare
from stepaware.model import ProcessSystem
from stepaware.contracts import MODEL, ROOT
from stepaware.future_path.model import Forecast, injection_coefficient
from stepaware.future_path.data import collate as forecast_collate
from stepaware.training.runner import collate as head_collate, move, configure_phase, save_state
from stepaware.training.process_metrics import HeadMetrics
from experiment_common.experiment_reporting import RunLog, new_output, Progress
from experiment_common.experiment_utils import device_for, write_json
from .metrics import SensorMetrics
from experiment_common.training_control import ValidationStop, validation_scheduler


def fit(stage, data, spec, args, output, log, pstate=None):
    profile, artifact, config, scaler, bank, baselines = spec
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.stage = stage
    device = device_for(args.device)
    head = stage == 'P'
    label = 'P' if head else 'D-basic'
    if head:
        model = ProcessSystem(profile, artifact, len(config['sensors'])).to(device)
        groups = model.parameters()
    else:
        model = Forecast(profile, artifact, config, scaler, bank, True).to(device)
        model.load_p_checkpoint(pstate)
        groups = [dict(params=model.backbone.parameters(), lr=0.),
                  dict(params=[p for n,p in model.named_parameters() if not n.startswith('backbone.')],
                       lr=args.adapter_learning_rate)]
    optimizer = torch.optim.AdamW(groups, lr=args.learning_rate, weight_decay=.01)
    stop = ValidationStop(args.p_patience if head else args.patience, args.min_delta)
    scheduler = validation_scheduler(optimizer, args)
    train, valid = data['train'], data['valid']
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(train.sampling_weights(), len(train), replacement=True, generator=generator)
    collate = head_collate if head else forecast_collate
    loader = DataLoader(train, batch_size=args.batch_size, sampler=sampler, collate_fn=collate)
    best = None
    history = []
    checkpoint = output / ('head_best.pt' if head else 'best.pt')
    started = perf_counter()
    for epoch in range(1, (args.p_epochs if head else args.epochs) + 1):
        model.train()
        phase = configure_phase(model, optimizer, args, epoch)
        if not head:
            model.coefficient = injection_coefficient(epoch, args.warmup_epochs)
        progress = Progress(log, label, len(loader), 25, epoch)
        for iteration, cpu in enumerate(loader, 1):
            batch = move(cpu, device)
            optimizer.zero_grad(set_to_none=True)
            losses = model.losses(batch, model(batch))
            loss = losses['total' if head else 'joint']
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
            optimizer.step()
            progress.update(iteration, loss=float(loss.detach()))
        model.eval()
        metric = HeadMetrics(valid, baselines) if head else SensorMetrics(valid.base, 1e-6, args.state_threshold)
        with torch.no_grad():
            for cpu in DataLoader(valid, batch_size=args.batch_size, collate_fn=collate):
                result = model(move(cpu, device))
                if head:
                    metric.update(result, cpu, cpu['indices'])
                else:
                    metric.update(result['prediction'].cpu().numpy(), cpu['y'].numpy(), cpu['indices'], valid.base)
            report = metric.report()
            if head:
                auxiliary = report['all']['auxiliary_run_macro_loss']
                if auxiliary is None:
                    raise ValueError('Selected Valid steps contain insufficient observable process transitions for P selection')
                score = auxiliary + 1e-4 * float(model.encoder.residual_penalty())
            else:
                score = report['run_macro_scaled_mae']
        terminate = False
        if phase != 'warmup':
            improved, terminate = stop.update(score)
            scheduler.step(score)
            if improved:
                best = epoch
                metadata = dict(stage='P' if head else 'D-basic', epoch=epoch, validation=report, state_threshold=args.state_threshold)
                state = (model.checkpoint(config['sensors'], scaler, training=metadata) if head
                         else model.checkpoint(training=metadata))
                save_state(checkpoint, state)
                log.emit('checkpoint_saved', f'{label} best epoch {epoch}: {checkpoint.resolve()}')
        history.append(dict(epoch=epoch, score=score, best=best, phase=phase))
        write_json(output / f'{label}_history.json', history)
        log.emit('epoch', f'{label} epoch {epoch}: {score:.6f}, best={best}')
        if terminate:
            break
    if best is None:
        raise ValueError(f'{stage} produced no eligible checkpoint')
    write_json(output / f'{label}_summary.json',
               dict(best_epoch=best, epochs_completed=len(history), elapsed_seconds=perf_counter()-started,
                    best_score=stop.best, checkpoint=checkpoint.name, stop_reason='early_stopping' if terminate else 'epoch_limit'))
    return checkpoint


def main():
    parser = argparse.ArgumentParser()
    column_options(parser)
    window_options(parser, training=True)
    for role in ('train','valid'):
        parser.add_argument('--'+role+'-data', type=Path, default=ROOT / 'data' / role)
    for key, default in [('seed',42),('epochs',100),('p-epochs',30),('warmup-epochs',3),
                         ('batch-size',8),('patience',8),('p-patience',5),('lr-patience',2)]:
        parser.add_argument('--'+key, type=int, default=default)
    for key, default in [('learning-rate',1e-4),('adapter-learning-rate',1e-4),('backbone-learning-rate',1e-5),
                         ('min-delta',1e-4),('min-lr',1e-7),('lr-factor',.5)]:
        parser.add_argument('--'+key, type=float, default=default)
    parser.add_argument('--device', choices=['cpu','cuda','auto'], default='auto')
    args = parser.parse_args()
    if args.epochs <= args.warmup_epochs or min(args.p_epochs, args.batch_size, args.patience, args.p_patience) < 1:
        parser.error('Positive budgets and at least one joint epoch are required')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    output = new_output(args.output/'train', 'steploop-')
    with RunLog(output) as log:
        for name in ('sensor_profile', 'step_profile'):
            write_json(output / f'{name}.json', read_json(getattr(args, name)))
        write_json(output/'arguments.json', {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
        log.emit('started', f'{MODEL.name} train output: {output}')
        log.emit('window_settings', f"Context={args.context}s | horizon={args.horizon}s | "
                 f"stride={args.stride}s | Valid OPEN when prediction > {args.state_threshold:g}")
        prep, render, *spec = prepare(args, output, log)
        head = fit('P', prep, spec, args, output, log)
        best = fit('C', render, spec, args, output, log, torch.load(head, map_location='cpu', weights_only=True))
        pointer = args.output / f'.latest-{uuid.uuid4().hex}.json'
        write_json(pointer, dict(checkpoint=best.resolve().relative_to(args.output.resolve()).as_posix()))
        pointer.replace(args.output/'latest.json')
        log.emit('completed', f'D-basic best: {best.resolve()}')

if __name__ == '__main__':
    main()
