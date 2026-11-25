"""Collation, optimizer phases and checkpoint writes for TSV."""
import torch
from ..data import collate_windows


def collate(rows):
    batch=collate_windows(rows);batch['indices']=[r['index'] for r in rows];return batch


def move(batch,device):
    return {k:(move(v,device) if isinstance(v,dict) else v.to(device) if isinstance(v,torch.Tensor) else v) for k,v in batch.items()}


def phase_for(stage,epoch,warmup):return 'warmup' if stage=='C' and epoch<=warmup else ('joint' if stage=='C' else 'pretraining')


def configure_phase(model,optimizer,args,epoch):
    phase=phase_for(args.stage,epoch,args.warmup_epochs)
    if args.stage=='C':
        model.backbone.requires_grad_(phase=='joint')
        if phase=='warmup':model.backbone.eval();optimizer.param_groups[0]['lr']=0.
        elif epoch==args.warmup_epochs+1:
            optimizer.param_groups[0]['lr']=args.backbone_learning_rate
            optimizer.param_groups[1]['lr']=args.adapter_learning_rate
    return phase


def save_state(path,state):
    temporary=path.with_suffix('.pt.tmp');torch.save(state,temporary);temporary.replace(path)
