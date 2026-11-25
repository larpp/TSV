"""Validation-only stopping and learning-rate control."""
import numpy as np
import torch


class ValidationStop:
    """Keep the absolute best checkpoint, but require meaningful progress to wait longer."""
    def __init__(self, patience, min_delta):
        if patience < 1 or not np.isfinite(min_delta) or min_delta < 0:
            raise ValueError('Invalid early-stopping settings')
        self.patience = patience; self.min_delta = min_delta
        self.best = float('inf'); self.reference = float('inf'); self.stale = 0

    def update(self, score):
        if not np.isfinite(score): raise ValueError('Nonfinite validation score')
        checkpoint = score < self.best
        if checkpoint: self.best = float(score)
        if score < self.reference - self.min_delta:
            self.reference = float(score); self.stale = 0
        else: self.stale += 1
        return checkpoint, self.stale >= self.patience


def validation_scheduler(optimizer, args):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
        factor=args.lr_factor, patience=args.lr_patience, threshold=args.min_delta,
        threshold_mode='abs', min_lr=args.min_lr)
