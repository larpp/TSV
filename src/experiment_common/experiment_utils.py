"""Common device selection and portable JSON/CSV output."""
import json
from pathlib import Path
import torch


def device_for(name):
    if name == 'cuda' and not torch.cuda.is_available(): raise ValueError('CUDA unavailable')
    return torch.device('cuda' if name == 'cuda' or (name == 'auto' and torch.cuda.is_available()) else 'cpu')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path, rows):
    import csv
    rows = list(rows)
    if not rows: return
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
