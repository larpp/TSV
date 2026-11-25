"""Load the bundled official source and verify the original pretrained weights."""
import json
import sys

from stepaware.contracts import MODEL as HERE, ROOT
from experiment_common.tips_data import sha


def official_model_class():
    upstream = ROOT / 'vendor' / 'timestral'
    source = json.loads((HERE / 'source.json').read_text(encoding='utf-8'))
    for name, expected in source['files'].items():
        if sha(upstream / name) != expected:
            raise ValueError(f'Official Timestral source differs from {source["commit"]}: {name}')
    sys.path.insert(0, str(upstream / 'src'))
    from timestral.loading import load_model
    from timestral.forecaster import TimestralForecaster, pad_to_patch
    return load_model, TimestralForecaster, pad_to_patch


def verify_pretrained():
    source = json.loads((HERE / 'model_source.json').read_text(encoding='utf-8'))
    for name in ('config.json', 'model.safetensors'):
        if sha(HERE / source['local_directory'] / name) != source['files'][name]['sha256']:
            raise ValueError(f'Pretrained file hash mismatch: {name}. Check git lfs pull.')
    return source
