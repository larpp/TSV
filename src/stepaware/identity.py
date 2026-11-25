"""Bind portable checkpoints to the executable source tree using stable relative names."""
import hashlib
from pathlib import Path

from .contracts import ROOT


def source_identity():
    src = ROOT / 'src'
    return {
        'format': 'tsv-standalone-v1',
        'files': {
            p.relative_to(src).as_posix(): hashlib.sha256(
                p.read_bytes().replace(b'\r\n', b'\n')
            ).hexdigest()
            for p in sorted(src.rglob('*.py'))
        },
    }
