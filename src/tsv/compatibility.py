"""Explicit migration of the byte-identical published checkpoint to standalone source paths."""
import json

from stepaware.contracts import MODEL, digest
from stepaware.future_path.model import Forecast
from stepaware.identity import source_identity


def restore_forecast(state, device='cpu', *, checksum, context=None, horizon=None):
    release = json.loads((MODEL / 'release.json').read_text(encoding='utf-8'))
    if checksum == release['checkpoint_sha256']:
        current = source_identity()
        if digest(current) != release['standalone_code_sha256']:
            raise ValueError('The bundled checkpoint requires the published standalone model implementation')
        if digest(state['execution_code']) != release['original_code_sha256']:
            raise ValueError('Original checkpoint source signature differs')
        state = dict(state, execution_code=current, code=current)
    return Forecast.restore(state, device, context=context, horizon=horizon)
