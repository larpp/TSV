"""Explicit Main Step selection for whole-Run forecasting; Substeps are retained."""
import json
from pathlib import Path

DEFAULT_STEP_PROFILE = Path(__file__).resolve().parents[2] / 'config/step_profile.json'


def resolve_steps(profile):
    if not isinstance(profile, dict) or set(profile) != {'schema_version', 'include', 'exclude'}:
        raise ValueError('Step profile requires exactly schema_version, include, exclude')
    if type(profile['schema_version']) is not int or profile['schema_version'] != 1:
        raise ValueError('Unsupported step profile schema_version')
    for name in ['include', 'exclude']:
        values = profile[name]
        if (not isinstance(values, list) or any(type(v) is not int or not 0 <= v < 255 for v in values)
                or len(values) != len(set(values))):
            raise ValueError('Step include/exclude must contain unique integers 0..254')
    selected = sorted(set(profile['include']) - set(profile['exclude']))
    if not selected: raise ValueError('Step profile selects no Main Steps')
    return selected


def prepare_steps(path=None, explicit=None):
    if explicit is not None:
        profile = dict(schema_version=1, include=explicit, exclude=[])
    else:
        profile = json.loads(Path(path or DEFAULT_STEP_PROFILE).read_text(encoding='utf-8-sig'))
    return dict(selected_steps=resolve_steps(profile), profile_snapshot=profile,
                window_scope='run', ordering='numeric Main Step order; exclude wins',
                substeps='all observed Substeps and Subcycles within selected Main Steps')
