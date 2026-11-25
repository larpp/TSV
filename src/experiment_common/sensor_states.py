"""Frozen Train-derived OPEN/CLOSE types; numeric-only models opt out."""
import numpy as np

ENCODING_KEY = 'open_close_encoding'
MAPPING = {'OPEN': 1, 'CLOSE': 0}


def state_sensors(config):
    policy = config.get(ENCODING_KEY)
    if policy is None: return []
    if (not isinstance(policy, dict) or set(policy) != {'schema_version', 'mapping', 'sensors', 'fit_partition'}
            or type(policy['schema_version']) is not int or policy['schema_version'] != 1 or policy['mapping'] != MAPPING
            or any(type(v) is not int for v in policy['mapping'].values())
            or policy['fit_partition'] != 'train'):
        raise ValueError('Invalid frozen OPEN/CLOSE encoding')
    names = policy['sensors']
    if (not isinstance(names, list) or any(not isinstance(n, str) for n in names)
            or len(names) != len(set(names)) or not set(names) <= set(config['sensors'])):
        raise ValueError('OPEN/CLOSE sensors do not match frozen sensor order')
    return names


def validate_threshold(threshold):
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('State threshold must be finite and between 0 and 1')
    return float(threshold)


def apply_state_threshold(raw, sensors, states, threshold):
    """Keep continuous predictions; replace only state channels, strictly above cutoff."""
    validate_threshold(threshold)
    result = np.array(raw, copy=True)
    if result.ndim < 1 or result.shape[-1] != len(sensors) or not np.isfinite(result).all():
        raise ValueError('Invalid raw prediction shape or values')
    for name in states:
        index = sensors.index(name)
        result[..., index] = result[..., index] > threshold
    return result


def _has_state_tokens(values, numeric):
    return (not np.isfinite(numeric).all()
            and any(isinstance(v, str) and v.strip().upper() in MAPPING for v in values))


def encode_state(values, name):
    result = []
    for value in values:
        token = value.strip().upper() if isinstance(value, str) else value
        if isinstance(token, str) and token in MAPPING:
            result.append(MAPPING[token]); continue
        try: number = float(token)
        except (TypeError, ValueError):
            raise ValueError(f'{name}: expected OPEN/CLOSE or encoded 0/1') from None
        if not np.isfinite(number) or number not in (0, 1):
            raise ValueError(f'{name}: invalid or missing OPEN/CLOSE value')
        result.append(number)
    return np.asarray(result, dtype=np.float64)


class StateEncoder:
    def __init__(self, config, infer=False):
        if infer and ENCODING_KEY in config: raise ValueError('Cannot refit a frozen OPEN/CLOSE encoding')
        self.config = config; self.infer = infer
        self.states = set(state_sensors(config)); self.incompatible = set()

    def column(self, name, values, numeric):
        tokens = _has_state_tokens(values, numeric)
        if not self.infer and tokens and name not in self.states:
            raise ValueError(f'{name}: OPEN/CLOSE type absent from frozen Train encoding')
        if name in self.states or (self.infer and tokens):
            if name in self.incompatible:
                raise ValueError(f'{name}: inconsistent OPEN/CLOSE type across Train Runs')
            self.states.add(name)
            return encode_state(values, name)
        if self.infer and (not np.isfinite(numeric).all() or np.any((numeric != 0) & (numeric != 1))):
            self.incompatible.add(name)
        return numeric

    def freeze(self):
        if self.infer:
            self.config[ENCODING_KEY] = dict(schema_version=1, mapping=MAPPING.copy(),
                sensors=[n for n in self.config['sensors'] if n in self.states], fit_partition='train')
