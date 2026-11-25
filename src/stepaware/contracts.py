"""Frozen field mappings and Train-only artifact contracts."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODEL = ROOT / 'checkpoints' / 'TSV-28M'
FORBIDDEN = {'ProcessRecipeStepTime', 'ProcessRecipeStepRemainTime',
             'ProcessRecipeTotalTime', 'ProcessRecipeEndRemainTime',
             'ObservableTimestamp', '_cycle_id', 'ActiveCjobID', 'ActivePjobID'}
RESERVED = ['<PAD>', '<NONE>', '<UNKNOWN>']
CATEGORIES = ['recipe', 'main_id', 'occurrence', 'main_position', 'subrecipe',
              'sub_id', 'sub_position', 'phase', 'quality']
NUMERIC = ['log_K', 'log_k', 'log_remaining_loops', 'loop_fraction',
           'log_elapsed_main', 'log_elapsed_macro', 'log_elapsed_loop', 'log_elapsed_substep',
           'log_previous_loop_duration', 'K_known', 'loop_valid', 'main_elapsed_known',
           'macro_elapsed_known', 'loop_elapsed_known', 'substep_elapsed_known',
           'previous_loop_known', 'occurrence_known', 'state_conflict']


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_profile(path=HERE/'process_context_profile.json'):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if (value['context'], value['horizon'], value['patch_size'], value['state_embedding_dim']) != (128, 30, 32, 64):
        raise ValueError('This C/P contract requires Context128/H30/P32/d64')
    used = set(value['process_columns'].values()) | set(value['audit_columns'].values()) | set(value['absolute_set_columns'])
    if used & FORBIDDEN or value['inject_future']:
        raise ValueError('Future renderer and unreliable duration columns are excluded')
    if not 0 <= value['recipe_dropout_probability'] < 1:
        raise ValueError('Invalid recipe dropout')
    return value


def key(*values):
    return json.dumps(values, ensure_ascii=False, separators=(',', ':'))


def token(value):
    if value is None or str(value).strip().lower() in ('', 'nan', 'none', 'null'):
        return None
    return str(value).strip()


def integer(value):
    value = token(value)
    if value is None:
        return None
    try:
        n = float(value)
        return int(n) if n.is_integer() and n >= 0 else None
    except (ValueError, OverflowError):
        return None
