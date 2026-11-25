"""Resolve human-editable groups to fixed sensor channels, never from Test values."""
from collections import Counter, defaultdict
from difflib import get_close_matches
import json
from pathlib import Path
import re

from .tips_data import ROOT, contained, read_csv, sha, verify_split

HERE = Path(__file__).resolve().parent
DEFAULT_PROFILE = ROOT / 'config/sensor_profile.json'
DICTIONARY = ROOT / 'config/sensor_dictionary.json'


def sensor_candidate(name):
    return bool(re.match(r'^(Temp_|MFC|AUX|VG\d+[ _]|O2Density_|APC Valve|ValveAct_)', name))


def source_columns(names, sensors):
    """Only trim surrounding whitespace. Never silently choose between aliases."""
    mapping = defaultdict(list)
    for name in names: mapping[name.strip()].append(name)
    result = {}
    for sensor in sensors:
        found = mapping.get(sensor.strip(), [])
        if len(found) != 1: raise ValueError(f'Missing or ambiguous sensor column: {sensor!r}: {found}')
        result[sensor] = found[0]
    return result


def resolve(profile, dictionary):
    if not isinstance(profile, dict) or set(profile) != {'schema_version', 'groups', 'include', 'exclude'}:
        raise ValueError('Profile requires exactly schema_version, groups, include, exclude')
    if type(profile['schema_version']) is not int or profile['schema_version'] != 1:
        raise ValueError('Unsupported sensor profile schema_version')
    for key in ['groups', 'include', 'exclude']:
        if not isinstance(profile[key], list) or any(not isinstance(x, str) or not x.strip() for x in profile[key]):
            raise ValueError(f'{key} must be an array of nonempty strings')
    names = [x['name'].strip() for x in dictionary]
    if len(names) != len(set(names)): raise ValueError('Duplicate dictionary sensor name')
    groups = {x['group'] for x in dictionary}
    requested_groups = set(profile['groups'])
    unknown_groups = requested_groups - groups - {'all'}
    if unknown_groups: raise ValueError(f'Unknown groups: {sorted(unknown_groups)}; available: {sorted(groups)} or all')
    included = {x.strip() for x in profile['include']}
    excluded = {x.strip() for x in profile['exclude']}
    unknown = (included | excluded) - set(names)
    if unknown:
        raise ValueError('Unknown sensor names: ' + str({n: get_close_matches(n, names, n=3) for n in sorted(unknown)}))
    forbidden = [n for n in included if not sensor_candidate(n)]
    if forbidden: raise ValueError(f'Process metadata cannot be forecast sensors: {sorted(forbidden)}')
    selected = []; columns = []
    for item in dictionary:
        name = item['name'].strip(); eligible = sensor_candidate(name)
        requested = 'all' in requested_groups or item['group'] in requested_groups or name in included
        if not eligible: reason = 'protected_metadata'
        elif name in excluded: reason = 'explicit_exclude'
        elif requested:
            reason = 'explicit_include' if name in included else 'group'
            selected.append(name)
        else: reason = 'not_requested'
        columns.append(dict(name=name, group=item['group'], candidate=eligible,
                            selected=name in selected, reason=reason, unit=item.get('unit', '')))
    if not selected: raise ValueError('Sensor profile selects no sensors after exclusions')
    return dict(selected_sensors=selected, selected_count=len(selected), candidate_count=sum(c['candidate'] for c in columns),
                group_counts=dict(Counter(c['group'] for c in columns if c['selected'])), columns=columns,
                ordering='sensor dictionary order, independent of group/include order; exclude wins',
                limitations=['Candidates are not automatically valid continuous targets; settings/actuators remain selectable.',
                             'Schema availability only; no constants/missingness-based selection and no Valid/Test feature inspection.'])


def prepare_selection(split_dir, profile_path=None, explicit=None, location=None):
    import pyarrow.parquet as pq
    dictionary = json.loads(DICTIONARY.read_text(encoding='utf-8-sig'))
    if explicit is not None:
        if len(explicit) != len(set(x.strip() for x in explicit)): raise ValueError('Duplicate explicit sensor')
        document = dict(schema_version=1, groups=[], include=explicit, exclude=[])
        mode, profile_hash = 'explicit', None
    else:
        profile_path = Path(profile_path or DEFAULT_PROFILE).resolve()
        document = json.loads(profile_path.read_text(encoding='utf-8-sig'))
        mode, profile_hash = 'profile', sha(profile_path)
    report = resolve(document, dictionary)
    if location is None:
        _, source, source_m = verify_split(split_dir)
        rows = read_csv(Path(split_dir) / 'train/runs.csv')
    else:
        source,source_m,rows = location.source,location.source_manifest,location.rows
    aliases = defaultdict(set)
    for row in rows:
        path = location.row_path(row) if location is not None else contained(ROOT, row['parquet_path'])
        relative = path.relative_to(source).as_posix()
        if relative not in source_m['files']: raise ValueError('Train path absent from source manifest')
        mapping = source_columns(pq.read_schema(path).names, report['selected_sensors'])
        for name, physical in mapping.items(): aliases[name].add(physical)
    if not rows: raise ValueError('No Train Runs for sensor schema verification')
    report.update(mode=mode, profile_snapshot=document, profile_sha256=profile_hash,
                  dictionary_sha256=sha(DICTIONARY), train_schemas_checked=len(rows),
                  source_columns={n: sorted(v) for n, v in aliases.items()})
    return report
