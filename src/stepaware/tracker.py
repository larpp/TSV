"""One causal state transition function for CSV replay and online observations."""
import copy
import math
from datetime import datetime

from .contracts import CATEGORIES, NUMERIC, digest, integer, key, token


def timestamp_us(value):
    # Preserve the input time basis; never silently translate naive timestamps.
    dt = datetime.fromisoformat(str(value))
    if dt.tzinfo is not None:
        raise ValueError('Expected the frozen naive Timestamp basis, not mixed timezone input')
    return (dt - datetime(1970, 1, 1)) // __import__('datetime').timedelta(microseconds=1)


def parse_fields(row, profile):
    cols = profile['process_columns']; audit = profile['audit_columns']
    missing = (set(cols.values()) | set(audit.values())) - set(row)
    if missing:
        raise ValueError('Required process columns missing: ' + ', '.join(sorted(missing)))
    d = {name: token(row[col]) for name, col in cols.items()}
    d['timestamp'] = timestamp_us(d['timestamp'])
    for n in ('main_step_seq', 'main_step_id', 'substep_seq', 'substep_id', 'loop_setting', 'loop_index'):
        d[n] = integer(d[n])
    d.update(run_id=token(row[audit['run_id']]), observed=str(row[audit['row_observed']]) == '1',
             metadata_valid=str(row[audit['state_valid']]) == '1',
             main_name=token(row[audit['main_step_name']]))
    if d['run_id'] is None:
        raise ValueError('An explicit Run identity is required')
    return d


class StateTracker:
    def __init__(self, profile, schema, *, start_observed=False):
        self.profile = profile; self.schema = schema
        self.start_observed = start_observed
        self.reset()

    def reset(self):
        self.run_id = None; self.last_time = None; self.last = None
        self.counters = {}; self.occurrence_known = False; self.after_sub = False
        self.clocks = dict.fromkeys(('main', 'macro', 'loop', 'substep'))
        self.previous_loop = None; self.loop_clean = False; self.gap = False

    def save_state(self):
        return dict(version=1, contract=digest([self.profile, self.schema]),
                    state=copy.deepcopy({k:v for k,v in self.__dict__.items() if k not in ('profile', 'schema')}))

    def load_state(self, saved):
        if saved.get('version') != 1 or saved.get('contract') != digest([self.profile, self.schema]):
            raise ValueError('Tracker state belongs to another input/Train-schema contract')
        expected = set(self.__dict__) - {'profile', 'schema'}
        if set(saved['state']) != expected:
            raise ValueError('Invalid tracker state fields')
        self.__dict__.update(copy.deepcopy(saved['state']))

    def update(self, row):
        d = parse_fields(row, self.profile); t = d['timestamp']
        new_run = d['run_id'] != self.run_id
        if new_run:
            self.reset(); self.run_id = d['run_id']
        if self.last_time is not None and t <= self.last_time:
            raise ValueError('Duplicate/out-of-order observation; tracker was not advanced')
        contiguous = self.last_time is not None and t-self.last_time <= self.profile['maximum_observation_gap_seconds']*1e6 and not self.gap
        if self.last_time is not None and not contiguous:
            self.clocks = dict.fromkeys(self.clocks); self.occurrence_known = False
            self.previous_loop = None; self.loop_clean = False
        self.last_time = t
        recipe = d['main_recipe'] or (self.last['recipe'] if self.last else None)
        if not d['observed']:
            self.gap = True; self.occurrence_known = False; self.loop_clean = False; self.previous_loop = None
            self.clocks = dict.fromkeys(self.clocks)
            return self._state(t, recipe, None, None, None, None, None, 'UNKNOWN', None, None,
                               False, False, 'MISSING', True)
        seq, mid = d['main_step_seq'], d['main_step_id']
        K, k = d['loop_setting'], d['loop_index']; execution = d['execution_recipe']
        main_key = (recipe, seq, mid)
        old = self.last
        changed_main = old is None or main_key != (old['recipe'], old['main_seq'], old['main_id'])
        main_valid = recipe is not None and seq is not None and mid is not None
        if changed_main:
            self.after_sub = False; self.previous_loop = None; self.loop_clean = False
            if old is None:
                self.occurrence_known = bool(self.start_observed and key(recipe, seq, mid) in self.schema['start_keys'])
            if main_valid:
                self.counters = dict(self.counters)
                self.counters[str(mid)] = self.counters.get(str(mid), 0)+1
            self.clocks['main'] = t if main_valid and (contiguous or self.occurrence_known) else None
        conflict = not main_valid
        sub_path = bool(execution and execution.startswith('SubRecipe/'))
        loop_valid = bool(d['metadata_valid'] and sub_path and K is not None and k is not None and K > 0 and 1 <= k <= K)
        K_known = bool(d['metadata_valid'] and sub_path and K is not None and K > 0)
        subrecipe = execution if sub_path else None
        ss = sid = None
        if sub_path:
            phase = 'SUB'; conflict |= not loop_valid
            ss, sid = d['substep_seq'], d['substep_id']
            # Exact known valid Sub key wins. Unknown new Sub IDs stay observable UNKNOWN-vocab,
            # but stale Main fields at Sub entry cannot masquerade as a Substep.
            known_sub = key(execution, ss, sid) in self.schema['substep_keys']
            stale_main = ss == seq and sid == mid and not known_sub
            if stale_main or ss is None or sid is None:
                ss = sid = None; conflict = True
        elif execution == recipe and K == 0 and k == 0:
            if old and not changed_main and old['phase'] == 'SUB':
                self.after_sub = True
            phase = 'MAIN_AFTER_SUB' if self.after_sub else 'MAIN'
        else:
            phase = 'UNKNOWN'; conflict = True
        macro = (recipe, seq, phase, subrecipe)
        macro_change = old is None or macro != tuple(old['macro_key'])
        loop_change = changed_main or macro_change or old['k'] != k if old else True
        sub_change = loop_change or (old and (old['sub_seq'], old['sub_id']) != (ss, sid))
        if macro_change:
            self.clocks['macro'] = t if main_valid and phase != 'UNKNOWN' and (contiguous or self.occurrence_known and old is None) else None
        if loop_change:
            complete = bool(old and not changed_main and not macro_change and contiguous and self.loop_clean
                            and old['loop_valid'] and loop_valid and old['K'] == K and k == old['k']+1
                            and self.clocks['loop'] is not None)
            self.previous_loop = (t-self.clocks['loop'])/1e6 if complete else None
            self.clocks['loop'] = t if loop_valid and contiguous else None
            self.loop_clean = bool(loop_valid and contiguous and not conflict)
        if sub_change:
            self.clocks['substep'] = t if ss is not None and contiguous else None
        if old and not macro_change and phase == 'SUB' and old['K'] != K:
            self.previous_loop = None; self.loop_clean = False
        if conflict or not d['metadata_valid']:
            self.loop_clean = False
        if phase != 'SUB':
            self.clocks['loop'] = self.clocks['substep'] = None; self.previous_loop = None
        self.gap = not main_valid or phase == 'UNKNOWN'
        if self.gap:
            self.occurrence_known = False; self.clocks = dict.fromkeys(self.clocks)
        quality = 'UNKNOWN' if conflict else 'OBSERVED'
        state = self._state(t, recipe, seq, mid, subrecipe, ss, sid, phase, K, k,
                            K_known, loop_valid, quality, conflict)
        self.last = state
        return state

    def _state(self, t, recipe, seq, mid, subrecipe, ss, sid, phase, K, k, K_known, loop_valid, quality, conflict):
        elapsed = {name: max(0., (t-clock)/1e6) if clock is not None else 0. for name,clock in self.clocks.items()}
        return dict(timestamp_us=t, run_id=self.run_id, recipe=recipe, main_seq=seq, main_id=mid,
            subrecipe=subrecipe, sub_seq=ss, sub_id=sid, phase=phase, quality=quality,
            macro_key=[recipe, seq, phase, subrecipe], K=K, k=k, K_known=K_known, loop_valid=loop_valid,
            elapsed=elapsed, elapsed_known={n:v is not None for n,v in self.clocks.items()},
            previous_loop_duration=self.previous_loop or 0., previous_loop_known=self.previous_loop is not None,
            occurrence=self.counters.get(str(mid)) if self.occurrence_known else None,
            occurrence_known=self.occurrence_known, counters=self.counters, state_conflict=bool(conflict),
            terminal=key(recipe, seq, mid) in self.schema['end_keys'])

    def parse_run(self, rows):
        return [self.update(row) for row in rows]


def category_values(state):
    unk = '<UNKNOWN>'; none = '<NONE>'
    recipe = state['recipe'] or unk; mid = state['main_id']
    active = state['phase'] == 'SUB'; inactive = state['phase'] in ('MAIN', 'MAIN_AFTER_SUB')
    return dict(recipe=recipe, main_id=str(mid) if mid is not None else unk,
        occurrence=key(mid, state['occurrence']) if mid is not None and state['occurrence_known'] else unk,
        main_position=key(recipe, state['main_seq']) if state['main_seq'] is not None else unk,
        subrecipe=state['subrecipe'] or (none if inactive else unk),
        sub_id=str(state['sub_id']) if active and state['sub_id'] is not None else (none if inactive else unk),
        sub_position=key(state['subrecipe'],state['sub_seq']) if active and state['sub_seq'] is not None else (none if inactive else unk),
        phase=state['phase'] if state['phase'] != 'UNKNOWN' else unk, quality=state['quality'])


def numeric_values(s):
    valid=s['loop_valid']; K=s['K']; k=s['k']
    log=lambda x: math.log1p(max(0.,x))
    vals=[log(K) if s['K_known'] else 0., log(k) if valid else 0., log(K-k) if valid else 0., k/K if valid else 0.]
    vals += [log(s['elapsed'][n]) if s['elapsed_known'][n] else 0. for n in ('main','macro','loop','substep')]
    vals += [log(s['previous_loop_duration']) if s['previous_loop_known'] else 0., float(s['K_known']), float(valid)]
    vals += [float(s['elapsed_known'][n]) for n in ('main','macro','loop','substep')]
    vals += [float(s['previous_loop_known']),float(s['occurrence_known']),float(s['state_conflict'])]
    assert len(vals)==len(NUMERIC)
    return vals
