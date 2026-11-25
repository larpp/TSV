"""Auditable CSV-only deposition dataset. Python standard library; no model imports."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
import platform

ROOT = Path(__file__).resolve().parents[2]
RECIPES = ('0_P_IH_HD02', '0_P_IH_HD02-1')
PARTITIONS = ('train', 'valid', 'test')
META = {'Timestamp', 'ObservableTimestamp', 'EquipmentStatus', 'AlarmState', 'PMstatus',
        'PPExecname', 'PPExecStepSeqNo', 'PPExecStepID', 'PPExecStepName', 'ActiveCjobID',
        'ActivePjobID', 'PMStoredProcessRecipeName', 'PJobProcessingState',
        'SubRecipeLoopSettingValue', 'SubRecipeLoopMoniterValue'}
EXTRA = ['_run_id', '_source_row', '_source_timestamp', '_elapsed_seconds',
         '_delta_seconds', '_row_observed', '_row_usable', '_context_valid',
         '_segment_id', '_cycle_id', '_quality', '_missing_sensor_columns']
NULLS = {'', 'nan', 'null', 'none', 'inf', '+inf', '-inf', 'infinity', '-infinity'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_records(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)


def number(v):
    try:
        x = float(v)
        return int(x) if math.isfinite(x) and x.is_integer() else None
    except (ValueError, TypeError): return None


def iso(t): return t.isoformat(sep=' ', timespec='microseconds')


def compressed(values):
    out = []
    for v in values:
        if not out or out[-1] != v: out.append(v)
    return out


def sensor_columns(header):
    return [c for c in header if c not in META and not c.startswith('ProcessRecipe')]


def load(path):
    """Keep strings and original line positions; reject ragged rows instead of repairing them."""
    with path.open(encoding='utf-8-sig', newline='') as f:
        rd = csv.reader(f)
        original = next(rd, [])
        header = [s.strip() for s in original]
        if len(header) != 166: return header, [], 'column_count'
        if len(set(header)) != len(header) or any(not c or c.startswith('_') for c in header):
            return header, [], 'ambiguous_header'
        rows = []
        for line, row in enumerate(rd, 2):
            if len(row) != len(header): return header, [], 'ragged_csv_row'
            rows.append((line, dict(zip(header, row))))
    return header, rows, None


def inspect(path, source):
    header, rows, error = load(path)
    info = dict(source_path=path.relative_to(source).as_posix(), source_sha256=sha(path),
                columns=len(header), source_rows=len(rows), status='excluded', reason=error or '')
    if error: return info
    required = (META - {'EquipmentStatus', 'AlarmState', 'PMstatus', 'PJobProcessingState'}) | {
        'ProcessRecipeStepSeqNo', 'ProcessRecipeStepName'}
    if required - set(header): info['reason'] = 'required_columns_missing'; return info
    if not rows: info['reason'] = 'empty_csv'; return info
    recipes = {r['PMStoredProcessRecipeName'].strip() for _, r in rows}
    if len(recipes) != 1 or next(iter(recipes)) not in {'MainRecipe/RUN/' + r for r in RECIPES}:
        info['reason'] = 'recipe_not_selected_or_changes'; info['recipes'] = sorted(recipes); return info
    info['recipe'] = next(iter(recipes)).split('/')[-1]
    try: times = [datetime.fromisoformat(r['Timestamp']) for _, r in rows]
    except ValueError: info['reason'] = 'invalid_timestamp'; return info
    if any(t.tzinfo is not None for t in times): info['reason'] = 'unexpected_timezone'; return info
    if any(b <= a for a, b in zip(times, times[1:])):
        info['reason'] = 'nonincreasing_timestamp'; return info
    steps = [number(r['ProcessRecipeStepSeqNo']) for _, r in rows]
    missing = sorted(set(range(1, 43)) - set(steps))
    info['missing_main_steps'] = missing
    if 1 not in steps or 42 not in steps:
        info['reason'] = 'start_end_boundary_missing'; return info
    first, last = steps.index(1), len(steps) - 1 - steps[::-1].index(42)
    # Only explicit STANDBY before START and END/idle after END may be trimmed.
    if any(s != 0 for s in steps[:first]) or any(s not in (0, 43, 255) for s in steps[last+1:]):
        info['reason'] = 'unexpected_outside_run_steps'; return info
    sequence = compressed(steps[first:last+1])
    if set(sequence) == {1, 42}:
        info['reason'] = 'start_end_only'; return info
    if (any(s is None or not 1 <= s <= 42 for s in sequence)
            or any(b <= a for a,b in zip(sequence,sequence[1:]))):
        info['reason'] = 'main_step_order_or_multiple_runs'; return info
    if rows[first][1]['ProcessRecipeStepName'].strip() != 'START' or rows[last][1]['ProcessRecipeStepName'].strip() != 'END':
        info['reason'] = 'start_end_name_mismatch'; return info
    jobs = {r['ActivePjobID'].strip() for _, r in rows[first:last+1] if r['ActivePjobID'].strip()}
    if len(jobs) > 1: info['reason'] = 'changing_process_job'; return info
    info.update(status='selected', reason='', first_source_row=rows[first][0], last_source_row=rows[last][0],
                trimmed_rows=first + len(rows)-last-1, start=iso(times[first]), end=iso(times[last]), header=header)
    return info


def retained_rows(path, info):
    header, rows, error = load(path)
    if error or header != info['header'] or sha(path) != info['source_sha256']:
        raise ValueError('Source changed: ' + str(path))
    return [(n, r) for n, r in rows if info['first_source_row'] <= n <= info['last_source_row']]


def chronological_split(infos, ratios):
    if len(ratios) != 3 or any(r <= 0 for r in ratios) or not math.isclose(sum(ratios), 1.0):
        raise ValueError('Three positive ratios must sum to 1')
    ordered = sorted(infos, key=lambda r: (r['start'], r['source_path']))
    for a, b in zip(ordered, ordered[1:]):
        if a['end'] >= b['start']: raise ValueError('Overlapping Runs: ' + a['source_path'] + ', ' + b['source_path'])
    quota = [len(ordered) * r for r in ratios]
    sizes = [int(q) for q in quota]
    for i in sorted(range(3), key=lambda i: (-(quota[i]-sizes[i]), i))[:len(ordered)-sum(sizes)]: sizes[i] += 1
    if min(sizes) == 0: raise ValueError('Not enough Runs for three nonempty partitions')
    at = 0
    for split, size in zip(PARTITIONS, sizes):
        for info in ordered[at:at+size]: info['split'] = split
        at += size
    return ordered


def cycles_of(rows, info):
    cycles = []; active = None
    for idx, (line, row) in enumerate(rows):
        if not row['PPExecname'].strip().startswith('SubRecipe/'):
            active = None; continue
        key = (number(row['ProcessRecipeStepSeqNo']), row['PPExecname'].strip(),
               number(row['SubRecipeLoopSettingValue']), number(row['SubRecipeLoopMoniterValue']))
        if active is None or active['key'] != key:
            active = dict(key=key, indices=[], observed=[], cycle_id=f"{info['run_id']}_c{len(cycles)+1:04d}")
            cycles.append(active)
        active['indices'].append(idx)
        seq = number(row['PPExecStepSeqNo'])
        if not active['observed'] or active['observed'][-1] != seq: active['observed'].append(seq)
    return cycles


def reference_key(info, cycle):
    return '|'.join((info['recipe'], str(cycle['key'][0]), cycle['key'][1]))


def fit_references(source, infos):
    """Use only Train structure; never learn templates from validation/test observations."""
    patterns = defaultdict(Counter); supports = defaultdict(set)
    for info in infos:
        if info['split'] != 'train': continue
        for cycle in cycles_of(retained_rows(source / info['source_path'], info), info):
            seq = cycle['observed']; key = reference_key(info, cycle)
            # Candidate complete ascending substep sequences; modal candidate is a provisional reference.
            if len(seq) >= 2 and seq == list(range(len(seq))):
                patterns[key][tuple(seq)] += 1
                supports[(key, tuple(seq))].add(info['run_id'])
    return {key: dict(expected=list(seq), supporting_cycles=count,
                     supporting_train_runs=sorted(supports[(key, seq)]))
            for key, counter in sorted(patterns.items())
            for seq, count in [sorted(counter.items(), key=lambda p: (-p[1], p[0]))[0]]}


def classify(rows, info, refs, gap_threshold=2.0):
    cycles = cycles_of(rows, info); annotations = {}; ids = {}; audit = []; previous_cycles = {}
    times = [datetime.fromisoformat(r['Timestamp']) for _,r in rows]
    gap_indices = {i for i in range(1,len(rows)) if (times[i]-times[i-1]).total_seconds() >= gap_threshold}
    groups = defaultdict(list)
    for cycle in cycles:
        main, recipe, setting, current = cycle['key']; key = reference_key(info, cycle)
        expected = refs.get(key, {}).get('expected'); reasons = []
        if expected is None: reasons.append('substep_reference_unavailable')
        elif cycle['observed'] != expected: reasons.append('substep_missing_or_order_mismatch')
        if setting is None or setting <= 0 or current is None or not 1 <= current <= setting:
            reasons.append('invalid_loop_metadata')
        first, last = cycle['indices'][0], cycle['indices'][-1]
        context = (main, recipe)
        previous = previous_cycles.get(context)
        if previous is not None:
            previous_setting, previous_counter = previous
            if (setting != previous_setting or current is None or previous_counter is None
                    or current != previous_counter + 1):
                reasons.append('loop_counter_boundary')
        previous_cycles[context] = (setting, current)
        for idx in cycle['indices']:
            ids[idx] = cycle['cycle_id']
            if reasons: annotations[idx] = reasons
        groups[(main, recipe, setting)].append(current)
        audit.append(dict(run_id=info['run_id'], split=info['split'], cycle_id=cycle['cycle_id'],
            main_step=main, sub_recipe=recipe, loop_setting=setting, loop_counter=current,
            first_source_row=rows[first][0], last_source_row=rows[last][0],
            start=rows[first][1]['Timestamp'], end=rows[last][1]['Timestamp'],
            rows=len(cycle['indices']), status='retained',
            has_time_gap=int(any(first <= i <= last+1 for i in gap_indices)),
            action='preserve_observed_values',
            reason=';'.join(reasons), expected_substeps=json.dumps(expected),
            observed_substeps=json.dumps(cycle['observed']),
            missing_substeps=json.dumps(sorted(set(expected or []) - set(cycle['observed'])))))
    loop_audit = []
    for (main, recipe, setting), counters in groups.items():
        missing = sorted(set(range(1, setting+1)) - set(counters)) if setting and setting > 0 else []
        if missing or any(counters.count(c) > 1 for c in counters):
            loop_audit.append(dict(run_id=info['run_id'], main_step=main, sub_recipe=recipe,
                setting=setting, observed_counters=counters, missing_counters=missing,
                note='No timestamps invented for wholly unobserved cycles; use time-gap masks.'))
    return annotations, ids, audit, loop_audit


def make_rows(rows, info, refs, gap_threshold):
    header = info['header']; sensors = sensor_columns(header)
    annotations, ids, cycles, loop_audit = classify(rows, info, refs, gap_threshold)
    cycle_by_id = {c['cycle_id']:c for c in cycles}
    start = datetime.fromisoformat(rows[0][1]['Timestamp'])
    stats = Counter(); gaps = []; segment = 0; was_usable = False; previous = None; previous_output = None

    def metadata(t, line='', source_time='', observed=0, usable=0, context=0, cid='', quality='', missing=''):
        nonlocal segment, was_usable, previous_output
        if usable and not was_usable: segment += 1
        meta = dict(zip(EXTRA, [info['run_id'], line, source_time, f'{(t-start).total_seconds():.6f}',
                    '' if previous_output is None else f'{(t-previous_output).total_seconds():.6f}',
                    observed, usable, context, segment if usable else '', cid, quality, missing]))
        was_usable = bool(usable); previous_output = t
        stats['output_rows'] += 1; stats['usable_rows'] += usable
        return meta

    def generate():
        nonlocal previous, was_usable
        for idx, (line, raw) in enumerate(rows):
            t = datetime.fromisoformat(raw['Timestamp'])
            if previous is not None:
                delta = (t-previous).total_seconds()
                if delta >= gap_threshold:
                    # Preserve millisecond observations; insert nominal absent seconds, never interpolate values.
                    count = max(1, int(math.floor(delta + 0.5)) - 1)
                    if count > 86400: raise ValueError('Gap exceeds one day; inspect source instead of allocating blindly')
                    gaps.append(dict(run_id=info['run_id'], split=info['split'], before=iso(previous), after=iso(t),
                        seconds=delta, inserted_rows=count, before_source_row=rows[idx-1][0], after_source_row=line,
                        before_main_step=rows[idx-1][1]['ProcessRecipeStepSeqNo'], after_main_step=raw['ProcessRecipeStepSeqNo'],
                        before_cycle_id=ids.get(idx-1,''), after_cycle_id=ids.get(idx,''),
                        first_inserted=iso(previous+timedelta(seconds=1)), last_inserted=iso(previous+timedelta(seconds=count))))
                    main_before = number(rows[idx-1][1]['ProcessRecipeStepSeqNo'])
                    main_after = number(raw['ProcessRecipeStepSeqNo'])
                    main_missing = list(range(main_before+1,main_after)) if main_before is not None and main_after is not None and main_after>main_before else []
                    related = [cycle_by_id[c] for c in sorted({ids.get(idx-1,''),ids.get(idx,'')}) if c in cycle_by_id]
                    sub_missing = {c['cycle_id']:json.loads(c['missing_substeps']) for c in related if json.loads(c['missing_substeps'])}
                    gaps[-1].update(missing_main_steps=json.dumps(main_missing),
                        related_missing_substeps=json.dumps(sub_missing),
                        step_missing_scope=';'.join(s for s,condition in [('main',main_missing),('subcycle',sub_missing)] if condition) or 'none_detected')
                    for k in range(1, count+1):
                        blank_time = previous + timedelta(seconds=k)
                        if blank_time >= t: raise ValueError('Invalid gap reconstruction')
                        blank = dict.fromkeys(header, ''); blank['Timestamp'] = iso(blank_time)
                        blank.update(metadata(blank_time, quality='timestamp_gap'))
                        stats['inserted_gap_rows'] += 1
                        yield blank
            out = raw.copy(); missing = [c for c in sensors if raw[c].strip().lower() in NULLS]
            for c in missing: out[c] = ''
            quality = list(annotations.get(idx, []))
            if idx:
                prev_step = number(rows[idx-1][1]['ProcessRecipeStepSeqNo'])
                this_step = number(raw['ProcessRecipeStepSeqNo'])
                if prev_step is not None and this_step is not None and this_step>prev_step+1:
                    quality.append('main_steps_not_observed')
            if missing: quality.append('source_sensor_missing')
            # Step warnings never erase observations or split otherwise continuous time.
            usable = int(not missing)
            if idx in annotations or 'main_steps_not_observed' in quality: stats['step_warning_rows_preserved'] += 1
            out['Timestamp'] = iso(t)
            context_valid = int('invalid_loop_metadata' not in quality)
            out.update(metadata(t, line, raw['Timestamp'], 1, usable, context_valid,
                                ids.get(idx,''), ';'.join(quality), ';'.join(missing)))
            stats['source_rows_in_run'] += 1
            yield out
            previous = t
        stats['usable_segments'] = segment
    return generate(), stats, cycles, gaps, loop_audit


class DigestText:
    def __init__(self): self.hash = hashlib.sha256(b'\xef\xbb\xbf')
    def write(self, value): self.hash.update(value.encode('utf-8')); return len(value)


def serialize(stream, header, iterator):
    w = csv.DictWriter(stream, fieldnames=header + EXTRA)
    w.writeheader(); w.writerows(iterator)


def main_step_issues(rows, info, gap_threshold):
    issues = []
    for (n0,r0),(n1,r1) in zip(rows,rows[1:]):
        a,b = number(r0['ProcessRecipeStepSeqNo']),number(r1['ProcessRecipeStepSeqNo'])
        if a is not None and b is not None and b>a+1:
            seconds = (datetime.fromisoformat(r1['Timestamp'])-datetime.fromisoformat(r0['Timestamp'])).total_seconds()
            issues.append(dict(run_id=info['run_id'], split=info['split'], scope='main', cycle_id='',
                main_step=f'{a}->{b}', loop_counter='', start=r0['Timestamp'],end=r1['Timestamp'],
                missing_steps=json.dumps(list(range(a+1,b))), has_time_gap=int(seconds>=gap_threshold),
                reason='main_steps_not_observed',action='preserve_observed_values',
                before_source_row=n0,after_source_row=n1))
    return issues


def report_markdown(dest, selected, cycles, gaps, issues, summary, gap_threshold):
    by_id = {r['run_id']:r for r in selected}
    def file_link(rid):
        r=by_id[rid]
        return f"[{Path(r['source_path']).name}](<{r['csv_path']}>)"
    def md(value): return str(value).replace('|','/').replace('\n',' ')
    def table(headers, records):
        return ['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |'] + [
            '| '+' | '.join(md(v) for v in row)+' |' for row in records]
    lines=['# CSV 시간 결측 처리 수정 결과','',
        '## 적용 기준','',
        '- Run 번호는 runs.csv를 따른다. run_id_mapping.csv가 있으면 이전→현재 번호 대응표이며, 파일명·행 내부 Run/Cycle ID·감사 파일은 현재 번호를 사용한다.',
        '- Main Step이 START(1)와 END(42)만 있는 Run은 시간 공백 유무와 관계없이 제외한다. 제외 이유는 exclusions.json의 start_end_only다.',
        f'- START/END-only 제외 파일 수: {summary["exclusion_reasons"].get("start_end_only",0)}개. 그 외 중간 Main/Substep 미관측만으로 Run을 제외하거나 관측 행을 비우지 않는다.',
        f'- 실제 관측 시각 차이가 {gap_threshold:g}초 이상인 곳에만 빈 행을 추가한다. Step 미관측 동반 여부와 무관하게 실제 시간 공백을 표시한다.',
        '- 시간 공백과 Step 미관측이 함께 있어도 실제 관측된 앞뒤 센서값은 보존한다. 빈 행에는 Step·센서값을 추정해 넣지 않는다.',
        '- Substep 기준은 Train에서만 추정한 참고 순서다. 미관측은 실제 미실행/고장으로 확정하지 않는다.',
        '- 원래 밀리초 시각은 보존한다. 엄밀한 고정 1Hz 재표본화는 아니다. 원본 파일과 Timestral 모델은 변경하지 않았다.','',
        '## 파일 구성','']
    lines += table(['분할','Run','HD02','HD02-1','출력 행','추가 결측 행'],[
        [s,summary['partitions'][s]['runs'],summary['partitions'][s]['recipes'].get(RECIPES[0],0),
         summary['partitions'][s]['recipes'].get(RECIPES[1],0),summary['partitions'][s]['output_rows'],
         summary['partitions'][s]['inserted_gap_rows']] for s in PARTITIONS])
    lines += ['',f"총 {summary['kept_runs']} Run. 관측된 센서값을 Step 사유로 비운 행은 0개다.",
        '각 CSV는 동일한 원래 166열과 품질·출처 정보 12열을 가진다. 파일별 분할·출처는 [runs.csv](runs.csv)에 있다.','']
    examples=[c for c in cycles if 'process_0049_' in by_id[c['run_id']]['source_path'] and c['main_step']==14 and c['loop_counter']==4]
    if examples:
        c=examples[0]
        lines += ['## process_0049 확인','',f"{file_link(c['run_id'])}: Main Step 14 / Loop 4, {c['start']} ~ {c['end']}, "
            f"{c['rows']}행, 미관측 Substep {c['missing_substeps']}, 시간 공백 {bool(c['has_time_gap'])}. "
            '관측 센서값을 원본대로 보존한다. `_row_observed=1`, 센서 셀 자체 결측이 없으면 `_row_usable=1`이다.','']
    lines += [
        '## 실제 빈 행을 추가한 모든 구간','',
        '앞뒤 관측 시각 사이에만 빈 행이 들어간다. Substep 동반 표시는 공백 양쪽에 접한 Subcycle에서 '
        '미관측 Step이 발견됐다는 뜻이며, 그 Step이 바로 이 공백 안에서 실행됐다는 증거는 아니다.','']
    lines += table(['파일','앞 관측 → 뒤 관측','추가 행 시각','행 수','동반 미관측'],[
        [file_link(g['run_id']),g['before']+' → '+g['after'],g['first_inserted']+' ~ '+g['last_inserted'],
         g['inserted_rows'],g['step_missing_scope']+'; Main='+g['missing_main_steps']+'; Sub='+g['related_missing_substeps']] for g in gaps])
    for flag,title in [(0,'시간 공백 없이 Step만 미관측되어 그대로 보존한 모든 구간'),
                       (1,'시간 공백과 Step 미관측이 함께 확인된 구간')]:
        subset=[r for r in issues if r['has_time_gap']==flag]
        lines += ['',f'## {title}','',f'{len(subset)}개 구간. 관측된 센서값은 모두 보존한다.','']
        lines += table(['파일','종류','Main Step','Loop','관측 시각 범위','미관측 Step'],[
            [file_link(r['run_id']),r['scope'],r['main_step'],r['loop_counter'],r['start']+' ~ '+r['end'],r['missing_steps']] for r in subset])
    other=[c for c in cycles if c['reason'] and not json.loads(c['missing_substeps'])]
    lines += ['','## Step 순서·메타데이터 참고 경고','',
        '아래도 센서값을 보존했다. 단순 순서 차이·기준 미확보·Loop 메타데이터 문제를 센서 결측으로 바꾸지 않았다.','']
    lines += table(['파일','Main / Loop','관측 시각 범위','관측 Substep','경고'],[
        [file_link(c['run_id']),f"{c['main_step']} / {c['loop_counter']}",c['start']+' ~ '+c['end'],c['observed_substeps'],c['reason']] for c in other])
    lines += ['','## 해석과 검증 범위','',
        '- START/END 외에 중간 Main Step이 하나 이상 관측된 Run은 다른 중간 Step이 미관측이라는 이유만으로 제외하지 않는다. 보존은 정상 완료 인증을 뜻하지 않는다.',
        '- START와 END만 관측된 Run은 제외한다. 그 외 보존된 Run도 과거 120·미래 30 시점보다 짧으면 해당 설정의 학습 Window를 만들 수 없다.',
        '- `_quality`의 Step 경고는 감사 정보다. 이를 이유로 행을 제거하지 않는다. `_context_valid=0`은 실제 Loop 메타데이터가 잘못됐거나 삽입된 행일 때만 적용한다.',
        '- `_segment_id`는 시간 공백·실제 센서 결측에서만 끊는다. Step 누락·Loop 경고만으로 끊지 않는다.',
        '- 원본 해시·CSV 구조·관측 센서값 보존과 재생성 일치는 생성기 `--verify --reproduce`로 검증할 수 있다.',
        '- 이전 정책 데이터셋은 비교 근거로 보존하며 최신 폴더는 상위 latest.json을 따른다. Timestral-28M-ProcessContext는 CSV 관측 마스크를 지원한다.','']
    (dest/'processing_report.md').write_text('\n'.join(lines),encoding='utf-8')


def policy(ratios, gap):
    if not math.isfinite(gap) or gap < 2: raise ValueError('Gap threshold must be at least 2 seconds')
    return dict(version='mt-nit-csv-time-gaps-2', input_columns=166, recipes=list(RECIPES),
        main_step_column='ProcessRecipeStepSeqNo', substep_column='PPExecStepSeqNo',
        expected_main_steps=list(range(1,43)), main_rule='Exclude START/END-only Runs regardless of time gaps. Otherwise preserve missing interior Main Steps as audit only; require START/END boundaries and monotonic observed Main Steps; trim outside START..END.',
        run_filter_revision=3,
        subcycle_rule='Train-only modal contiguous Substep reference for audit only. Preserve all observed sensor values regardless of missing/reordered Substeps or loop-counter warnings; no Step-only segment break.',
        gap_threshold_seconds=gap, gap_rule='For delta >= threshold insert round-half-up(delta)-1 blank one-second rows; original timestamps unchanged.',
        sensor_rule='Keep all source sensors including OPEN/CLOSE and constants; no sensor selection/scaler fitting.',
        split=dict(unit='whole_run', order='start_timestamp', ratios=list(ratios), allocation='largest_remainder', shuffle=False),
        limitations=['Substep references are empirical Train-only templates, not approved Recipe definitions.',
            'Missingness is a collection/structure flag, not a process fault label.',
            'Original timestamp jitter remains; this is not a strictly resampled 1-Hz grid.',
            'Wholly missing loops have no fabricated durations or Step identities.',
            'Step nonobservation may reflect short sampling intervals or early termination; preserved does not mean confirmed complete/normal process.',
            'Time-gap and Substep co-occurrence is assessed within adjacent observed cycles; it does not prove a missing Step happened inside that gap.',
            'Timestral-28M-ProcessContext supports this CSV mask contract; comparison models retain their original input contract.',
            'Use timestamps and masks; never drop blanks, fill missing values with observed zeros, or score absent targets.',
            'These dates may overlap earlier experiments; this is a new dataset version, not a guaranteed untouched holdout.'])


def preserve_membership(selected, manifest_path, renumber=False):
    """Keep existing partitions, optionally assigning consecutive chronological IDs."""
    manifest_path = Path(manifest_path)
    previous = json.loads(manifest_path.read_text(encoding='utf-8'))['selected']
    lookup = {r['source_path']: r for r in previous}
    if len(lookup) != len(previous): raise ValueError('Duplicate prior source paths')
    ids = set(); partitions = []
    for index,info in enumerate(selected,1):
        old = lookup.get(info['source_path'])
        if old is None or old['source_sha256'] != info['source_sha256']:
            raise ValueError('Selected source missing or changed in prior membership')
        if old['run_id'] in ids or old['split'] not in PARTITIONS:
            raise ValueError('Invalid prior Run identity or split')
        ids.add(old['run_id']); info.update(run_id=old['run_id'],split=old['split'])
        if renumber:
            info.update(previous_run_id=old['run_id'],previous_csv_path=old['csv_path'],run_id=f'run_{index:04d}')
        partitions.append(PARTITIONS.index(old['split']))
    if partitions != sorted(partitions) or set(partitions) != {0,1,2}:
        raise ValueError('Prior membership is not chronological with three nonempty partitions')
    return dict(manifest_sha256=sha(manifest_path),
                rule=('Preserve split membership; assign consecutive chronological Run IDs and retain previous-to-new mapping.' if renumber else
                      'Preserve prior source-bound Run IDs and split membership; filtering only, no reassignment.'))


def build(source, output, ratios=(0.7,0.15,0.15), gap=2.0, membership_from=None, renumber_runs=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Source and output must be disjoint')
    if renumber_runs and not membership_from: raise ValueError('--renumber-runs requires --membership-from')
    files = sorted(source.rglob('*.csv'))
    if not files: raise ValueError('No source CSV files')
    pol = policy(ratios, gap); infos = []
    for i, path in enumerate(files,1):
        infos.append(inspect(path, source))
        if i % 25 == 0: print(f'Inspected {i}/{len(files)}', flush=True)
    selected = [r for r in infos if r['status'] == 'selected']
    if not selected: raise ValueError('No selected Runs')
    canonical = min(selected, key=lambda r:(r['start'],r['source_path']))['header']
    seen = set()
    for info in selected:
        if info['header'] != canonical: info.update(status='excluded', reason='schema_mismatch')
        elif info['source_sha256'] in seen: info.update(status='excluded', reason='duplicate_source')
        seen.add(info['source_sha256'])
    selected = chronological_split([r for r in selected if r['status']=='selected'], ratios)
    for i, info in enumerate(selected,1): info['run_id'] = f'run_{i:04d}'
    membership = preserve_membership(selected, Path(membership_from)/'manifest.json',renumber_runs) if membership_from else None
    if membership: pol['split']['preserved_membership'] = membership
    refs = fit_references(source, selected)
    identity = dict(policy=pol, source_files=[(r['source_path'],r['source_sha256']) for r in infos],
                    generator_sha256=sha(Path(__file__)))
    fingerprint = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
    dest = output / fingerprint
    if dest.exists(): raise FileExistsError('Dataset already exists; use --verify: ' + str(dest))
    for split in PARTITIONS: (dest/split).mkdir(parents=True, exist_ok=False)
    all_cycles=[]; all_gaps=[]; loops=[]; issues=[]
    for i, info in enumerate(selected,1):
        rows = retained_rows(source/info['source_path'], info)
        iterator, stats, cycles, gaps, loop_audit = make_rows(rows,info,refs,gap)
        rel = info['split']+'/'+info['run_id']+'__'+Path(info['source_path']).name
        with (dest/rel).open('x',encoding='utf-8-sig',newline='') as f: serialize(f,canonical,iterator)
        info.update(csv_path=rel, **stats, subcycles=len(cycles),
            subcycles_with_warnings=sum(bool(c['reason']) for c in cycles),
            masked_subcycles=0,masked_subcycle_rows=0)
        issues.extend(main_step_issues(rows,info,gap))
        issues.extend(dict(run_id=c['run_id'],split=c['split'],scope='subcycle',cycle_id=c['cycle_id'],
            main_step=c['main_step'],loop_counter=c['loop_counter'],start=c['start'],end=c['end'],
            missing_steps=c['missing_substeps'],has_time_gap=c['has_time_gap'],reason=c['reason'],
            action='preserve_observed_values',before_source_row=c['first_source_row'],after_source_row=c['last_source_row'])
            for c in cycles if json.loads(c['missing_substeps']))
        all_cycles.extend(cycles); all_gaps.extend(gaps); loops.extend(loop_audit)
        if i % 20 == 0: print(f'Wrote {i}/{len(selected)} Runs',flush=True)
    fields = ['run_id','split','recipe','start','end','source_path','source_sha256','csv_path',
              'source_rows','trimmed_rows','source_rows_in_run','output_rows','usable_rows','inserted_gap_rows',
              'masked_subcycle_rows','usable_segments','step_warning_rows_preserved','subcycles','subcycles_with_warnings','masked_subcycles']
    write_records(dest/'runs.csv',selected,fields)
    if renumber_runs:
        write_records(dest/'run_id_mapping.csv',selected,
                      ['previous_run_id','run_id','split','source_path','source_sha256','previous_csv_path','csv_path'])
    write_records(dest/'cycles.csv',all_cycles,list(all_cycles[0]) if all_cycles else ['run_id','cycle_id','status'])
    write_records(dest/'time_gaps.csv',all_gaps,list(all_gaps[0]) if all_gaps else ['run_id','before','after','seconds','inserted_rows'])
    write_records(dest/'step_issues.csv',issues,list(issues[0]) if issues else ['run_id','scope','missing_steps','has_time_gap','action'])
    excluded = [{k:v for k,v in r.items() if k!='header'} for r in infos if r['status']=='excluded']
    dump(dest/'exclusions.json',excluded); dump(dest/'loop_issues.json',loops)
    dump(dest/'substep_references.json',refs); dump(dest/'policy.json',pol)
    dump(dest/'schema.json',dict(original_columns=canonical, measurement_columns=sensor_columns(canonical), metadata_columns=EXTRA,
        null_encoding='Empty cell; OPEN/CLOSE strings retained; Timestamp and source row preserve lineage.',
        row_observed='1 for real source row; 0 for inserted time-gap row. Step warnings never erase observations.',
        row_usable='1 if no missing measurement cells; Step warnings do not invalidate sensor rows.',
        context_valid='0 on inserted gaps or invalid loop metadata. Unobserved other Steps do not invalidate current observed context.',
        segment_id='Contiguous usable interval separated only by missing time/measurement rows, not Step-only or loop-counter warnings.',
        missing_sensor_columns='Only originally missing measurement cells. Observed sensor values are never blanked for Step warnings.'))
    summary = dict(source_files=len(infos), input_widths=dict(Counter(r['columns'] for r in infos)),
        excluded=len(excluded), exclusion_reasons=dict(Counter(r['reason'] for r in excluded)), kept_runs=len(selected),
        partitions={s:dict(runs=sum(r['split']==s for r in selected),
            recipes=dict(Counter(r['recipe'] for r in selected if r['split']==s)),
            **{k:sum(r.get(k,0) for r in selected if r['split']==s) for k in
               ['output_rows','usable_rows','inserted_gap_rows','masked_subcycle_rows','step_warning_rows_preserved','subcycles','subcycles_with_warnings','masked_subcycles']}) for s in PARTITIONS},
        time_gap_events=len(all_gaps),loop_issue_groups=len(loops),
        step_only_issues=sum(not r['has_time_gap'] for r in issues),
        step_and_time_issues=sum(bool(r['has_time_gap']) for r in issues),
        time_gaps_with_step_nonobservation=sum(g['step_missing_scope']!='none_detected' for g in all_gaps))
    dump(dest/'summary.json',summary)
    report_markdown(dest,selected,all_cycles,all_gaps,issues,summary,gap)
    (dest/'README.md').write_text(
        '# MT NIT deposition CSV dataset\n\n'
        'One source Run per CSV in train/valid/test. Source files are preserved.\n\n'
        'Read policy.json and schema.json before training. Blank cells are missing, never observed zeros. '
        'Retain timestamps; never concatenate across Runs or compact masked intervals. '
        'Real timestamps retain jitter; synthetic blank rows expose missing seconds. '
        'Use _row_usable and _segment_id for conservative windows, or explicit per-cell masks for a verified missing-aware loader. '
        'Never use missing/held-out values as labels. Timestral-28M-ProcessContext supports this CSV format.\n\n'
        'runs.csv: retained files and split; run_id_mapping.csv (when renumbered): previous-to-current IDs and paths; cycles.csv: preserved cycles and warnings; '
        'time_gaps.csv: inserted intervals; exclusions.json: rejected files; loop_issues.json: unobserved counters; '
        'substep_references.json: Train-only empirical references; summary.json: counts; '
        'processing_report.md: complete Korean interval report; step_issues.csv: Step warnings with/without time gaps.\n\n'
        'See processing_report.md for the partition counts, preserved Step warnings and inserted time gaps.\n',encoding='utf-8')
    # Preserve source lineage for excluded schemas too; verify before publishing latest.
    manifest = dict(identity=identity, source_root=source.relative_to(ROOT).as_posix() if source.is_relative_to(ROOT) else str(source),
        python=platform.python_version(), selected=selected, files={p.relative_to(dest).as_posix():sha(p) for p in sorted(dest.rglob('*')) if p.is_file()})
    dump(dest/'manifest.json',manifest)
    verify(dest, source, reproduce=False)
    dump(output/'latest.json',dict(dataset=fingerprint, path=dest.relative_to(ROOT).as_posix() if dest.is_relative_to(ROOT) else str(dest)))
    print(json.dumps(dict(dataset=str(dest),summary=summary),ensure_ascii=False,indent=2))
    return dest


def verify(dest, source=None, reproduce=False):
    dest = Path(dest).resolve(); m = json.loads((dest/'manifest.json').read_text(encoding='utf-8'))
    source = Path(source).resolve() if source else ROOT/m['source_root']
    for rel, digest in m['identity']['source_files']:
        if sha(source/rel) != digest: raise ValueError('Source hash mismatch: '+rel)
    for rel, digest in m['files'].items():
        if sha(dest/rel) != digest: raise ValueError('Output hash mismatch: '+rel)
    for info in m['selected']:
        # Independent source-cell check, not merely regeneration of the same transformation.
        _,original_rows,error=load(source/info['source_path'])
        if error: raise ValueError('Source cannot be read for cell verification')
        original_by_line=dict(original_rows)
        previous = None; last_source_row = 1; previous_segment = 0; after_mask = False; count = Counter()
        with (dest/info['csv_path']).open(encoding='utf-8-sig',newline='') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != info['header'] + EXTRA: raise ValueError('Output column mismatch')
            sensors = sensor_columns(info['header'])
            for r in reader:
                if None in r or any(v is None for v in r.values()): raise ValueError('Ragged output')
                t = datetime.fromisoformat(r['Timestamp'])
                if previous is not None and t <= previous: raise ValueError('Output time order')
                previous = t
                if r['_run_id'] != info['run_id']: raise ValueError('Output Run identity')
                if r['_row_observed'] not in ('0','1') or r['_row_usable'] not in ('0','1'): raise ValueError('Invalid mask')
                if r['_row_observed'] == '0':
                    if any(r[c] for c in info['header'] if c != 'Timestamp'): raise ValueError('Invented missing-row observation')
                    if r['_source_row'] or r['_source_timestamp'] or r['_row_usable']!='0': raise ValueError('Invalid synthetic provenance')
                    count['inserted_gap_rows'] += 1
                else:
                    line = int(r['_source_row'])
                    if line <= last_source_row or datetime.fromisoformat(r['_source_timestamp']) != t: raise ValueError('Source row/time mismatch')
                    original=original_by_line[line]
                    if m['identity']['policy']['version']=='mt-nit-csv-time-gaps-2':
                        for c in sensors:
                            expected='' if original[c].strip().lower() in NULLS else original[c]
                            if r[c]!=expected: raise ValueError(f'Observed sensor changed: {info["csv_path"]}:{line}:{c}')
                    last_source_row = line; count['source_rows_in_run'] += 1
                if r['_row_usable'] == '1':
                    if any(r[c].strip().lower() in NULLS for c in sensors): raise ValueError('Missing sensor labeled usable')
                    seg = int(r['_segment_id'])
                    if seg < previous_segment or (after_mask and seg <= previous_segment): raise ValueError('Segment bridges missingness')
                    previous_segment = seg; after_mask = False; count['usable_rows'] += 1
                else:
                    if r['_segment_id']: raise ValueError('Unusable row belongs to usable segment')
                    after_mask = True
                count['output_rows'] += 1
        for key in ('inserted_gap_rows','source_rows_in_run','usable_rows','output_rows'):
            if count[key] != info.get(key,0): raise ValueError('Output count mismatch: '+key)
    if reproduce:
        if sha(Path(__file__)) != m['identity']['generator_sha256']: raise ValueError('Generator differs from dataset version')
        refs = json.loads((dest/'substep_references.json').read_text(encoding='utf-8'))
        for info in m['selected']:
            rows = retained_rows(source/info['source_path'],info)
            it, *_ = make_rows(rows,info,refs,m['identity']['policy']['gap_threshold_seconds'])
            stream = DigestText(); serialize(stream,info['header'],it)
            if stream.hash.hexdigest() != m['files'][info['csv_path']]: raise ValueError('Reproduction mismatch: '+info['csv_path'])
    print('Verified source/output hashes, CSV structure, masks and segments' + (' and byte-identical CSV reproduction' if reproduce else ''),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'data/mt_nit/data_complete')
    p.add_argument('--output',type=Path,default=ROOT/'etc/data/mt_nit_deposition_csv')
    p.add_argument('--ratios',type=float,nargs=3,default=(0.7,0.15,0.15))
    p.add_argument('--gap-seconds',type=float,default=2.0)
    p.add_argument('--verify',type=Path)
    p.add_argument('--reproduce',action='store_true')
    p.add_argument('--membership-from',type=Path,help='Prior dataset directory: preserve retained Run IDs and split membership')
    p.add_argument('--renumber-runs',action='store_true',help='With --membership-from, preserve splits but number Runs consecutively; save ID mapping')
    a=p.parse_args()
    if a.verify: verify(a.verify,a.source,a.reproduce)
    else: build(a.source,a.output,a.ratios,a.gap_seconds,a.membership_from,a.renumber_runs)


if __name__ == '__main__': main()
