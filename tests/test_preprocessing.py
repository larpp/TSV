"""CSV quality boundaries and missing-time preservation; no model/runtime dependencies."""
import csv
from datetime import datetime, timedelta
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest

from scripts.dataset import build_mt_nit_csv_dataset as ds


HEADER = sorted(ds.META | {'ProcessRecipeStepSeqNo', 'ProcessRecipeStepName', 'Temp', 'Valve'})
HEADER += ['sensor_'+str(i) for i in range(166-len(HEADER))]


def row(step=14, sub=None, counter=1, t=0):
    r = dict.fromkeys(HEADER,'1')
    r.update(Timestamp=(datetime(2025,1,1)+timedelta(seconds=t)).isoformat(sep=' '),
        PMStoredProcessRecipeName='MainRecipe/RUN/0_P_IH_HD02',
        ProcessRecipeStepSeqNo=str(step), ProcessRecipeStepName={1:'START',42:'END'}.get(step,'STEP'),
        PPExecname='SubRecipe/example' if sub is not None else 'MainRecipe/RUN/0_P_IH_HD02',
        PPExecStepSeqNo=str(sub if sub is not None else step), PPExecStepName='SUB',
        SubRecipeLoopSettingValue='3',SubRecipeLoopMoniterValue=str(counter),Temp='005.0',Valve='OPEN')
    return r


def info(): return dict(run_id='run_0001',recipe='0_P_IH_HD02',split='train',header=HEADER)


def refs(): return {'0_P_IH_HD02|14|SubRecipe/example':{'expected':[0,1,2]}}


def write(path,rows,header=HEADER):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=header);w.writeheader();w.writerows(rows)


class CsvDatasetTests(unittest.TestCase):
    def test_main_is_process_column_not_executing_substep(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'run.csv';rows=[row(s,sub=0 if s==14 else None,t=s) for s in range(1,43)]
            write(p,rows)
            self.assertEqual(ds.inspect(p,Path(d))['status'],'selected')
            write(p,[r for r in rows if r['ProcessRecipeStepSeqNo']!='17'])
            self.assertEqual(ds.inspect(p,Path(d))['missing_main_steps'],[17])

    def test_trim_end43_and_reject_internal_order(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'run.csv';rows=[row(s,t=s) for s in range(1,44)];write(p,rows)
            self.assertEqual(ds.inspect(p,Path(d))['trimmed_rows'],1)
            rows[9]['ProcessRecipeStepSeqNo']='11';rows[10]['ProcessRecipeStepSeqNo']='10';write(p,rows)
            self.assertEqual(ds.inspect(p,Path(d))['reason'],'main_step_order_or_multiple_runs')

    def test_schema_recipe_timestamp_rejections(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'run.csv';rows=[row(s,t=s) for s in range(1,43)]
            for r in rows:r['PMStoredProcessRecipeName']='MainRecipe/RUN/F2_CLN_COAT_S'
            write(p,rows);self.assertEqual(ds.inspect(p,Path(d))['reason'],'recipe_not_selected_or_changes')
            rows=[row(s,t=s) for s in range(1,43)];rows[1]['Timestamp']=rows[0]['Timestamp'];write(p,rows)
            self.assertEqual(ds.inspect(p,Path(d))['reason'],'nonincreasing_timestamp')
            p.write_text('a,b\n1,2\n',encoding='utf-8')
            self.assertEqual(ds.inspect(p,Path(d))['reason'],'column_count')
            write(p,[row()]);p.write_text(p.read_text(encoding='utf-8-sig')+'bad,row\n',encoding='utf-8')
            self.assertEqual(ds.inspect(p,Path(d))['reason'],'ragged_csv_row')

    def test_twenty_second_gap_keeps_elapsed_time_and_blank_mask(self):
        rows=[(2,row(step=1,t=0.123)),(3,row(step=2,t=20.123))]
        it,stats,_,gaps,_=ds.make_rows(rows,info(),{},2);out=list(it)
        self.assertEqual(len(out),21);self.assertEqual(stats['inserted_gap_rows'],19)
        self.assertEqual(out[-1]['_elapsed_seconds'],'20.000000')
        self.assertEqual(out[0]['_segment_id'],1);self.assertEqual(out[-1]['_segment_id'],2)
        self.assertTrue(all(r['Temp']=='' and r['_row_observed']==0 and r['_row_usable']==0 for r in out[1:-1]))
        self.assertTrue(all(r['ProcessRecipeStepSeqNo']=='' for r in out[1:-1]))
        self.assertEqual(gaps[0]['seconds'],20)

    def test_timing_jitter_preserved_not_flagged_and_two_seconds_included(self):
        rows=[(i+2,row(step=1,t=t)) for i,t in enumerate([0,1.001,3.001])]
        it,stats,*_=ds.make_rows(rows,info(),{},2);out=list(it)
        self.assertEqual(stats['inserted_gap_rows'],1)
        self.assertEqual(out[1]['_source_timestamp'],rows[1][1]['Timestamp'])
        self.assertEqual(out[1]['_segment_id'],out[0]['_segment_id'])

    def test_step_only_nonobservation_preserves_every_sensor_and_segment(self):
        rows=[(i+2,row(sub=s,counter=c,t=i)) for i,(c,s) in enumerate([(1,0),(1,1),(1,2),(2,0),(2,2),(3,0),(3,1),(3,2)])]
        it,stats,cycles,*_=ds.make_rows(rows,info(),refs(),2);out=list(it)
        self.assertEqual(len(out),len(rows));self.assertEqual(stats['inserted_gap_rows'],0)
        self.assertEqual([c['status'] for c in cycles],['retained']*3)
        self.assertEqual(cycles[1]['missing_substeps'],'[1]')
        self.assertEqual(cycles[1]['has_time_gap'],0)
        for (_,original),result in zip(rows,out):
            for column in ds.sensor_columns(HEADER):self.assertEqual(result[column],original[column])
            self.assertEqual(result['_segment_id'],1)
            self.assertEqual(result['_row_usable'],1)
            self.assertEqual(result['_context_valid'],1)

    def test_missing_substep_with_time_gap_inserts_only_absent_seconds(self):
        rows=[(2,row(sub=0,t=0)),(3,row(sub=2,t=3))]
        it,stats,cycles,gaps,_=ds.make_rows(rows,info(),refs(),2);out=list(it)
        self.assertEqual(stats['inserted_gap_rows'],2)
        self.assertEqual(cycles[0]['has_time_gap'],1)
        self.assertEqual(gaps[0]['step_missing_scope'],'subcycle')
        self.assertEqual([r['Temp'] for r in out],['005.0','','','005.0'])
        self.assertEqual([r['_row_observed'] for r in out],[1,0,0,1])

    def test_missing_main_steps_are_audit_only_with_and_without_time_gaps(self):
        for delta,expected_gaps in [(1,0),(4,3)]:
            rows=[(2,row(step=1,t=0)),(3,row(step=14,t=1)),(4,row(step=42,t=1+delta))]
            with tempfile.TemporaryDirectory() as d:
                p=Path(d)/'run.csv';write(p,[r for _,r in rows])
                self.assertEqual(ds.inspect(p,Path(d))['status'],'selected')
            it,stats,*_=ds.make_rows(rows,info(),{},2);out=list(it)
            self.assertEqual(stats['inserted_gap_rows'],expected_gaps)
            self.assertEqual(out[0]['Temp'],'005.0');self.assertEqual(out[-1]['Temp'],'005.0')
            issues=ds.main_step_issues(rows,info(),2)
            self.assertEqual(any(i['has_time_gap'] for i in issues),delta>=2)
            self.assertEqual(issues[0]['action'],'preserve_observed_values')

    def test_start_end_only_excluded_regardless_of_gap_and_repetition(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'run.csv'
            for delta in (1,20):
                write(p,[row(1,t=0),row(1,t=1),row(42,t=1+delta),row(42,t=2+delta)])
                result=ds.inspect(p,Path(d))
                self.assertEqual(result['status'],'excluded')
                self.assertEqual(result['reason'],'start_end_only')
            write(p,[row(1,t=0),row(2,t=1),row(42,t=2)])
            self.assertEqual(ds.inspect(p,Path(d))['status'],'selected')

    def test_preserved_membership_rejects_changes_and_keeps_gapped_ids(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'manifest.json'
            rows=[dict(source_path=str(i),source_sha256=str(i),run_id=f'run_{i:04d}',split=s)
                  for i,s in [(1,'train'),(2,'train'),(3,'valid'),(4,'test')]]
            for r in rows:r['csv_path']=r['split']+'/'+r['run_id']+'.csv'
            p.write_text(json.dumps(dict(selected=rows)))
            selected=[dict(source_path=str(i),source_sha256=str(i)) for i in (1,3,4)]
            ds.preserve_membership(selected,p)
            self.assertEqual([r['run_id'] for r in selected],['run_0001','run_0003','run_0004'])
            self.assertEqual([r['split'] for r in selected],['train','valid','test'])
            selected[0]['source_sha256']='changed'
            with self.assertRaises(ValueError):ds.preserve_membership(selected,p)

    def test_renumber_preserves_splits_and_records_old_ids(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'manifest.json'
            rows=[dict(source_path=str(i),source_sha256=str(i),run_id=f'run_{i:04d}',split=s,csv_path=s+'/'+str(i)+'.csv')
                  for i,s in [(1,'train'),(3,'train'),(4,'valid'),(5,'test')]]
            p.write_text(json.dumps(dict(selected=rows)))
            selected=[dict(source_path=r['source_path'],source_sha256=r['source_sha256']) for r in rows]
            ds.preserve_membership(selected,p,renumber=True)
            self.assertEqual([r['run_id'] for r in selected],[f'run_{i:04d}' for i in range(1,5)])
            self.assertEqual([r['previous_run_id'] for r in selected],[r['run_id'] for r in rows])
            self.assertEqual([r['split'] for r in selected],[r['split'] for r in rows])
            self.assertEqual([r['previous_csv_path'] for r in selected],[r['csv_path'] for r in rows])

    def test_unknown_reference_does_not_learn_from_test(self):
        rows=[(2,row(sub=0)),(3,row(sub=1,t=1))]
        it,_,cycles,*_=ds.make_rows(rows,info(),{},2);out=list(it)
        self.assertTrue(all(r['_row_usable']==1 and r['Valve']=='OPEN' for r in out))
        self.assertIn('reference_unavailable',cycles[0]['reason'])

    def test_skipped_and_reset_loop_counts_warn_without_inventing_time_gaps(self):
        for counters in [(1,3),(2,1)]:
            rows=[(i+2,row(sub=s,counter=c,t=i)) for i,(c,s) in
                  enumerate((c,s) for c in counters for s in [0,1,2])]
            it,stats,*_=ds.make_rows(rows,info(),refs(),2);out=list(it)
            self.assertTrue(all(r['_row_usable']==1 for r in out))
            self.assertEqual([r['_segment_id'] for r in out],[1]*6)
            self.assertIn('loop_counter_boundary',out[3]['_quality'])
            self.assertEqual(stats['inserted_gap_rows'],0)

    def test_nan_cell_is_blank_not_zero(self):
        r=row(step=1);r['Temp']='NaN'
        it,*_=ds.make_rows([(2,r)],info(),{},2);out=list(it)[0]
        self.assertEqual(out['Temp'],'');self.assertEqual(out['Valve'],'OPEN')
        self.assertEqual(out['_row_usable'],0);self.assertEqual(out['_missing_sensor_columns'],'Temp')

    def test_split_is_chronological_whole_runs_and_rejects_overlap(self):
        runs=[dict(start=f'2025-01-{i:02} 01:00:00',end=f'2025-01-{i:02} 02:00:00',source_path=str(i)) for i in range(1,21)]
        out=ds.chronological_split(runs[::-1],(.7,.15,.15))
        self.assertEqual([r['split'] for r in out],['train']*14+['valid']*3+['test']*3)
        runs[1]['start']=runs[0]['end']
        with self.assertRaises(ValueError):ds.chronological_split(runs,(.7,.15,.15))

    def test_training_reference_ignores_validation_sequence(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);infos=[]
            for split, seq in [('train',[0,1,2]),('valid',[0,1,2,3])]:
                path=root/(split+'.csv');write(path,[row(sub=s,t=s) for s in seq])
                x=info();x.update(source_path=path.name,source_sha256=ds.sha(path),first_source_row=2,last_source_row=len(seq)+1,split=split)
                infos.append(x)
            self.assertEqual(next(iter(ds.fit_references(root,infos).values()))['expected'],[0,1,2])

    def test_digest_serialization_matches_file(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'out.csv';rows=[(2,row(step=1)),(3,row(step=2,t=4))]
            it,*_=ds.make_rows(rows,info(),{},2)
            with p.open('w',encoding='utf-8-sig',newline='') as f:ds.serialize(f,HEADER,it)
            it,*_=ds.make_rows(rows,info(),{},2);digest=ds.DigestText();ds.serialize(digest,HEADER,it)
            self.assertEqual(digest.hash.hexdigest(),ds.sha(p))

    def test_end_to_end_same_basename_no_overwrite_and_reproduce(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'source';output=root/'output'
            for day in range(1,8):
                folder=source/str(day);folder.mkdir(parents=True)
                rows=[row(s,t=s+day*86400) for s in range(1,43)]
                write(folder/'run.csv',rows)
            before={str(p):ds.sha(p) for p in source.rglob('*.csv')}
            with redirect_stdout(io.StringIO()):
                dest=ds.build(source,output,gap=3)
                ds.verify(dest,source,reproduce=True)
            self.assertIn('3초 이상',(dest/'processing_report.md').read_text(encoding='utf-8'))
            files=[p for split in ds.PARTITIONS for p in (dest/split).glob('*.csv')]
            self.assertEqual(len(files),7)
            self.assertEqual(before,{str(p):ds.sha(p) for p in source.rglob('*.csv')})


if __name__=='__main__':unittest.main()
