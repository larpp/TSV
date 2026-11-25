"""Train-only simple baselines and denominator-aware process-head validation."""
import math
from collections import Counter,defaultdict
import numpy as np
import torch


def candidate_key(categories):
    # Recipe/Main position/phase/Subrecipe identify macro; occurrence varies by current history.
    return ','.join(str(int(categories[i])) for i in (0,1,3,4,7))


def fit_baselines(data):
    if data.partition!='train':raise ValueError('Head baselines must use Train only')
    counts=defaultdict(Counter);durations=defaultdict(list);global_counts=Counter();all_durations=[]
    for _,info,label,candidates in data.label_rows():
        state=str(info['macro'])
        if label['next_valid']:
            target=candidate_key(candidates[label['next_label']]);counts[state][target]+=1;global_counts[target]+=1
        if label['remaining_valid']:
            durations[state].append(label['remaining_seconds']);all_durations.append(label['remaining_seconds'])
    if not global_counts or not all_durations:raise ValueError('No observed Train process labels')
    return dict(fit_partition='train',next_counts={s:dict(v) for s,v in counts.items()},global_next=dict(global_counts),
                remaining_medians={s:float(np.median(v)) for s,v in durations.items()},
                global_remaining_median=float(np.median(all_durations)),
                weighting='Observed Train windows; conditional macro frequency and remaining-time median')


class HeadMetrics:
    def __init__(self,data,baselines):
        self.data=data;self.baselines=baselines;self.groups=defaultdict(list)
    def update(self,output,batch,indices):
        o={k:v.detach().cpu() for k,v in output.items() if isinstance(v,torch.Tensor)}
        l={k:v.detach().cpu() for k,v in batch['labels'].items()}
        for j,index in enumerate(indices):
            info=self.data.info(index);state=str(info['macro'])
            record=dict(run=info['run_id'],next_valid=False,remaining_valid=False)
            if bool(l['next_valid'][j]) and bool(o['available'][j]):
                label=int(l['next_label'][j]);pred=int(o['logits'][j].argmax())
                ce=float(torch.nn.functional.cross_entropy(o['logits'][j:j+1],torch.tensor([label])))
                candidates=batch['candidate_categories'][j][batch['candidate_mask'][j]].cpu().numpy()
                counts=self.baselines['next_counts'].get(state,self.baselines['global_next'])
                baseline=max(range(len(candidates)),key=lambda k:counts.get(candidate_key(candidates[k]),0))
                record.update(next_valid=True,ce=ce,correct=float(pred==label),baseline_correct=float(baseline==label))
            if bool(l['remaining_valid'][j]) and bool(o['remaining_available'][j]):
                truth=float(l['remaining_seconds'][j]);mu=float(o['mu'][j]);sigma=float(o['sigma'][j])
                log=math.log(max(truth,1.));nll=log+math.log(sigma)+.5*((log-mu)/sigma)**2+.5*math.log(2*math.pi)
                predicted=float(o['remaining_seconds'][j]);baseline=self.baselines['remaining_medians'].get(state,self.baselines['global_remaining_median'])
                lower=max(1.,math.exp(min(math.log(1e12),mu-1.2815515655*sigma)))
                upper=max(1.,math.exp(min(math.log(1e12),mu+1.2815515655*sigma)))
                record.update(remaining_valid=True,nll=nll,absolute_error=abs(predicted-truth),
                    baseline_absolute_error=abs(baseline-truth),coverage80=float(lower<=truth<=upper))
            for group in ('all','recipe:'+info['recipe'],'phase:'+info['phase'],
                          'last_loop' if info['last_loop'] else 'not_last_loop'):
                self.groups[group].append(record)
    def report(self):
        result={}
        for group,rows in self.groups.items():
            next_rows=[r for r in rows if r['next_valid']];rem=[r for r in rows if r['remaining_valid']]
            def average(records,key,macro=False):
                if not records:return None
                if not macro:return float(np.mean([r[key] for r in records]))
                byrun=defaultdict(list)
                for r in records:byrun[r['run']].append(r[key])
                return float(np.mean([np.mean(v) for v in byrun.values()]))
            r=dict(windows=len(rows),next_labels=len(next_rows),remaining_labels=len(rem),
                next_coverage=len(next_rows)/len(rows),remaining_coverage=len(rem)/len(rows),
                next_accuracy=average(next_rows,'correct'),next_run_macro_accuracy=average(next_rows,'correct',True),
                baseline_next_accuracy=average(next_rows,'baseline_correct'),
                baseline_next_run_macro_accuracy=average(next_rows,'baseline_correct',True),
                next_ce=average(next_rows,'ce'),next_run_macro_ce=average(next_rows,'ce',True),
                remaining_mae_seconds=average(rem,'absolute_error'),remaining_run_macro_mae_seconds=average(rem,'absolute_error',True),
                baseline_remaining_mae_seconds=average(rem,'baseline_absolute_error'),
                baseline_remaining_run_macro_mae_seconds=average(rem,'baseline_absolute_error',True),
                remaining_median_absolute_error_seconds=float(np.median([v['absolute_error'] for v in rem])) if rem else None,
                remaining_p90_absolute_error_seconds=float(np.quantile([v['absolute_error'] for v in rem],.9)) if rem else None,
                remaining_nll=average(rem,'nll'),remaining_run_macro_nll=average(rem,'nll',True),
                remaining_interval80_coverage=average(rem,'coverage80'))
            r['auxiliary_run_macro_loss']=r['next_run_macro_ce']+r['remaining_run_macro_nll'] if next_rows and rem else None
            result[group]=r
        return result
