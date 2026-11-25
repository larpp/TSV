"""Observed-target-only metrics with missingness coverage; no zero imputation."""
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader
from experiment_common.evaluation import Metrics
from experiment_common.mixed_evaluation import MixedMetrics, _classification
from experiment_common.sensor_states import state_sensors, apply_state_threshold
from experiment_common.experiment_reporting import Progress


class MaskedMetrics(Metrics):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.candidates=0;self.observed=0

    def update(self,prediction,target,indices,data):
        valid=np.isfinite(target)&np.isfinite(prediction)
        self.candidates+=target.size;self.observed+=int(valid.sum())
        scaled=np.where(valid,prediction-target,0.0);error=scaled*data.std
        steps=[];runs=[];actual=[]
        for i,index in enumerate(indices):
            s,j=data.location(int(index));runs.append(s['run_id'])
            steps.append(s['main_steps'][s['targets'][j]])
            actual.append(s['values'][s['targets'][j]])
            count=int(valid[i].sum())
            if count:self.scaled_runs[s['run_id']]+=[np.abs(scaled[i]).sum()/count,1]
        steps=np.asarray(steps);runs=np.asarray(runs);actual=np.asarray(actual)
        eligible=valid&(np.abs(actual)>self.mape_epsilon)
        ape=np.zeros_like(error);np.divide(np.abs(error)*100,np.abs(actual),out=ape,where=eligible)
        def sums(e,a,ok,obs,axis=0):
            return np.stack([np.abs(e).sum(axis=axis),(e*e).sum(axis=axis),e.sum(axis=axis),
                             obs.sum(axis=axis),a.sum(axis=axis),ok.sum(axis=axis),
                             np.where(obs&~ok,np.abs(e),0).sum(axis=axis)],axis=-1)
        for rid in np.unique(runs):
            for step in np.unique(steps[runs==rid]):
                selected=(runs[:,None]==rid)&(steps==step)
                total=sums(error[selected],ape[selected],eligible[selected],valid[selected])
                for c,name in enumerate(self.sensors):
                    if total[c,3]:self.groups[(int(rid),int(step),name)]+=total[c]
        total=sums(error,ape,eligible,valid)
        for c,name in enumerate(self.sensors):
            for h in range(error.shape[1]):
                if total[h,c,3]:self.horizons[(name,h+1)]+=total[h,c]

    @staticmethod
    def scores(v):
        if v[3]:return Metrics.scores(v)
        return dict(mae=None,rmse=None,bias=None,points=0,mape_percent=None,mape_points=0,
                    mape_excluded_near_zero_points=0,mape_coverage_percent=None,near_zero_mae=None)

    def report(self):
        # Keep every selected sensor in the output, including one with no labels.
        per_sensor={}
        for name in self.sensors:
            byrun=defaultdict(lambda:np.zeros(7))
            for (rid,_,sensor),v in self.groups.items():
                if sensor==name:byrun[rid]+=v
            total=sum(byrun.values(),np.zeros(7));scores=[self.scores(v) for v in byrun.values()]
            def mean(key):
                values=[s[key] for s in scores if s[key] is not None]
                return float(np.mean(values)) if values else None
            per_sensor[name]=dict(window_weighted=self.scores(total),run_macro_mae=mean('mae'),
                run_macro_rmse=mean('rmse'),run_macro_mape_percent=mean('mape_percent'))
        macro={}
        for key in ('mae','rmse','bias','mape_percent'):
            vals=[s['window_weighted'][key] for s in per_sensor.values() if s['window_weighted'][key] is not None]
            macro[key]=float(np.mean(vals)) if vals else None
        totals=self.scores(sum(self.groups.values(),np.zeros(7)))
        aggregate=dict(sensor_macro=macro,sensor_count=len(self.sensors),
            mape_eligible_sensors=sum(s['window_weighted']['mape_points']>0 for s in per_sensor.values()),
            mape_coverage_percent=totals['mape_coverage_percent'],all_target_points=totals,
            observed_target_points=self.observed,missing_target_points=self.candidates-self.observed,
            target_coverage_percent=100*self.observed/self.candidates if self.candidates else None,
            mape_policy=dict(epsilon_raw_units=self.mape_epsilon,rule='observed targets only AND abs(actual)>epsilon; no denominator clipping',
                weighting='sensor macro over overlapping Window targets',official_policy_approved=False))
        return dict(run_macro_scaled_mae=float(np.mean([v[0]/v[1] for v in self.scaled_runs.values()])) if self.scaled_runs else None,
                    sensors=per_sensor,aggregate=aggregate,runs_evaluated=len(self.scaled_runs))


class MaskedMixed(MixedMetrics):
    def __init__(self,data,epsilon,threshold):
        super().__init__(data,epsilon,threshold)
        self.all=MaskedMetrics(data.config['sensors'],epsilon)
        self.numeric=MaskedMetrics([data.config['sensors'][i] for i in self.continuous],epsilon) if self.continuous else None

    def update(self,prediction,target,indices,data):
        self.all.update(prediction,target,indices,data)
        if self.numeric:self.numeric.update(prediction[...,self.continuous],target[...,self.continuous],indices,self.view)
        raw=prediction*data.std+data.mean
        labels=apply_state_threshold(raw,data.config['sensors'],self.states,self.threshold)
        for c,name in zip(self.indices,self.states):
            y=target[...,c]*data.std[c]+data.mean[c]
            valid=np.isfinite(y);truth=np.rint(y[valid]).astype(int)
            if not np.allclose(y[valid],truth,atol=1e-6) or not np.isin(truth,[0,1]).all():raise ValueError('Invalid binary labels')
            predicted=labels[...,c][valid].astype(int)
            self.counts[name]+=np.bincount(2*truth+predicted,minlength=4).reshape(2,2)


def evaluate(model,data,device,batch_size,max_batches=0,log=None,log_every=100,
             phase='valid',epoch=None,mape_epsilon=1e-6,state_threshold=0.5,observer=None):
    if observer is not None:raise ValueError('Legacy Parquet residual observer cannot read masked CSV; use CSV by_run_step_sensor.csv metrics')
    mixed=bool(state_sensors(data.config))
    def tracker():return MaskedMixed(data,mape_epsilon,state_threshold) if mixed else MaskedMetrics(data.config['sensors'],mape_epsilon)
    metric=tracker();baseline=tracker();model.eval();windows=0
    regimes={key:dict(windows=0,observed_targets=0,absolute_scaled_error=0.) for key in ('complete_input','missing_input')}
    loader=DataLoader(data,batch_size=batch_size,shuffle=False,num_workers=0)
    total=min(len(loader),max_batches) if max_batches else len(loader)
    progress=Progress(log,phase,total,log_every,epoch) if log else None
    with torch.no_grad():
        for batch,(x,y,indices) in enumerate(loader):
            if max_batches and batch>=max_batches:break
            pred=model(x.to(device)).cpu().numpy();truth=y.numpy()
            if not np.isfinite(pred).all():raise ValueError('Nonfinite prediction')
            metric.update(pred,truth,indices,data)
            values=x[...,:len(data.config['sensors'])].numpy()
            observed=np.isfinite(values)
            last=np.where(observed,np.arange(values.shape[1])[None,:,None],-1).max(axis=1)
            if np.any(last<0):raise ValueError('Persistence requires an observed context per sensor')
            past=np.take_along_axis(values,last[:,None,:],axis=1)
            baseline.update(np.repeat(past,data.config['horizon'],axis=1),truth,indices,data)
            for i in range(len(indices)):
                key='complete_input' if observed[i].all() else 'missing_input';r=regimes[key]
                valid=np.isfinite(truth[i]);r['windows']+=1;r['observed_targets']+=int(valid.sum())
                r['absolute_scaled_error']+=float(np.abs(pred[i][valid]-truth[i][valid]).sum())
            windows+=len(indices)
            if progress:progress.update(batch+1,windows=windows)
    if not windows:raise ValueError('No evaluated CSV windows')
    for r in regimes.values():r['scaled_mae']=r.pop('absolute_scaled_error')/r['observed_targets'] if r['observed_targets'] else None
    coverage=(metric.all if mixed else metric).report()['aggregate']
    if log:
        log.emit('target_coverage',f'{phase}: observed targets={coverage["observed_target_points"]}; '
                 f'missing targets excluded={coverage["missing_target_points"]}; '
                 f'target coverage={coverage["target_coverage_percent"]:.2f}%',
                 observed_targets=coverage['observed_target_points'],missing_targets=coverage['missing_target_points'])
    return dict(windows_evaluated=windows,windows_available=len(data),partial=windows<len(data),
                model=metric.report(),persistence=baseline.report(),input_missingness=regimes,
                target_coverage={k:coverage[k] for k in ('observed_target_points','missing_target_points','target_coverage_percent')}),metric
