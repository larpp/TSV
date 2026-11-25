"""H-bounded detached planning from t0 state, past-completed pattern and predicted heads."""
import copy
import math
from ..contracts import key
from ..metadata import candidate_states
from .patterns import choose_pattern,validate_bank


def advance(s,seconds):
    s=copy.deepcopy(s);s['timestamp_us']+=int(round(seconds*1e6))
    for n,known in s['elapsed_known'].items():
        if known:s['elapsed'][n]+=seconds
    return s


def unknown(s):
    s=copy.deepcopy(s);s.update(quality='UNKNOWN',state_conflict=True,sub_seq=None,sub_id=None)
    return s


class Renderer:
    def __init__(self,artifact,bank,horizon=30):
        self.artifact=artifact;self.bank=validate_bank(bank);self.horizon=horizon
        if not 1 <= horizon <= 32: raise ValueError('TSV renderer requires 1 to 32 forecast steps')

    def render(self,current,recent,predict):
        """predict(state,candidates) returns detached remaining/probabilities/sigma. No future rows."""
        state=copy.deepcopy(current);states=[];valid=[];events=[];reasons=[];head_calls=0
        visited={key(*state['macro_key'])};deadline=None;choice=None;pattern=None;plan=[]
        elapsed=0.;start_offset=0.;unknown_macro=False
        def schedule(s,offset):
            nonlocal head_calls
            candidates=candidate_states(s,self.artifact)
            if not candidates:return None,None,None,[],'no_candidates'
            forecast=predict(copy.deepcopy(s),candidates);head_calls+=1
            remaining=forecast.get('remaining',float('nan'))
            if not math.isfinite(remaining) or remaining<1:
                return None,None,None,[],'invalid_head'
            duration=max(1.,remaining);selected=None
            edges=self.bank['edges'].get(key(*s['macro_key']),{})
            probabilities=forecast.get('probabilities',[])
            legal=[i for i,c in enumerate(candidates) if key(*c['macro_key']) in edges and key(*c['macro_key']) not in visited]
            if legal and len(probabilities)==len(candidates) and all(math.isfinite(v) and v>=0 for v in probabilities):
                selected=candidates[max(legal,key=lambda i:probabilities[i])]
            p=choose_pattern(s,recent if offset==0 else None,self.bank) if s['phase']=='SUB' else None
            segments=[];fallback=None
            if s['phase']=='SUB' and s['loop_valid'] and p:
                steps=p['steps'];idx=next((i for i,x in enumerate(steps) if (x['seq'],x['id'])==(s['sub_seq'],s['sub_id'])),None)
                used=s['elapsed']['substep'] if s['elapsed_known']['substep'] else 0.
                if idx is not None and used<steps[idx]['duration']:
                    segments=[dict(x,duration=x['duration']-used if j==idx else x['duration'],elapsed_start=used if j==idx else 0.) for j,x in enumerate(steps) if j>=idx]
                    if p['source']=='observed_completed_loop' and s['elapsed_known']['loop'] and s['elapsed']['loop']<p['duration']:
                        duration=sum(x['duration'] for x in segments)
                    else:fallback='head_timed_pattern'
                else:
                    # Do not restart at the first substep when current position is incompatible.
                    fallback='pattern_position_mismatch'
            if segments:
                total=sum(x['duration'] for x in segments)
                segments=[dict(x,duration=x['duration']*duration/total) for x in segments]
            events.append(dict(kind='head_plan',offset=offset,remaining=duration,sigma=forecast.get('sigma'),
                               probabilities=probabilities,pattern_source=p['source'] if p else None))
            return offset+max(1.,duration),selected,p,segments,fallback

        if state['terminal']:
            return dict(states=[None]*self.horizon,process_forecast_valid=[False]*self.horizon,
                        events=[],fallbacks=['observed_END'],head_calls=0,source='predicted_path_not_observations')
        if state['quality']!='OBSERVED' or state['phase']=='UNKNOWN':
            return dict(states=[advance(unknown(state),t) for t in range(1,self.horizon+1)],
                        process_forecast_valid=[False]*self.horizon,events=[],fallbacks=['unknown_current'],head_calls=0,source='predicted_path_not_observations')
        if state['phase']=='SUB' and not state['loop_valid']:
            return dict(states=[advance(unknown(state),t) for t in range(1,self.horizon+1)],
                        process_forecast_valid=[False]*self.horizon,events=[],fallbacks=['unknown_loop_setting'],head_calls=0,source='predicted_path_not_observations')
        deadline,choice,pattern,plan,reason=schedule(state,0)
        if reason:reasons.append(reason)
        if deadline is None:state=unknown(state);unknown_macro=True
        for t in range(1,self.horizon+1):
            # At most one boundary per second, including very short predicted states.
            state=advance(state,1)
            if not unknown_macro and deadline is not None and t>=deadline:
                if state['phase']=='SUB' and state['loop_valid'] and state['k']<state['K']:
                    state['k']+=1;state['elapsed'].update(loop=0.,substep=0.)
                    state['elapsed_known'].update(loop=True,substep=True)
                    events.append(dict(kind='loop',offset=t,k=state['k'],K=state['K']))
                    if pattern:
                        plan=copy.deepcopy(pattern['steps']);state['sub_seq']=plan[0]['seq'];state['sub_id']=plan[0]['id']
                        deadline=t+max(1.,pattern['duration']);start_offset=t
                    else:
                        state=unknown(state);deadline=None;unknown_macro=True;reasons.append('no_loop_pattern')
                elif choice is not None:
                    previous=state;state=copy.deepcopy(choice);state['timestamp_us']=previous['timestamp_us']
                    state['counters']=dict(previous['counters'])
                    if (state['main_seq'],state['main_id'])!=(previous['main_seq'],previous['main_id']):
                        state['counters'][str(state['main_id'])]=state['counters'].get(str(state['main_id']),0)+1
                        state['elapsed']['main']=0.;state['elapsed_known']['main']=True
                    else:state['elapsed']['main']=previous['elapsed']['main']
                    state['elapsed'].update(macro=0.,loop=0.,substep=0.)
                    state['elapsed_known'].update(macro=True,loop=False,substep=False)
                    state.update(K=None,k=None,K_known=False,loop_valid=False,previous_loop_known=False,previous_loop_duration=0.)
                    visited.add(key(*state['macro_key']));events.append(dict(kind='macro',offset=t,key=state['macro_key']))
                    start_offset=t;plan=[]
                    if state['terminal']:deadline=None
                    elif state['phase']=='SUB':
                        state=unknown(state);unknown_macro=True;deadline=None;reasons.append('new_sub_unknown_K')
                    else:
                        deadline,choice,pattern,plan,reason=schedule(state,t)
                        if reason:reasons.append(reason)
                        if deadline is None:state=unknown(state);unknown_macro=True
                else:
                    state=unknown(state);deadline=None;unknown_macro=True;reasons.append('no_legal_next_macro')
            if state['terminal']:
                states.append(None);valid.append(False);continue
            if state['phase']=='SUB' and not unknown_macro:
                if plan:
                    position=max(0.,t-start_offset);acc=0.;chosen=plan[-1];subelapsed=0.
                    for segment in plan:
                        if position<acc+segment['duration']:
                            chosen=segment;subelapsed=segment.get('elapsed_start',0.)+position-acc;break
                        acc+=segment['duration']
                    state['sub_seq']=chosen['seq'];state['sub_id']=chosen['id'];state['elapsed']['substep']=subelapsed
                    state['elapsed_known']['substep']=True
                else:
                    state['sub_seq']=state['sub_id']=None;state['elapsed_known']['substep']=False
                    state['quality']='UNKNOWN';reasons.append('unknown_substep')
            states.append(copy.deepcopy(state));valid.append(not unknown_macro and state['quality']=='OBSERVED')
        return dict(states=states,process_forecast_valid=valid,events=events,fallbacks=sorted(set(reasons)),head_calls=head_calls,
                    source='predicted_path_not_observations')
