"""Remove both quality routing and its probability provider using frozen states."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import pickle
import numpy as np
from .calibration import late_fusion
from .force_nested_oof import temperature_probability


def role_variants(probabilities,ids,qmean,qmin):
    w=np.ones(len(ids))/len(ids)
    q={n:np.ones(len(qmean)) if n in ('F6_IMU','F9_Quality') else qmin if n in ('F0','F2b_CSP') else qmean for n in ids}
    without={n:p for n,p in probabilities.items() if n!='F9_Quality'}
    return {'full':late_fusion(probabilities,ids,w,q),
        'without_F9_routing':late_fusion(probabilities,ids,w),
        'without_F9_provider':late_fusion(without,ids,w,q),
        'without_F9_routing_and_provider':late_fusion(without,ids,w)}


def units(archive,source,dataset,states,manifest,saved):
    ids=tuple(manifest['families'])
    if dataset=='unibo':
        from .unibo_full_fusion import batch,weights
        from emgimu.datasets.unibo_physiology import load_chronological_raw_windows
        target=load_chronological_raw_windows(archive,manifest['target_days']);prob={}
        for n,(f,s,m) in states.items():
            x=f.transform(batch(target));prob[n]=temperature_probability(m.predict_proba(s.transform(x)),manifest['temperatures'][n])
            if n=='F9_Quality':qmean=np.clip(x[:,-3],0,1);qmin=np.clip(x[:,-2],0,1)
        for key,value in (('labels',target.labels),('trials',target.trial_id)):
            np.testing.assert_array_equal(value,saved[key])
        w=weights(target)
        for user in sorted(set(target.subject_id)):
            mask=target.subject_id==user
            yield user,'day_posture',target.labels[mask],np.array([f'posture_{p}' for p in target.posture[mask]]),target.trial_id[mask],w[mask],{n:p[mask] for n,p in prob.items()},qmean[mask],qmin[mask],saved['full'][mask]
    elif dataset=='manus':
        from emgimu.datasets.semg_manus import load_semg_manus_windows
        from .manus_study import GESTURES,_aggregate
        phase=manifest['phase'];target=load_semg_manus_windows(archive,users=range(3,9),sessions=(2 if phase=='validation' else 3,),gestures=GESTURES);prob={}
        for n,(f,s,m) in states.items():
            x,y,u,_,speed,trials=_aggregate(f.transform(target.batch),target)
            prob[n]=temperature_probability(m.predict_proba(s.transform(x)),manifest['temperatures'][n])
            if n=='F9_Quality':qmean=np.clip(x[:,-3],0,1);qmin=np.clip(x[:,-2],0,1)
        np.testing.assert_array_equal(trials,saved['trials']);np.testing.assert_array_equal(y,saved['labels'])
        for user in sorted(set(u)):
            mask=u==user
            yield user,'session_speed',y[mask],speed[mask],trials[mask],np.ones(mask.sum()),{n:p[mask] for n,p in prob.items()},qmean[mask],qmin[mask],saved[f'{user}_0_combined_full']
    else:
        splits=json.loads((source/('split_trial_ids.json' if dataset=='wearing' else 'scenario_split_trial_ids.json')).read_text())
        users=sorted(int(k) for k in splits) if dataset=='wearing' else sorted({r['subject'] for r in splits})
        for user in users:
            if dataset=='wearing':
                from .electrode_shift_study import _aggregate
                prob={}
                from emgimu.datasets.electrode_shift import load_electrode_shift_windows
                target=load_electrode_shift_windows(archive,subjects=(user,),domains=('trial_1','trial_2','trial_3','trial_4'))
                for n in ids:
                    f,s,m=states[(user,n)];x,y,condition,trials=_aggregate(f.transform(target.batch),target)
                    prob[n]=temperature_probability(m.predict_proba(s.transform(x)),manifest['temperatures'][f'{user}_{n}'])
                    if n=='F9_Quality':qmean=np.clip(x[:,-3],0,1);qmin=np.clip(x[:,-2],0,1)
                np.testing.assert_array_equal(trials,saved[f'{user}_trials']);np.testing.assert_array_equal(y,saved[f'{user}_labels'])
                yield user,'wearing',y,condition,trials,np.ones(len(y)),prob,qmean,qmin,saved[f'{user}_prediction_full']
            else:
                from emgimu.datasets.emg_fmg import load_emg_fmg_windows
                from .emg_fmg_study import _trial_probabilities
                data=load_emg_fmg_windows(archive,subjects=(user,),loads=(0,250,500,750,1000),positions=range(1,9))
                for split in (r for r in splits if r['subject']==user):
                    scenario=split['scenario'];target=data.take(np.flatnonzero(np.isin(data.trials,split['test'])));prob={};prefix=f'{user}_{scenario}'
                    for n in ids:
                        f,s,m=states[(user,scenario,n)];x=f.transform(target.batch)
                        p,y,loads,positions=_trial_probabilities(m.predict_proba(s.transform(x)),target)
                        prob[n]=temperature_probability(p,manifest['temperatures'][f'{prefix}_{n}'])
                        if n=='F9_Quality':q,_,_,_=_trial_probabilities(x,target);qmean=np.clip(q[:,-3],0,1);qmin=np.clip(q[:,-2],0,1)
                    trials=np.unique(target.trials);np.testing.assert_array_equal(trials,saved[f'{prefix}_trials']);np.testing.assert_array_equal(y,saved[f'{prefix}_labels'])
                    yield user,scenario,y,loads.astype(str) if scenario=='load_shift' else positions.astype(str),trials,np.ones(len(y)),prob,qmean,qmin,saved[f'{prefix}_prediction_full']


def run(archive,source,output,dataset):
    if output.exists():raise FileExistsError(output)
    manifest=json.loads((source/'run_manifest.json').read_text());phase=manifest['phase']
    if dataset=='unibo':
        state_path=Path(manifest['source_package'])/'source_fitted_states.pkl';states,_=pickle.loads(state_path.read_bytes())
        from .unibo_study import _metrics as score
    elif dataset=='manus':
        state_path=source/'fitted_states.pkl';states,_=pickle.loads(state_path.read_bytes())
        manifest['temperatures']=json.loads((source/'probability_calibration.json').read_text())['temperatures']
        from .manus_study import _metrics
        def score(y,p,w):return _metrics(y,p)
    else:
        state_path=source/'fitted_states.pkl';states=pickle.loads(state_path.read_bytes())
        if dataset=='wearing':from .electrode_shift_study import _metrics
        else:from .emg_fmg_study import _metrics
        def score(y,p,w):return _metrics(y,p)
    before=pickle.dumps(states);rows=[];collected={};predictions={};full_checked=0;error=0.
    with np.load(source/'heldout_predictions.npz',allow_pickle=False) as saved:
        for user,scenario,y,condition,trials,w,prob,qmean,qmin,reference in units(archive,source,dataset,states,manifest,saved):
            print(f'{dataset} {phase}: frozen subject {user} {scenario}',flush=True)
            ps=role_variants(prob,tuple(manifest['families']),qmean,qmin)
            np.testing.assert_allclose(ps['full'],reference,rtol=1e-7,atol=1e-8)
            error=max(error,float(np.abs(ps['full']-reference).max()));full_checked+=1
            prefix=f'{user}_{scenario}';predictions[f'{prefix}_labels']=y;predictions[f'{prefix}_trials']=trials
            for method,p in ps.items():
                predictions[f'{prefix}_{method}']=p
                for value in ('ALL',*sorted(set(condition))):
                    mask=np.ones(len(y),bool) if value=='ALL' else condition==value
                    row={'dataset':{'unibo':'unibo_inail','wearing':'libemg_electrode_shift','manus':'semg_manus','load_position':'emg_fmg'}[dataset],
                        'phase':phase,'subject':user,'scenario':scenario,'condition':value,'shots_per_class':0,
                        'method':method,'feature_bank':'|'.join(manifest['families']),**score(y[mask],p[mask],w[mask])}
                    rows.append(row);collected.setdefault((scenario,value,method),[]).append((row,y[mask],p[mask],w[mask]))
    for _,values in collected.items():
        if dataset in ('unibo','wearing'):
            result=score(np.concatenate([v[1] for v in values]),np.concatenate([v[2] for v in values]),np.concatenate([v[3] for v in values]))
        else:
            result={k:float(np.mean([v[0][k] for v in values])) for k in ('macro_f1','accuracy','log_loss','brier','ece')}
            pc=[json.loads(v[0]['per_class_f1_json']) for v in values]
            result['per_class_f1_json']=json.dumps({k:float(np.mean([v[k] for v in pc])) for k in pc[0]})
        rows.append({**values[0][0],'subject':'ALL',**result})
    if pickle.dumps(states)!=before:raise AssertionError('ablation changed frozen states')
    output.mkdir(parents=True)
    with (output/'ablation_full_bank.csv').open('w',newline='',encoding='utf-8') as h:
        writer=csv.DictWriter(h,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    np.savez_compressed(output/'heldout_predictions.npz',**predictions)
    (output/'run_manifest.json').write_text(json.dumps({'source_run':source.name,'phase':phase,'dataset':dataset,
        'source_state_sha256':hashlib.sha256(state_path.read_bytes()).hexdigest(),
        'source_predictions_sha256':hashlib.sha256((source/'heldout_predictions.npz').read_bytes()).hexdigest(),
        'classifier_or_family_fit':False,'target_calibration_trials':0,'full_units_replayed':full_checked,
        'max_full_probability_error':error,'scope':'zero-target-calibration joint quality-role removal; other calibrated budgets excluded'},indent=2))
    print(json.dumps({'status':'ok','rows':len(rows),'full_units_replayed':full_checked,'max_full_probability_error':error}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--dataset',choices=('wearing','load_position','unibo','manus'),required=True)
    a=p.parse_args();run(a.archive,a.source,a.output,a.dataset)
