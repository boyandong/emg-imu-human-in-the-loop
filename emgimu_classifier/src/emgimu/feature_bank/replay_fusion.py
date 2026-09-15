"""Replay trusted local fitted fusion states without classifier/family fitting."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import pickle
import numpy as np
from .calibration import ReliabilityWeights, SessionSignature, late_fusion
from .epn_study import aggregate_trials
from .full_fusion_study import FAMILIES


def replay(archive:Path,run_root:Path,dataset:str,phase:str,joint_output:Path|None=None)->dict:
    if joint_output is not None:
        if dataset!='semg_manus':raise ValueError('joint calibrated quality audit requires MANUS')
        if joint_output.exists():raise FileExistsError(joint_output)
    states,anchors=pickle.loads((run_root/'fitted_states.pkl').read_bytes())
    before=pickle.dumps(states);splits=json.loads((run_root/'split_trial_ids.json').read_text(encoding='utf-8'))
    calibration_path=run_root/'probability_calibration.json'
    temperatures=json.loads(calibration_path.read_text())['temperatures'] if calibration_path.exists() else {}
    users=sorted({s['user'] for s in splits});source=None
    if dataset=='semg_manus':
        from emgimu.datasets.semg_manus import load_semg_manus_windows
        from .manus_study import GESTURES,_aggregate
        target=load_semg_manus_windows(archive,users=users,sessions=(2 if phase=='validation' else 3,),gestures=GESTURES)
        source=load_semg_manus_windows(archive,users=users,sessions=(1,),gestures=GESTURES)
        def aggregate(x,data):
            x,y,u,_,_,trials=_aggregate(x,data)
            return x,y,u,trials,None
    else:
        from emgimu.datasets.epn612 import load_epn612_windows
        target=load_epn612_windows(archive,users=users);aggregate=aggregate_trials
    features={};raw={};profiles={}
    for name,(family,scaler,model) in states.items():
        b,y,u,trials,_=aggregate(family.transform(target.batch),target)
        if name=='F9_Quality':qmean=np.clip(b[:,-3],0,1);qmin=np.clip(b[:,-2],0,1)
        features[name]=scaler.transform(b);raw[name]=model.predict_proba(features[name])
        if temperatures:
            from .force_nested_oof import temperature_probability
            raw[name]=temperature_probability(raw[name],temperatures[name])
        if source is not None:
            a,ay,au,_,_=aggregate(family.transform(source.batch),source);profiles[name]=scaler.transform(a)
    population=np.ones(len(FAMILIES))/len(FAMILIES)
    reliability=ReliabilityWeights(tuple(range(6)),FAMILIES,population,n0=8)
    checked=set();max_error=0.;joint_rows=[];joint_predictions={};groups={}
    with np.load(run_root/'heldout_predictions.npz',allow_pickle=False) as saved:
        for key,value in (('labels',y),('users',u),('trials',trials)):
            np.testing.assert_array_equal(saved[key],value)
        for split in splits:
            user=split['user'];shots=split['shots'];cal=np.flatnonzero(np.isin(trials,split['calibration']))
            ev=np.flatnonzero(np.isin(trials,split['evaluation']))
            if set(trials[cal])&set(trials[ev]):raise AssertionError('trial leakage')
            weights=population if shots==0 else reliability.personal({n:(features[n][cal],y[cal]) for n in FAMILIES})
            personal={n:raw[n][ev] for n in FAMILIES}
            if shots:
                for n in FAMILIES:
                    anchor,temp=anchors[(user,shots,n)]
                    logits=-anchor.transform(features[n][ev])[:,:6].astype(float)/temp
                    logits-=logits.max(1,keepdims=True);p=np.exp(logits);p/=p.sum(1,keepdims=True)
                    alpha=shots/(shots+2);personal[n]=(1-alpha)*personal[n]+alpha*p
            variants={'full':(personal,weights,None),'without_F7_anchor':({n:raw[n][ev] for n in FAMILIES},weights,None),
                'uniform_population':({n:raw[n][ev] for n in FAMILIES},population,None),
                **{f'without_{n}':({f:p for f,p in personal.items() if f!=n},weights,None) for n in FAMILIES}}
            if source is not None:
                context=weights.copy()
                if shots:
                    for i,n in enumerate(FAMILIES):
                        signature=SessionSignature().fit_long_term(profiles[n][au==user],ay[au==user])
                        vector=signature.from_session_calibration(features[n][cal],y[cal])
                        context[i]*=float(np.clip((vector[6:12].mean()+1)/2,.05,1))
                    context/=context.sum()
                quality={n:(np.ones(len(ev)) if n in ('F6_IMU','F9_Quality') else qmin[ev] if n in ('F0','F2b_CSP') else qmean[ev]) for n in FAMILIES}
                variants.update({'with_F8_context':(personal,context,None),'with_F8_F9_quality':(personal,context,quality),
                    'combined_full':(personal,context,quality),'combined_without_F8_context':(personal,weights,quality),
                    'combined_without_F9_quality':(personal,context,None),
                    'combined_without_F7_anchor':({n:raw[n][ev] for n in FAMILIES},context,quality),
                    **{f'combined_without_{n}':({f:p for f,p in personal.items() if f!=n},context,quality) for n in FAMILIES}})
            for method,(providers,w,q) in variants.items():
                key=f'{user}_{shots}_{method}'
                if key not in saved.files:continue
                p=late_fusion(providers,FAMILIES,w,q)
                np.testing.assert_allclose(p,saved[key],rtol=1e-7,atol=1e-8,err_msg=key)
                max_error=max(max_error,float(np.abs(p-saved[key]).max()));checked.add(key)
            if joint_output is not None:
                from .manus_study import _metrics
                providers={n:p for n,p in personal.items() if n!='F9_Quality'}
                controls={'combined_full':late_fusion(personal,FAMILIES,context,quality),
                    'without_F9_routing':late_fusion(personal,FAMILIES,context),
                    'without_F9_provider':late_fusion(providers,FAMILIES,context,quality),
                    'without_F9_routing_and_provider':late_fusion(providers,FAMILIES,context)}
                np.testing.assert_allclose(controls['combined_full'],saved[f'{user}_{shots}_combined_full'],rtol=1e-7,atol=1e-8)
                condition=np.array([np.unique(target.speeds[target.trials==trial]).item() for trial in trials[ev]])
                prefix=f'{user}_{shots}';joint_predictions[f'{prefix}_labels']=y[ev];joint_predictions[f'{prefix}_trials']=trials[ev]
                joint_predictions[f'{prefix}_conditions']=condition
                for method,p in controls.items():
                    joint_predictions[f'{prefix}_{method}']=p
                    for value in ('ALL',*sorted(set(condition))):
                        mask=np.ones(len(ev),bool) if value=='ALL' else condition==value
                        if not mask.any():continue
                        row={'dataset':dataset,'phase':phase,'subject':user,'scenario':'session_speed','condition':value,
                            'shots_per_class':shots,'method':method,**_metrics(y[ev][mask],p[mask])}
                        joint_rows.append(row);groups.setdefault((shots,value,method),[]).append(row)
        expected={k for k in saved.files if k not in ('labels','trials','users')}
        if checked!=expected:raise AssertionError(f'unreplayed predictions: {expected-checked}')
    if pickle.dumps(states)!=before:raise AssertionError('replay mutated family/classifier state')
    if joint_output is not None:
        for rows in groups.values():
            result={k:float(np.mean([r[k] for r in rows])) for k in ('macro_f1','accuracy','log_loss','brier','ece')}
            pc=[json.loads(r['per_class_f1_json']) for r in rows]
            result['per_class_f1_json']=json.dumps({k:float(np.mean([v[k] for v in pc])) for k in pc[0]})
            joint_rows.append({**rows[0],'subject':'ALL',**result})
        joint_output.mkdir(parents=True)
        with (joint_output/'ablation_full_bank.csv').open('w',newline='',encoding='utf-8') as h:
            writer=csv.DictWriter(h,fieldnames=list(joint_rows[0]));writer.writeheader();writer.writerows(joint_rows)
        np.savez_compressed(joint_output/'heldout_predictions.npz',**joint_predictions)
        (joint_output/'split_trial_ids.json').write_text(json.dumps(splits,indent=2))
        (joint_output/'run_manifest.json').write_text(json.dumps({'source_run':run_root.name,'dataset':dataset,'phase':phase,
            'source_state_sha256':hashlib.sha256((run_root/'fitted_states.pkl').read_bytes()).hexdigest(),
            'source_predictions_sha256':hashlib.sha256((run_root/'heldout_predictions.npz').read_bytes()).hexdigest(),
            'classifier_or_family_fit':False,'full_units_replayed':len(splits),'rows':len(joint_rows),
            'scope':'quality routing/provider removals; F7 anchors, reliability and F8 context held fixed; same evaluation trials within each budget',
            'budgets':sorted({s['shots'] for s in splits})},indent=2))
    result={'status':'ok','run_id':run_root.name,'prediction_arrays_checked':len(checked),
        'max_absolute_probability_error':max_error,'classifier_or_family_fit':False,
        'session_profile_rebuilt_from_source_only':source is not None,'source_oof_probability_calibration':bool(temperatures),
        'scope':'all saved probability variants in this run'}
    (run_root/'replay_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('run_root',type=Path)
    p.add_argument('--dataset',choices=('epn612','semg_manus'),required=True);p.add_argument('--phase',choices=('validation','final'),required=True)
    p.add_argument('--joint-output',type=Path)
    a=p.parse_args();print(json.dumps(replay(a.archive,a.run_root,a.dataset,a.phase,a.joint_output)))
