"""Offline diagnosis from trusted local fitted states; never fit to evaluation data."""
from pathlib import Path
import argparse
import csv
import json
import pickle
import hashlib
import numpy as np
from emgimu.datasets.semg_manus import load_semg_manus_windows
from .manus_study import GESTURES, _aggregate, _metrics


def run(archive:Path,run_root:Path,output:Path,phase:str,dataset:str='semg_manus')->None:
    if output.exists():raise FileExistsError(output)
    session=2 if phase=='validation' else 3
    aggregate=_aggregate;score=_metrics;users=tuple(range(3,9))
    if dataset=='epn612':
        from emgimu.datasets.epn612 import load_epn612_windows
        from .epn_study import aggregate_trials, _metrics as epn_metrics
        users=(16,17,18) if phase=='validation' else (19,20,21)
        source=load_epn612_windows(archive,users=range(1,16));target=load_epn612_windows(archive,users=users)
        def aggregate(x,data):
            x,y,u,trials,_=aggregate_trials(x,data)
            return x,y,u,None,None,trials
        def score(y,p):return epn_metrics(y,p,np.ones(len(y)))
    else:
        source=load_semg_manus_windows(archive,users=users,sessions=(1,),gestures=GESTURES)
        target=load_semg_manus_windows(archive,users=users,sessions=(session,),gestures=GESTURES)
    state_path=run_root/'fitted_states.pkl';states,anchors=pickle.loads(state_path.read_bytes())
    splits=json.loads((run_root/'split_trial_ids.json').read_text(encoding='utf-8'))
    before=pickle.dumps(states);features={};raw={};drift=[]
    print('[1/2] transforming frozen families and measuring class-matched session drift',flush=True)
    for name,(family,scaler,model) in states.items():
        a,ay,au,_,_,_= aggregate(family.transform(source.batch),source)
        b,y,u,_,_,trials= aggregate(family.transform(target.batch),target)
        a=scaler.transform(a);b=scaler.transform(b);features[name]=b;raw[name]=model.predict_proba(b)
        for user in users:
            source_centers=np.stack([a[(ay==label) if dataset=='epn612' else (au==user)&(ay==label)].mean(0) for label in range(6)])
            target_centers=np.stack([b[(u==user)&(y==label)].mean(0) for label in range(6)])
            dn=float(np.linalg.norm(target_centers-source_centers,axis=1).mean())
            dg=float(np.mean([np.linalg.norm(target_centers[i]-target_centers[j]) for i in range(6) for j in range(i+1,6)]))
            drift.append({'dataset':dataset,'phase':phase,'subject':user,'family':name,
                'D_nuisance':dn,'D_gesture':dg,'J':dg/(dn+1e-12),'feature_dimension':b.shape[1],
                'definition':('train-standardized source-population to target-user class displacement / within-target-user class separation' if dataset=='epn612' else 'train-standardized same-user class centroid session displacement / target within-user class separation')})
    if pickle.dumps(states)!=before:raise AssertionError('evaluation transform mutated fitted state')
    gains=[];complementarity=[];anchor_centroids={}
    print('[2/2] evaluating calibrated versus uncalibrated providers on identical trials',flush=True)
    for split in splits:
        user=split['user'];shots=split['shots'];ev=np.flatnonzero(np.isin(trials,split['evaluation']))
        if set(split['calibration'])&set(split['evaluation']):raise AssertionError('trial leakage')
        predictions={}
        for name in states:
            p=raw[name][ev];calibrated=p
            if shots:
                anchor,temperature=anchors[(user,shots,name)]
                logits=-anchor.transform(features[name][ev])[:,:6].astype(float)/temperature
                logits-=logits.max(1,keepdims=True);q=np.exp(logits);q/=q.sum(1,keepdims=True)
                alpha=shots/(shots+2);calibrated=(1-alpha)*p+alpha*q
                if dataset=='epn612':
                    relative=anchor.transform(features[name][ev])[:,:6]
                    for view,coordinates in (('raw_source_standardized',features[name][ev]),('personal_anchor_distances',relative)):
                        for label in range(6):
                            anchor_centroids[(name,shots,view,user,label)]=coordinates[y[ev]==label].mean(0)
            base=score(y[ev],p);personal=score(y[ev],calibrated)
            gains.append({'dataset':dataset,'phase':phase,'subject':user,'family':name,'shots_per_class':shots,
                'zero_shot_same_eval_macro_f1':base['macro_f1'],'calibrated_macro_f1':personal['macro_f1'],
                'Gain_cal_macro_f1':personal['macro_f1']-base['macro_f1'],
                'Gain_cal_logloss':base['log_loss']-personal['log_loss'],'evaluation_trials':len(ev)})
            predictions[name+'_raw']=p;predictions[name+'_anchor']=calibrated
        pairs=(('F1_X1H_raw','F4_Spectral_raw'),('F1_X1H_raw','F2b_CSP_raw'),
               ('F3_Ring_raw','F3_Ring_anchor'),('F0_raw','F3_Ring_raw'),('F5_Temporal_raw','F3_Ring_anchor'))
        for a,b in pairs:
            ea=predictions[a].argmax(1)!=y[ev];eb=predictions[b].argmax(1)!=y[ev]
            corr=float(np.corrcoef(ea,eb)[0,1]) if ea.std()>0 and eb.std()>0 else 0.
            complementarity.append({'dataset':dataset,'phase':phase,'subject':user,'shots_per_class':shots,
                'family_a':a,'family_b':b,'error_correlation':corr,'disagreement':float(np.mean(ea!=eb)),
                'a_correct_b_wrong':float(np.mean(~ea&eb)),'a_wrong_b_correct':float(np.mean(ea&~eb))})
    output.mkdir(parents=True)
    if anchor_centroids:
        variation=[]
        for name in states:
            for shots in (1,2,5):
                for view in ('raw_source_standardized','personal_anchor_distances'):
                    nuisance=[];gesture=[]
                    for label in range(6):
                        cs=[anchor_centroids[(name,shots,view,user,label)] for user in users]
                        nuisance.extend(np.linalg.norm(a-b) for i,a in enumerate(cs) for b in cs[i+1:])
                    for user in users:
                        cs=[anchor_centroids[(name,shots,view,user,label)] for label in range(6)]
                        gesture.extend(np.linalg.norm(a-b) for i,a in enumerate(cs) for b in cs[i+1:])
                    dn=float(np.mean(nuisance));dg=float(np.mean(gesture))
                    variation.append({'dataset':dataset,'phase':phase,'family':name,'shots_per_class':shots,'view':view,
                        'D_cross_user':dn,'D_gesture':dg,'J':dg/(dn+1e-12),
                        'comparison':'same evaluation trials; compare dimensionless J, not raw distances across coordinate systems'})
        with (output/'anchor_variation_diagnostics.csv').open('w',newline='',encoding='utf-8') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(variation[0]));writer.writeheader();writer.writerows(variation)
    for name,rows in (('per_family_calibration_gain',gains),('cross_user_family_diagnostics' if dataset=='epn612' else 'cross_session_family_diagnostics',drift),('error_complementarity',complementarity)):
        with (output/f'{name}.csv').open('w',newline='',encoding='utf-8') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/'run_manifest.json').write_text(json.dumps({'phase':phase,'source_run':run_root.name,
        'dataset':dataset,'state_sha256':hashlib.sha256(state_path.read_bytes()).hexdigest(),'fit_performed':False,
        'evaluation_label_use':'offline diagnostics only; no parameter selection','transform_state_immutable':True},indent=2),encoding='utf-8')
    print(json.dumps({'status':'ok','calibration_cells':len(gains),'drift_cells':len(drift)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('run_root',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase',choices=('validation','final'),required=True)
    p.add_argument('--dataset',choices=('epn612','semg_manus'),default='semg_manus');a=p.parse_args();run(a.archive,a.run_root,a.output,a.phase,a.dataset)
