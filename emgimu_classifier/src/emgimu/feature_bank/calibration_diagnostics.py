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


def run(archive:Path,run_root:Path,output:Path,phase:str)->None:
    if output.exists():raise FileExistsError(output)
    session=2 if phase=='validation' else 3
    source=load_semg_manus_windows(archive,users=range(3,9),sessions=(1,),gestures=GESTURES)
    target=load_semg_manus_windows(archive,users=range(3,9),sessions=(session,),gestures=GESTURES)
    state_path=run_root/'fitted_states.pkl';states,anchors=pickle.loads(state_path.read_bytes())
    splits=json.loads((run_root/'split_trial_ids.json').read_text(encoding='utf-8'))
    before=pickle.dumps(states);features={};raw={};drift=[]
    print('[1/2] transforming frozen families and measuring class-matched session drift',flush=True)
    for name,(family,scaler,model) in states.items():
        a,ay,au,_,_,_= _aggregate(family.transform(source.batch),source)
        b,y,u,_,_,trials= _aggregate(family.transform(target.batch),target)
        a=scaler.transform(a);b=scaler.transform(b);features[name]=b;raw[name]=model.predict_proba(b)
        for user in range(3,9):
            source_centers=np.stack([a[(au==user)&(ay==label)].mean(0) for label in range(6)])
            target_centers=np.stack([b[(u==user)&(y==label)].mean(0) for label in range(6)])
            dn=float(np.linalg.norm(target_centers-source_centers,axis=1).mean())
            dg=float(np.mean([np.linalg.norm(target_centers[i]-target_centers[j]) for i in range(6) for j in range(i+1,6)]))
            drift.append({'dataset':'semg_manus','phase':phase,'subject':user,'family':name,
                'D_nuisance':dn,'D_gesture':dg,'J':dg/(dn+1e-12),'feature_dimension':b.shape[1],
                'definition':'train-standardized same-user class centroid session displacement / target within-user class separation'})
    if pickle.dumps(states)!=before:raise AssertionError('evaluation transform mutated fitted state')
    gains=[];complementarity=[]
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
            base=_metrics(y[ev],p);personal=_metrics(y[ev],calibrated)
            gains.append({'dataset':'semg_manus','phase':phase,'subject':user,'family':name,'shots_per_class':shots,
                'zero_shot_same_eval_macro_f1':base['macro_f1'],'calibrated_macro_f1':personal['macro_f1'],
                'Gain_cal_macro_f1':personal['macro_f1']-base['macro_f1'],
                'Gain_cal_logloss':base['log_loss']-personal['log_loss'],'evaluation_trials':len(ev)})
            predictions[name+'_raw']=p;predictions[name+'_anchor']=calibrated
        pairs=(('F1_X1H_raw','F4_Spectral_raw'),('F1_X1H_raw','F2b_CSP_raw'),
               ('F3_Ring_raw','F3_Ring_anchor'),('F0_raw','F3_Ring_raw'),('F5_Temporal_raw','F3_Ring_anchor'))
        for a,b in pairs:
            ea=predictions[a].argmax(1)!=y[ev];eb=predictions[b].argmax(1)!=y[ev]
            corr=float(np.corrcoef(ea,eb)[0,1]) if ea.std()>0 and eb.std()>0 else 0.
            complementarity.append({'dataset':'semg_manus','phase':phase,'subject':user,'shots_per_class':shots,
                'family_a':a,'family_b':b,'error_correlation':corr,'disagreement':float(np.mean(ea!=eb)),
                'a_correct_b_wrong':float(np.mean(~ea&eb)),'a_wrong_b_correct':float(np.mean(ea&~eb))})
    output.mkdir(parents=True)
    for name,rows in (('per_family_calibration_gain',gains),('cross_session_family_diagnostics',drift),('error_complementarity',complementarity)):
        with (output/f'{name}.csv').open('w',newline='',encoding='utf-8') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/'run_manifest.json').write_text(json.dumps({'phase':phase,'source_run':run_root.name,
        'state_sha256':hashlib.sha256(state_path.read_bytes()).hexdigest(),'fit_performed':False,
        'evaluation_label_use':'offline diagnostics only; no parameter selection','transform_state_immutable':True},indent=2),encoding='utf-8')
    print(json.dumps({'status':'ok','calibration_cells':len(gains),'drift_cells':len(drift)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('run_root',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase',choices=('validation','final'),required=True);a=p.parse_args();run(a.archive,a.run_root,a.output,a.phase)
