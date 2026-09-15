"""Force-only mechanism diagnostics exclude No Movement; classifiers stay frozen."""
from itertools import combinations
from pathlib import Path
import argparse
import csv
import hashlib
import json
import pickle
import numpy as np
from sklearn.metrics import accuracy_score,f1_score,log_loss
from emgimu.datasets.libemg_force import load_libemg_force_windows
from emgimu.feature_bank.force_full_fusion import aggregate
from emgimu.feature_bank.calibration_study import CONDITIONS


def export(raw,processed,output):
    state_path=processed/'feature_bank_force_full_fusion_validation_20260915/fitted_states.pkl'
    states,_=pickle.loads(state_path.read_bytes());before=pickle.dumps(states);rows=[]
    for phase,users in (('validation',(7,8)),('final',(9,10))):
        data=load_libemg_force_windows(raw,subjects=users,conditions=CONDITIONS)
        condition_by_trial={t:np.unique(data.conditions[data.trials==t]).item() for t in set(data.trials)}
        for name,(family,scaler,model) in states.items():
            x,y,u,trials=aggregate(family.transform(data.batch),data);x=scaler.transform(x)
            condition=np.array([condition_by_trial[t] for t in trials]);p=model.predict_proba(x)
            for user in users:
                active=(u==user)&(y!=0);labels=tuple(range(1,7));centers={}
                for value in CONDITIONS:
                    centers[value]=np.stack([x[active&(condition==value)&(y==h)].mean(0) for h in labels])
                pair_drifts=[np.linalg.norm(centers[a]-centers[b],axis=1).mean() for a,b in combinations(CONDITIONS,2)]
                separations={value:float(np.mean([np.linalg.norm(c[a]-c[b]) for a,b in combinations(range(6),2)])) for value,c in centers.items()}
                for value in ('ALL',*CONDITIONS):
                    mask=active if value=='ALL' else active&(condition==value)
                    dn=float(np.mean(pair_drifts)) if value=='ALL' else float(np.mean([np.linalg.norm(centers[value]-centers[b],axis=1).mean() for b in CONDITIONS if b!=value]))
                    dg=float(np.mean(list(separations.values()))) if value=='ALL' else separations[value]
                    truth=y[mask];prob=p[mask];prediction=prob.argmax(1)
                    rows.append({'dataset':'libemg_contraction_intensity','phase':phase,'subject':user,'condition':value,'family':name,
                        'feature_dimension':x.shape[1],'active_native_labels_json':json.dumps(labels),'rest_label_excluded':0,
                        'D_nuisance_active':dn,'D_gesture_active':dg,'J_active':dg/(dn+1e-10),
                        'macro_f1_active6':float(f1_score(truth,prediction,labels=labels,average='macro',zero_division=0)),
                        'accuracy_active':float(accuracy_score(truth,prediction)),
                        'log_loss_active_truth_7way_probability':float(log_loss(truth,prob,labels=np.arange(7))),
                        'active_trials':int(mask.sum()),'rest_trials_excluded':int(np.sum((u==user)&(y==0)&(np.ones(len(y),bool) if value=='ALL' else condition==value)))})
    if pickle.dumps(states)!=before:raise AssertionError('Force diagnostics changed frozen state')
    output.mkdir(parents=True,exist_ok=True)
    with (output/'active_force_family_diagnostics.csv').open('w',newline='',encoding='utf-8') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    audit={'status':'ok','rows':len(rows),'source_state_sha256':hashlib.sha256(state_path.read_bytes()).hexdigest(),
        'original_info_sha256':hashlib.sha256((raw/'Info.txt').read_bytes()).hexdigest(),'classifier_or_family_fit':False,
        'scope':'offline class-matched active-force sensitivity/separation; target labels used for descriptive diagnosis only, never transforms or models',
        'D_nuisance':'mean Euclidean same-active-class centroid distance across target force-condition pairs',
        'D_gesture':'mean pairwise active-class centroid separation averaged over force conditions',
        'limitations':'source-standardized native feature coordinates; J is not mutual information; source trained7way classification remains unchanged'}
    (output/'active_force_diagnostics_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps({'status':'ok','rows':len(rows)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('processed',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();export(a.raw,a.processed,a.output)
