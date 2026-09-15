"""Trial-level population/personal shrinkage fusion, with untouched evaluation trials."""
from __future__ import annotations
import argparse
import csv
import json
import pickle
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from emgimu.datasets.epn612 import load_epn612_windows
from .calibration import ReliabilityWeights, late_fusion
from .epn_study import aggregate_trials, _metrics
from .screening import FAMILY_FACTORIES, SEED


def run(archive: Path, output: Path, phase: str) -> None:
    if output.exists():
        raise FileExistsError(output)
    users=(16,17,18) if phase=='validation' else (19,20,21)
    print(f'[1/3] loading source and {phase} users {users}',flush=True)
    train=load_epn612_windows(archive,users=range(1,16))
    target=load_epn612_windows(archive,users=users)
    ids=('F0','F3_Ring')
    features={}; probabilities={}; states={}
    for name in ids:
        print(f'[2/3] fitting {name}',flush=True)
        family=FAMILY_FACTORIES[name]()
        source_x,source_y,_,_,_=aggregate_trials(family.fit_transform(train.batch,train.labels),train)
        target_x,y,u,trials,weight=aggregate_trials(family.transform(target.batch),target)
        scaler=StandardScaler().fit(source_x)
        classifier=LogisticRegression(C=1,class_weight='balanced',max_iter=1000,random_state=SEED).fit(scaler.transform(source_x),source_y)
        features[name]=scaler.transform(target_x)
        probabilities[name]=classifier.predict_proba(features[name])
        states[name]=(family,scaler,classifier)
    reliability=ReliabilityWeights(tuple(range(6)),ids,np.asarray([.5,.5]),n0=8)
    rows=[]; split_ids=[]; saved_predictions={}
    for shots in (0,1,2,5):
        for user in users:
            calibration=[]
            rng=np.random.default_rng(SEED+user)
            for label in range(6):
                candidates=np.flatnonzero((u==user)&(y==label))
                calibration.extend(rng.permutation(candidates)[:shots])
            calibration=np.asarray(calibration,dtype=int)
            evaluate=np.flatnonzero((u==user)&~np.isin(np.arange(len(y)),calibration))
            if set(trials[calibration])&set(trials[evaluate]):
                raise AssertionError('calibration/evaluation trial leakage')
            weights=reliability.population if shots==0 else reliability.personal({
                name:(features[name][calibration],y[calibration]) for name in ids})
            for method,w in (('uniform',reliability.population),('personal_shrinkage',weights)):
                p=late_fusion({name:probabilities[name][evaluate] for name in ids},ids,w)
                rows.append({'dataset':'epn612','phase':phase,'subject':user,'condition':'cross_user',
                    'feature_bank':'F0|F3_Ring','method':method,'shots_per_class':shots,
                    'weights_json':json.dumps(w.tolist()),**_metrics(y[evaluate],p,np.ones(len(evaluate)))})
                saved_predictions[f'{user}_{shots}_{method}']=p
            split_ids.append({'user':user,'shots_per_class':shots,'calibration':trials[calibration].tolist(),
                              'evaluation':trials[evaluate].tolist()})
    # Keep aggregation explicit: mean of per-user metrics, never pooled F1.
    for shots in (0,1,2,5):
        for method in ('uniform','personal_shrinkage'):
            selected=[r for r in rows if r['shots_per_class']==shots and r['method']==method]
            rows.append({**selected[0],'subject':'ALL','weights_json':'per-user weights',
                **{key:float(np.mean([r[key] for r in selected])) for key in ('macro_f1','accuracy','log_loss','brier','ece')},
                'per_class_f1_json':'mean per-user aggregation'})
    print('[3/3] saving states, trial splits and held-out predictions',flush=True)
    output.mkdir(parents=True)
    with (output/'calibration_curve.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/'split_trial_ids.json').write_text(json.dumps(split_ids,indent=2),encoding='utf-8')
    (output/'run_manifest.json').write_text(json.dumps({'phase':phase,'seed':SEED,'train_users':list(range(1,16)),
        'target_users':users,'families':ids,'population_weights':[.5,.5],'n0':8,'temperature':1,
        'classifier_C':1,'fit_unit':'source trial averages','aggregation':'mean per-user metrics',
        'personal_fit_unit':'calibration trials only'},indent=2),encoding='utf-8')
    with (output/'fitted_states.pkl').open('wb') as handle:pickle.dump(states,handle)
    np.savez_compressed(output/'heldout_predictions.npz',**saved_predictions)
    print(json.dumps({'status':'ok','rows':len(rows),'output':str(output)}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('archive',type=Path)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--phase',choices=('validation','final'),required=True)
    args=parser.parse_args();run(args.archive,args.output,args.phase)
