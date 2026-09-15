"""Source-force-only personal anchors, evaluated at unseen target intensities."""
from pathlib import Path
import argparse
import csv
import json
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from emgimu.datasets.libemg_force import load_libemg_force_windows
from .calibration import PersonalAnchor
from .calibration_study import FROZEN_FAMILIES, CONDITIONS
from .screening import FAMILY_FACTORIES, CLASSES, SEED, metrics


def run(root:Path,output:Path,phase:str)->None:
    if output.exists():raise FileExistsError(output)
    users=(7,8) if phase=='validation' else (9,10)
    print(f'[1/3] loading source-force Ramp and unseen-intensity {phase}',flush=True)
    train=load_libemg_force_windows(root,subjects=range(1,7),conditions=('Ramp',))
    cal=load_libemg_force_windows(root,subjects=users,conditions=('Ramp',))
    target=load_libemg_force_windows(root,subjects=users,conditions=CONDITIONS)
    aa=[];bb=[];cc=[];families=[]
    for name in FROZEN_FAMILIES:
        f=FAMILY_FACTORIES[name]();aa.append(f.fit_transform(train.batch,train.labels))
        bb.append(f.transform(target.batch));cc.append(f.transform(cal.batch));families.append(f)
    a=np.concatenate(aa,1);b=np.concatenate(bb,1);c=np.concatenate(cc,1)
    scaler=StandardScaler().fit(a,sample_weight=train.sample_weight);a=scaler.transform(a);b=scaler.transform(b);c=scaler.transform(c)
    model=LogisticRegression(C=1,class_weight='balanced',max_iter=2000,random_state=SEED).fit(a,train.labels,sample_weight=train.sample_weight)
    base=model.predict_proba(b);rows=[];splits=[];anchors={};saved={}
    print('[2/3] fitting personal anchors using Ramp calibration only',flush=True)
    for user in users:
        for shots in (0,1,2,5):
            if shots==5:continue
            rng=np.random.default_rng(SEED+user);chosen=[]
            for label in CLASSES:
                ids=np.unique(cal.trials[(cal.subjects==user)&(cal.labels==label)])
                if len(ids)<shots:raise ValueError('unsupported calibration budget')
                chosen.extend(rng.permutation(ids)[:shots])
            ci=np.flatnonzero(np.isin(cal.trials,chosen));ev=np.flatnonzero(target.subjects==user)
            if set(chosen)&set(target.trials[ev]):raise AssertionError('calibration leakage')
            probability=base[ev]
            if shots:
                anchor=PersonalAnchor().fit(c[ci],cal.labels[ci])
                # Temperature uses calibration distances only; no evaluation distribution fit.
                own=anchor.transform(c[ci])[:,:len(CLASSES)]
                temperature=max(float(np.median(own)),1e-10)
                distance=anchor.transform(b[ev])[:,:len(CLASSES)].astype(float)
                logits=-distance/temperature;logits-=logits.max(1,keepdims=True)
                p=np.exp(logits);p/=p.sum(1,keepdims=True)
                alpha=shots/(shots+2);probability=(1-alpha)*base[ev]+alpha*p
                anchors[(user,shots)]=(anchor,temperature)
            for condition in ('ALL',*CONDITIONS):
                selected=np.ones(len(ev),dtype=bool) if condition=='ALL' else target.conditions[ev]==condition
                rows.append({'dataset':'libemg_contraction_intensity','phase':phase,'protocol':'Force-ZeroShot',
                    'subject':user,'condition':condition,'shots_per_class':shots,'feature_bank':'+'.join(FROZEN_FAMILIES),
                    'mode':'source_force_anchor' if shots else 'population_zero_shot',**metrics(target.labels[ev][selected],probability[selected],target.sample_weight[ev][selected])})
            saved[f'{user}_{shots}']=probability
            splits.append({'user':user,'shots':shots,'calibration':list(chosen),'evaluation':sorted(set(target.trials[ev].tolist()))})
    for shots in (0,1,2):
        for condition in ('ALL',*CONDITIONS):
            subset=[r for r in rows if r['shots_per_class']==shots and r['condition']==condition]
            rows.append({**subset[0],'subject':'ALL',**{k:float(np.mean([r[k] for r in subset])) for k in ('macro_f1','accuracy','log_loss','brier','ece')},'per_class_f1_json':'mean per-user aggregation'})
    print('[3/3] saving disjoint force protocol evidence',flush=True);output.mkdir(parents=True)
    with (output/'calibration_curve.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/'split_trial_ids.json').write_text(json.dumps(splits,indent=2),encoding='utf-8')
    (output/'run_manifest.json').write_text(json.dumps({'phase':phase,'seed':SEED,'protocol':'Force-ZeroShot',
        'source_train_users':list(range(1,7)),'target_users':users,'calibration_force':'Ramp only',
        'evaluation_force':CONDITIONS,'temperature_fit':'calibration only','alpha':'shots/(shots+2)',
        'unsupported_5':'only four Ramp trials/class'},indent=2),encoding='utf-8')
    with (output/'fitted_states.pkl').open('wb') as handle:pickle.dump((families,scaler,model,anchors),handle)
    np.savez_compressed(output/'heldout_predictions.npz',**saved,labels=target.labels,trials=target.trials,subjects=target.subjects)
    print(json.dumps({'status':'ok','rows':len(rows)}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=('validation','final'),required=True);args=parser.parse_args();run(args.root,args.output,args.phase)
