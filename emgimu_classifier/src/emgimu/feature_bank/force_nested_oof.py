"""Nested subject-fold source-only OOF probabilities and temperature calibration."""
from pathlib import Path
import argparse
import csv
import json
import pickle
from itertools import combinations
import numpy as np
from scipy.optimize import minimize_scalar
from emgimu.datasets.libemg_force import load_libemg_force_windows
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from .force_full_fusion import IDS, aggregate
from .screening import FAMILY_FACTORIES, SEED, metrics


def temperature_probability(probability, temperature):
    p=np.asarray(probability,dtype=float)
    if p.ndim!=2 or not np.all(np.isfinite(p)) or np.any(p<0) or not np.allclose(p.sum(1),1):
        raise ValueError('expected finite normalized probabilities')
    if not np.isfinite(temperature) or temperature<=0:raise ValueError('temperature must be positive')
    z=np.log(np.maximum(p,1e-15))/temperature;z-=z.max(1,keepdims=True)
    p=np.exp(z);return p/p.sum(1,keepdims=True)


def fit_temperature(probability, labels):
    y=np.asarray(labels);p=np.asarray(probability)
    if len(y)!=len(p) or not len(y):raise ValueError('calibration labels must align')
    def loss(log_temperature):
        q=temperature_probability(p,np.exp(log_temperature))
        return -np.log(np.maximum(q[np.arange(len(y)),y],1e-15)).mean()
    result=minimize_scalar(loss,bounds=(np.log(.25),np.log(4)),method='bounded')
    candidates=(0.,result.x)
    return float(np.exp(min(candidates,key=loss)))


def predict(train, evaluation, name):
    if set(train.subjects)&set(evaluation.subjects):raise AssertionError('subject leakage')
    if set(train.trials)&set(evaluation.trials):raise AssertionError('trial leakage')
    family=FAMILY_FACTORIES[name]()
    a,ay,_,_=aggregate(family.fit_transform(train.batch,train.labels),train)
    scaler=StandardScaler().fit(a)
    model=LogisticRegression(C=1,class_weight='balanced',max_iter=2000,random_state=SEED).fit(scaler.transform(a),ay)
    state=(family,scaler,model);before=pickle.dumps(state)
    b,y,u,trials=aggregate(family.transform(evaluation.batch),evaluation)
    p=model.predict_proba(scaler.transform(b))
    if pickle.dumps(state)!=before:raise AssertionError('evaluation mutated fitted state')
    return p,y,u,trials,state


def run(root:Path,output:Path):
    if output.exists():raise FileExistsError(output)
    data=load_libemg_force_windows(root,subjects=range(1,7),conditions=('Ramp',))
    _,y,u,trials=aggregate(np.zeros((len(data.labels),1)),data)
    raw={n:np.zeros((len(y),7)) for n in IDS};calibrated={n:np.zeros((len(y),7)) for n in IDS}
    fold_id=np.full(len(y),-1);splits=[];states={};temperatures=[]
    groups=((1,2),(3,4),(5,6))
    for fold,held_users in enumerate(groups):
        source_users=sorted(set(range(1,7))-set(held_users))
        train=data.take(np.flatnonzero(np.isin(data.subjects,source_users)))
        held=data.take(np.flatnonzero(np.isin(data.subjects,held_users)))
        split={'fold':fold,'train':np.unique(train.trials).tolist(),'validation':np.unique(held.trials).tolist(),'inner':[]}
        inner_groups=(source_users[:2],source_users[2:])
        inner_predictions={n:[] for n in IDS};inner_labels=[]
        print(f'[{fold+1}/3] source-user outer fold {held_users}; internal probability calibration',flush=True)
        for inner_users in inner_groups:
            inner_train=train.take(np.flatnonzero(~np.isin(train.subjects,inner_users)))
            inner_eval=train.take(np.flatnonzero(np.isin(train.subjects,inner_users)))
            split['inner'].append({'train':np.unique(inner_train.trials).tolist(),'validation':np.unique(inner_eval.trials).tolist()})
            for n in IDS:
                p,iy,_,_,_=predict(inner_train,inner_eval,n);inner_predictions[n].append(p)
            inner_labels.append(iy)
        for n in IDS:
            temp=fit_temperature(np.concatenate(inner_predictions[n]),np.concatenate(inner_labels))
            p,hy,hu,ht,state=predict(train,held,n)
            index=np.asarray([np.flatnonzero(trials==t).item() for t in ht])
            np.testing.assert_array_equal(y[index],hy);np.testing.assert_array_equal(u[index],hu)
            raw[n][index]=p;calibrated[n][index]=temperature_probability(p,temp)
            fold_id[index]=fold;states[(fold,n)]=state
            temperatures.append({'fold':fold,'family':n,'temperature':temp,'fit_users':source_users})
        splits.append(split)
    if np.any(fold_id<0):raise AssertionError('missing OOF rows')
    rows=[];pairs=[]
    for n in IDS:
        for method,bank in (('raw_nested_oof',raw),('temperature_nested_oof',calibrated)):
            rows.append({'dataset':'libemg_contraction_intensity','phase':'source_nested_oof','family':n,
                'condition':'Ramp','method':method,**metrics(y,bank[n],np.ones(len(y)))})
    for a,b in combinations(IDS,2):
        pa=calibrated[a].argmax(1);pb=calibrated[b].argmax(1);ca=pa==y;cb=pb==y
        correlation=float(np.corrcoef(~ca,~cb)[0,1]) if np.std(ca) and np.std(cb) else ''
        pairs.append({'dataset':'libemg_contraction_intensity','phase':'source_nested_oof','family_a':a,'family_b':b,
            'error_correlation':correlation,'disagreement_rate':float(np.mean(pa!=pb)),
            'a_correct_b_wrong':float(np.mean(ca&~cb)),'a_wrong_b_correct':float(np.mean(~ca&cb))})
    output.mkdir(parents=True)
    for name,values in (('feature_family_results',rows),('error_complementarity',pairs)):
        with (output/f'{name}.csv').open('w',newline='',encoding='utf-8') as h:
            w=csv.DictWriter(h,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    with (output/'fitted_states.pkl').open('wb') as h:pickle.dump(states,h)
    np.savez_compressed(output/'oof_predictions.npz',labels=y,users=u,trials=trials,folds=fold_id,
        **{f'raw_{n}':p for n,p in raw.items()},**{f'calibrated_{n}':p for n,p in calibrated.items()})
    (output/'split_trial_ids.json').write_text(json.dumps(splits,indent=2))
    (output/'run_manifest.json').write_text(json.dumps({'seed':SEED,'families':IDS,'outer_users':groups,
        'source_users':list(range(1,7)),'source_condition':'Ramp','target_users_opened':False,
        'temperature_bounds':[.25,4],'temperatures':temperatures,'probability_calibration':'two inner subject folds per outer fold',
        'fit_scope':'every family, scaler and classifier refit inside its subject fold','evaluation_unit':'trial mean',
        'scope':'source Ramp nested OOF only; does not replace target-force evaluations'},indent=2))
    print(json.dumps({'status':'ok','oof_trials':len(y),'families':len(IDS),'outer_folds':3}))


def replay(root:Path,output:Path):
    manifest=json.loads((output/'run_manifest.json').read_text())
    splits=json.loads((output/'split_trial_ids.json').read_text())
    states=pickle.loads((output/'fitted_states.pkl').read_bytes());before=pickle.dumps(states)
    data=load_libemg_force_windows(root,subjects=manifest['source_users'],conditions=('Ramp',))
    temperature={(r['fold'],r['family']):r['temperature'] for r in manifest['temperatures']}
    checked=0;error=0.
    with np.load(output/'oof_predictions.npz',allow_pickle=False) as saved:
        for split in splits:
            fold=split['fold'];train_ids=set(split['train']);eval_ids=set(split['validation'])
            if train_ids&eval_ids:raise AssertionError('outer overlap')
            for inner in split['inner']:
                a,b=set(inner['train']),set(inner['validation'])
                if a&b or a|b!=train_ids or (a|b)&eval_ids:raise AssertionError('inner overlap or coverage')
            held=data.take(np.flatnonzero(np.isin(data.trials,list(eval_ids))))
            for n in IDS:
                family,scaler,model=states[(fold,n)]
                b,y,u,trials=aggregate(family.transform(held.batch),held)
                index=np.asarray([np.flatnonzero(saved['trials']==t).item() for t in trials])
                np.testing.assert_array_equal(saved['labels'][index],y)
                np.testing.assert_array_equal(saved['users'][index],u)
                np.testing.assert_array_equal(saved['folds'][index],np.full(len(index),fold))
                p=model.predict_proba(scaler.transform(b))
                for key,q in ((f'raw_{n}',p),(f'calibrated_{n}',temperature_probability(p,temperature[(fold,n)]))):
                    np.testing.assert_allclose(q,saved[key][index],atol=1e-8,rtol=1e-7)
                    error=max(error,float(np.abs(q-saved[key][index]).max()));checked+=1
        if set(np.concatenate([np.array(s['validation']) for s in splits]))!=set(saved['trials']):
            raise AssertionError('missing outer trial coverage')
    if pickle.dumps(states)!=before:raise AssertionError('replay mutated fitted state')
    audit={'status':'ok','probability_blocks_checked':checked,'max_absolute_probability_error':error,
        'classifier_or_family_fit':False,'nested_partitions_verified':True,
        'scope':'all outer source OOF probabilities and stored temperatures; inner fitting is recorded by run code and partitions'}
    (output/'replay_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--replay',action='store_true')
    a=p.parse_args();(replay if a.replay else run)(a.root,a.output)
