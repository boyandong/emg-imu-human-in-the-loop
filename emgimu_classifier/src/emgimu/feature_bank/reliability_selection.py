"""Select reliability temperature/shrinkage on source-user OOF folds only."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import pickle
import numpy as np
from .calibration import ReliabilityWeights, late_fusion
from .force_nested_oof import protocol, aggregate
from .screening import SEED


def population(probabilities, labels, ids):
    loss=np.array([-np.log(np.maximum(probabilities[n][np.arange(len(labels)),labels],1e-15)).mean() for n in ids])
    score=np.exp(-loss+loss.min())
    return score/score.sum()


def run(archive:Path, oof:Path, output:Path, dataset:str):
    if output.exists():raise FileExistsError(output)
    data,ids,_,score,users,_,dataset_id,condition=protocol(archive,dataset)
    manifest=json.loads((oof/'run_manifest.json').read_text())
    if manifest['dataset']!=dataset_id or set(manifest['source_users'])!=set(users):
        raise AssertionError('Source OOF dataset/users differ')
    partitions=json.loads((oof/'split_trial_ids.json').read_text())
    states=pickle.loads((oof/'fitted_states.pkl').read_bytes());before=pickle.dumps(states)
    rows=[];saved={};cal_splits=[]
    with np.load(oof/'oof_predictions.npz',allow_pickle=False) as predictions, np.load(oof/'inner_oof_predictions.npz',allow_pickle=False) as inner:
        if set(predictions['trials'])!=set(data.trials):raise AssertionError('Source trial mismatch')
        for split in partitions:
            fold=split['fold'];train=set(split['train']);held_ids=set(split['validation'])
            if train&held_ids:raise AssertionError('Outer leakage')
            inner_labels=np.concatenate([inner[f'{fold}_{i}_labels'] for i in range(len(split['inner']))])
            inner_trials=np.concatenate([inner[f'{fold}_{i}_trials'] for i in range(len(split['inner']))])
            if set(inner_trials)!=train:raise AssertionError('Population prior must use inner training OOF only')
            prior=population({n:np.concatenate([inner[f'{fold}_{i}_{n}'] for i in range(len(split['inner']))]) for n in ids},inner_labels,ids)
            held=data.take(np.flatnonzero(np.isin(data.trials,list(held_ids))))
            features={};prob={}
            for n in ids:
                family,scaler,model=states[(fold,n)]
                x,y,u,trials=aggregate(family.transform(held.batch),held)
                features[n]=scaler.transform(x)
                idx=np.array([np.flatnonzero(predictions['trials']==t).item() for t in trials])
                np.testing.assert_array_equal(y,predictions['labels'][idx])
                np.testing.assert_array_equal(u,predictions['users'][idx])
                np.testing.assert_array_equal(predictions['folds'][idx],np.full(len(idx),fold))
                np.testing.assert_allclose(model.predict_proba(features[n]),predictions[f'raw_{n}'][idx],atol=1e-8,rtol=1e-7)
                prob[n]=predictions[f'calibrated_{n}'][idx]
                saved[f'{fold}_features_{n}']=features[n]
            saved[f'{fold}_population']=prior;saved[f'{fold}_labels']=y;saved[f'{fold}_users']=u;saved[f'{fold}_trials']=trials
            print(f'source fold {fold}: users {sorted(set(u))}; reliability grid only',flush=True)
            for user in sorted(set(u)):
                for shots in (1,2):
                    rng=np.random.default_rng(SEED+int(user));chosen=[]
                    for label in sorted(set(y)):
                        candidates=np.flatnonzero((u==user)&(y==label))
                        if len(candidates)<=shots:raise ValueError('Insufficient whole trials')
                        chosen.extend(rng.permutation(candidates)[:shots])
                    cal=np.array(chosen);ev=np.flatnonzero((u==user)&~np.isin(np.arange(len(y)),cal))
                    if set(trials[cal])&set(trials[ev]):raise AssertionError('Calibration leakage')
                    cal_splits.append({'fold':fold,'user':int(user),'shots':shots,'calibration':trials[cal].tolist(),'evaluation':trials[ev].tolist()})
                    for n0 in (2.,8.,32.):
                        for tau in (.5,1.,2.):
                            rule=ReliabilityWeights(tuple(sorted(set(y))),ids,prior,n0=n0,temperature=tau)
                            w=rule.personal({n:(features[n][cal],y[cal]) for n in ids})
                            p=late_fusion({n:prob[n][ev] for n in ids},ids,w)
                            rows.append({'dataset':dataset_id,'phase':'source_cv','subject':int(user),'condition':condition,
                                'shots_per_class':shots,'n0':n0,'reliability_temperature':tau,'method':'reliability_without_anchor',
                                **score(y[ev],p,np.ones(len(ev)))})
                            saved[f'{fold}_{user}_{shots}_{n0}_{tau}_probability']=p
            for n in ids:saved[f'{fold}_probability_{n}']=prob[n]
        deployment_population=population({n:predictions[f'calibrated_{n}'] for n in ids},predictions['labels'],ids)
    if pickle.dumps(states)!=before:raise AssertionError('Source selection mutated fitted state')
    summary=[]
    for n0 in (2.,8.,32.):
        for tau in (.5,1.,2.):
            cells=[r for r in rows if r['n0']==n0 and r['reliability_temperature']==tau]
            summary.append({'n0':n0,'reliability_temperature':tau,'mean_log_loss':float(np.mean([r['log_loss'] for r in cells])),
                'mean_macro_f1':float(np.mean([r['macro_f1'] for r in cells])),'cells':len(cells)})
    selected=min(summary,key=lambda r:(r['mean_log_loss'],r['n0'],r['reliability_temperature']))
    output.mkdir(parents=True)
    with (output/'calibration_curve.csv').open('w',newline='',encoding='utf-8') as h:
        writer=csv.DictWriter(h,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    np.savez_compressed(output/'source_selection_predictions.npz',**saved)
    (output/'split_trial_ids.json').write_text(json.dumps(cal_splits,indent=2))
    policy={'dataset':dataset_id,'source_run':oof.name,'source_users':list(users),'source_trials':sorted(set(data.trials)),
        'families':ids,'selected':selected,'grid':summary,'population_weights':deployment_population.tolist(),
        'source_state_sha256':hashlib.sha256((oof/'fitted_states.pkl').read_bytes()).hexdigest(),
        'source_oof_sha256':hashlib.sha256((oof/'oof_predictions.npz').read_bytes()).hexdigest(),
        'target_data_opened':False,'classifier_or_family_fit':False,
        'scope':'source-user OOF hyperparameter selection; inner OOF priors; target calibration prototypes excluded from source classifiers; selected CV score is not unbiased final performance'}
    (output/'run_manifest.json').write_text(json.dumps(policy,indent=2))
    print(json.dumps({'status':'ok','selected':selected,'rows':len(rows),'target_data_opened':False}))


def replay(output:Path):
    policy=json.loads((output/'run_manifest.json').read_text());ids=tuple(policy['families'])
    splits=json.loads((output/'split_trial_ids.json').read_text());checked=set();error=0.
    with np.load(output/'source_selection_predictions.npz',allow_pickle=False) as saved:
        for split in splits:
            fold=split['fold'];user=split['user'];shots=split['shots'];trials=saved[f'{fold}_trials'];y=saved[f'{fold}_labels']
            cal=np.flatnonzero(np.isin(trials,split['calibration']));ev=np.flatnonzero(np.isin(trials,split['evaluation']))
            if set(trials[cal])&set(trials[ev]):raise AssertionError('Calibration leakage')
            np.testing.assert_array_equal(saved[f'{fold}_users'][cal],np.full(len(cal),user))
            np.testing.assert_array_equal(saved[f'{fold}_users'][ev],np.full(len(ev),user))
            for point in policy['grid']:
                n0=point['n0'];tau=point['reliability_temperature']
                rule=ReliabilityWeights(tuple(sorted(set(y))),ids,saved[f'{fold}_population'],n0=n0,temperature=tau)
                w=rule.personal({n:(saved[f'{fold}_features_{n}'][cal],y[cal]) for n in ids})
                p=late_fusion({n:saved[f'{fold}_probability_{n}'][ev] for n in ids},ids,w)
                key=f'{fold}_{user}_{shots}_{n0}_{tau}_probability'
                np.testing.assert_allclose(p,saved[key],atol=1e-10,rtol=1e-9)
                error=max(error,float(np.abs(p-saved[key]).max()));checked.add(key)
        if checked!={k for k in saved.files if k.endswith('_probability')}:
            raise AssertionError('Incomplete selection replay')
    result={'status':'ok','prediction_arrays_checked':len(checked),'max_absolute_probability_error':error,
        'classifier_or_family_fit':False,'target_data_opened':False,
        'scope':'all source selection probability arrays rebuilt from retained frozen fold features and calibration trial IDs'}
    (output/'replay_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('oof',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--dataset',choices=('epn612','semg_manus'),required=True)
    p.add_argument('--replay',action='store_true')
    a=p.parse_args();replay(a.output) if a.replay else run(a.archive,a.oof,a.output,a.dataset)
