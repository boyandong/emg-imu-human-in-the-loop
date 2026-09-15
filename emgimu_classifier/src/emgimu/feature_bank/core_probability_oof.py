"""Source-user OOF probability calibration for frozen concatenated Core specifications."""
from dataclasses import fields, replace
from pathlib import Path
import argparse
import hashlib
import json
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from .screening import FAMILY_FACTORIES, SEED
from .force_nested_oof import fit_temperature, temperature_probability
from .core_incremental_oof import write


def take(data,indices):
    return replace(data,**{f.name:getattr(data,f.name).take(indices) if f.name=='batch' else getattr(data,f.name)[indices] for f in fields(data)})


def run(raw,source,output):
    if output.exists():raise FileExistsError(output)
    manifest=json.loads((source/'run_manifest.json').read_text())
    if manifest['dataset']=='semg_manus':
        from emgimu.datasets.semg_manus import load_semg_manus_windows
        from .manus_study import GESTURES, FACTORIES, _aggregate, _metrics
        if manifest['source_users']!=list(range(3,9)) or manifest['source_session']!=1:raise AssertionError('Only source MANUS session1/users3-8')
        data=load_semg_manus_windows(raw,users=range(3,9),sessions=(1,),gestures=GESTURES)
        factories=FACTORIES;window_users=data.users;max_iter=1000
        def aggregate(x,d):
            x,y,u,_,_,t=_aggregate(x,d);return x,y,u,t
        score=_metrics
    elif manifest['dataset']=='libemg_contraction_intensity':
        from emgimu.datasets.libemg_force import load_libemg_force_windows
        from .force_full_fusion import aggregate
        from .screening import metrics
        if manifest['source_users']!=list(range(1,7)) or manifest['source_condition']!='Ramp':raise AssertionError('Only source force Ramp/users1-6')
        data=load_libemg_force_windows(raw,subjects=range(1,7),conditions=('Ramp',))
        factories=FAMILY_FACTORIES;window_users=data.subjects;max_iter=2000
        def score(y,p):return metrics(y,p,np.ones(len(y)))
    else:raise ValueError('Unsupported native source dataset')
    _,full_models=pickle.loads((source/'fitted_states.pkl').read_bytes())
    specs={name:members for name,(_,_,members) in full_models.items()}
    used=tuple(dict.fromkeys(n for members in specs.values() for n in members))
    _,y,u,trials=aggregate(np.zeros((data.batch.windows,1)),data)
    if set(trials)!=set(manifest['source_trials']):raise AssertionError('Source fitting trial mismatch')
    users=sorted(int(v) for v in set(u));probabilities={name:np.zeros((len(y),len(set(y)))) for name in specs};coverage=np.zeros(len(y),int)
    retained={};fitted={};splits=[]
    for fold,held in enumerate(np.asarray(users).reshape(3,2)):
        train=take(data,np.flatnonzero(~np.isin(window_users,held)));validation=take(data,np.flatnonzero(np.isin(window_users,held)))
        if set(train.trials)&set(validation.trials):raise AssertionError('OOF trial leakage')
        a={};b={};families={};models={}
        for name in used:
            family=factories[name]().fit(train.batch,train.labels)
            a[name],ay,_,at=aggregate(family.transform(train.batch),train)
            before=pickle.dumps(family)
            b[name],by,bu,bt=aggregate(family.transform(validation.batch),validation)
            if pickle.dumps(family)!=before:raise AssertionError('Held fold altered family')
            families[name]=family;retained[f'fold{fold}::{name}']=b[name]
        index=np.array([np.flatnonzero(trials==t).item() for t in bt]);coverage[index]+=1
        for name,members in specs.items():
            x=np.concatenate([a[n] for n in members],1);scaler=StandardScaler().fit(x)
            model=LogisticRegression(C=1,class_weight='balanced',max_iter=max_iter,random_state=SEED).fit(scaler.transform(x),ay)
            if int(model.n_iter_.max())>=max_iter:raise AssertionError('Source-fold classifier did not converge')
            p=model.predict_proba(scaler.transform(np.concatenate([b[n] for n in members],1)))
            probabilities[name][index]=p;models[name]=(scaler,model,members)
        retained[f'fold{fold}::indices']=index;fitted[fold]=(families,models)
        splits.append({'fold':fold,'held_users':held.tolist(),'train':at.tolist(),'validation':bt.tolist()})
        print(f'{manifest["dataset"]}: source OOF fold {fold+1}/3',flush=True)
    if np.any(coverage!=1):raise AssertionError('OOF coverage must be exactly once per source trial')
    temperatures={n:fit_temperature(p,y) for n,p in probabilities.items()}
    calibrated={n:temperature_probability(p,temperatures[n]) for n,p in probabilities.items()};rows=[]
    for name in specs:
        for kind,prob in (('raw_source_OOF',probabilities[name]),('temperature_fit_on_same_source_OOF',calibrated[name])):
            for user in (*users,'ALL'):
                mask=np.ones(len(y),bool) if user=='ALL' else u==user
                rows.append({'dataset':manifest['dataset'],'phase':'source_probability_calibration','subject':user,
                    'condition':'source_user_OOF','calibration_budget':0,'feature_family':'|'.join(specs[name]),'model':name,
                    'probability':kind,'method':f'{name}::{kind}',
                    'evaluation_boundary':'raw is held-user OOF; calibrated source scores reuse temperature fitting labels and are descriptive',**score(y[mask],prob[mask])})
    output.mkdir(parents=True)
    write(output/'feature_family_results.csv',rows)
    np.savez_compressed(output/'oof_predictions.npz',labels=y,users=u,trials=trials,**{f'raw_{n}':p for n,p in probabilities.items()},**{f'calibrated_{n}':p for n,p in calibrated.items()},**retained)
    (output/'fold_states.pkl').write_bytes(pickle.dumps(fitted))
    (output/'split_trial_ids.json').write_text(json.dumps(splits,indent=2))
    evidence={'dataset':manifest['dataset'],'phase':'source_probability_calibration','source_fit_run':source.name,
        'source_fit_sha256':hashlib.sha256((source/'fitted_states.pkl').read_bytes()).hexdigest(),
        'source_manifest_sha256':hashlib.sha256((source/'run_manifest.json').read_bytes()).hexdigest(),
        'source_users':users,'fit_trials':trials.tolist(),'families':list(manifest['families']),'specs':specs,
        'temperatures':temperatures,'target_data_opened':False,'seed':SEED,
        'preprocessing':'family/scaler/classifier refitted inside each source-user fold; no full-source family reuse in OOF',
        'limits':'temperature fit on source OOF labels; calibrated source metrics are tuning evidence, not outer-held calibration generalization'}
    (output/'run_manifest.json').write_text(json.dumps(evidence,indent=2))
    (output/'probability_calibration.json').write_text(json.dumps(evidence,indent=2))


def replay(source,output):
    manifest=json.loads((output/'run_manifest.json').read_text())
    for name,key in (('fitted_states.pkl','source_fit_sha256'),('run_manifest.json','source_manifest_sha256')):
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=manifest[key]:raise AssertionError('Changed source provenance')
    fitted=pickle.loads((output/'fold_states.pkl').read_bytes());error=0.;count=0
    with np.load(output/'oof_predictions.npz',allow_pickle=False) as z:
        for fold,(_,models) in fitted.items():
            index=z[f'fold{fold}::indices']
            for name,(scaler,model,members) in models.items():
                p=model.predict_proba(scaler.transform(np.concatenate([z[f'fold{fold}::{n}'] for n in members],1)))
                np.testing.assert_allclose(p,z[f'raw_{name}'][index],atol=1e-10,rtol=1e-9)
                error=max(error,float(np.abs(p-z[f'raw_{name}'][index]).max()));count+=1
        for name,temp in manifest['temperatures'].items():
            np.testing.assert_allclose(temp,fit_temperature(z[f'raw_{name}'],z['labels']),atol=1e-10,rtol=1e-9)
            np.testing.assert_allclose(z[f'calibrated_{name}'],temperature_probability(z[f'raw_{name}'],temp),atol=1e-10,rtol=1e-9)
    audit={'status':'ok','fold_probability_arrays_checked':count,'temperatures_checked':len(manifest['temperatures']),
        'max_absolute_probability_error':error,'source_hashes_checked':True,'target_data_opened':False}
    (output/'replay_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('source',type=Path);p.add_argument('output',type=Path);p.add_argument('--replay',action='store_true')
    a=p.parse_args();replay(a.source,a.output) if a.replay else run(a.raw,a.source,a.output)
