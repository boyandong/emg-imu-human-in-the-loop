"""Source-only calibrated log-band classifiers and calibration-relative spectral controls."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from emgimu.datasets.epn612 import load_epn612_windows
from .epn_study import aggregate_trials,_metrics
from .relative_spectrum import LogBandEnergyFamily,RelativeSpectrumCoordinates
from .force_nested_oof import fit_temperature,temperature_probability
from .screening import SEED


def chosen(y,u,user,shots):
    rng=np.random.default_rng(SEED+int(user));out=[]
    for h in sorted(set(y)):out.extend(rng.permutation(np.flatnonzero((u==user)&(y==h)))[:shots])
    return np.array(out,dtype=int)


def fit_models(x,y,u,users):
    references={};centered=x.copy()
    for user in users:
        cal=chosen(y,u,user,5);ref=RelativeSpectrumCoordinates().fit_calibration(x[cal]);references[user]=ref
        mask=u==user;centered[mask]=ref.transform(x[mask])
    models={}
    for name,features in (('raw_log_bands',x),('calibration_relative_log_bands',centered)):
        scaler=StandardScaler().fit(features)
        model=LogisticRegression(C=1,class_weight='balanced',max_iter=2000,random_state=SEED).fit(scaler.transform(features),y)
        models[name]=(scaler,model)
    return models,references


def prepare(archive,output):
    if output.exists():raise FileExistsError(output)
    data=load_epn612_windows(archive,users=range(1,16));family=LogBandEnergyFamily().fit(data.batch)
    x,y,u,trials,_=aggregate_trials(family.transform(data.batch),data)
    models,references=fit_models(x,y,u,range(1,16));oof={n:[] for n in models};labels=[];held_trials=[];splits=[];fold_states={}
    for fold,held_users in enumerate((range(1,6),range(6,11),range(11,16))):
        print(f'source fold {fold}: log-band probability calibration',flush=True)
        train_users=sorted(set(u)-set(held_users));train_idx=np.flatnonzero(np.isin(data.users,train_users))
        fold_family=LogBandEnergyFamily().fit(data.batch.take(train_idx))
        fold_x=aggregate_trials(fold_family.transform(data.batch),data)[0]
        train=np.isin(u,train_users);m,refs=fit_models(fold_x[train],y[train],u[train],train_users)
        cal=[];ev=[];relative=[]
        for user in held_users:
            c=chosen(y,u,user,5);e=np.flatnonzero((u==user)&~np.isin(np.arange(len(y)),c))
            ref=RelativeSpectrumCoordinates().fit_calibration(fold_x[c]);refs[user]=ref
            cal.extend(c);ev.extend(e);relative.append(ref.transform(fold_x[e]))
        ev=np.array(ev);cal=np.array(cal)
        if set(trials[cal])&set(trials[ev]):raise AssertionError('Source calibration leakage')
        for name,features in (('raw_log_bands',fold_x[ev]),('calibration_relative_log_bands',np.concatenate(relative))):
            scaler,model=m[name];oof[name].append(model.predict_proba(scaler.transform(features)))
        labels.append(y[ev]);held_trials.append(trials[ev]);fold_states[fold]=(fold_family,m,refs)
        splits.append({'fold':fold,'train':trials[train].tolist(),'calibration':trials[cal].tolist(),'evaluation':trials[ev].tolist()})
    labels=np.concatenate(labels);temps={n:fit_temperature(np.concatenate(p),labels) for n,p in oof.items()}
    population=RelativeSpectrumCoordinates().fit_calibration(np.stack([r.reference_ for r in references.values()]))
    output.mkdir(parents=True)
    with (output/'source_fitted_states.pkl').open('wb') as h:pickle.dump((family,models,population,references),h)
    with (output/'source_oof_states.pkl').open('wb') as h:pickle.dump(fold_states,h)
    np.savez_compressed(output/'source_oof_predictions.npz',labels=labels,trials=np.concatenate(held_trials),**{n:np.concatenate(p) for n,p in oof.items()})
    (output/'split_trial_ids.json').write_text(json.dumps(splits,indent=2))
    (output/'run_manifest.json').write_text(json.dumps({'dataset':'epn612','phase':'source','source_users':list(range(1,16)),
        'source_trials':trials.tolist(),'temperatures':temps,'dimension':32,'band_count':4,'source_reference_shots_per_class':5,
        'target_data_opened':False,'source_cv':'three disjoint user folds; simulated cal5 trials wholly excluded from OOF evaluation',
        'rule_selection':'fixed formula and logistic C=1; no target-score selection'},indent=2))


def run(archive,source,output,phase):
    if output.exists():raise FileExistsError(output)
    manifest=json.loads((source/'run_manifest.json').read_text());family,models,population,_=pickle.loads((source/'source_fitted_states.pkl').read_bytes())
    before=pickle.dumps((family,models,population));users=(16,17,18) if phase=='validation' else (19,20,21)
    target=load_epn612_windows(archive,users=users);x,y,u,trials,_=aggregate_trials(family.transform(target.batch),target)
    rows=[];saved={};splits=[];references={}
    for user in users:
        for shots in (0,1,2,5):
            cal=chosen(y,u,user,shots);ev=np.flatnonzero((u==user)&~np.isin(np.arange(len(y)),cal))
            if set(trials[cal])&set(trials[ev]) or set(trials[cal])&set(manifest['source_trials']):raise AssertionError('Target calibration leakage')
            reference=RelativeSpectrumCoordinates().fit_calibration(x[cal]) if shots else population
            references[(user,shots)]=reference
            for name,features in (('raw_log_bands',x[ev]),('calibration_relative_log_bands',reference.transform(x[ev]))):
                scaler,model=models[name];p=temperature_probability(model.predict_proba(scaler.transform(features)),manifest['temperatures'][name])
                rows.append({'dataset':'epn612','phase':phase,'subject':user,'condition':'cross_user','shots_per_class':shots,
                    'feature_bank':name,'method':name,'feature_dimension':32,**_metrics(y[ev],p,np.ones(len(ev)))})
                saved[f'{user}_{shots}_{name}']=p
            saved[f'{user}_{shots}_features']=x[ev];saved[f'{user}_{shots}_labels']=y[ev];saved[f'{user}_{shots}_trials']=trials[ev]
            splits.append({'user':user,'shots':shots,'calibration':trials[cal].tolist(),'evaluation':trials[ev].tolist()})
    for shots in (0,1,2,5):
        for name in models:
            group=[r for r in rows if r['shots_per_class']==shots and r['method']==name]
            result={k:float(np.mean([r[k] for r in group])) for k in ('macro_f1','accuracy','log_loss','brier','ece')}
            classes=[json.loads(r['per_class_f1_json']) for r in group]
            result['per_class_f1_json']=json.dumps({h:float(np.mean([r[h] for r in classes])) for h in classes[0]})
            rows.append({**group[0],'subject':'ALL',**result})
    if pickle.dumps((family,models,population))!=before:raise AssertionError('Target evaluation mutated source states')
    output.mkdir(parents=True)
    with (output/'calibration_curve.csv').open('w',newline='',encoding='utf-8') as h:
        writer=csv.DictWriter(h,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (output/'calibration_references.pkl').open('wb') as h:pickle.dump(references,h)
    np.savez_compressed(output/'heldout_predictions.npz',**saved)
    (output/'split_trial_ids.json').write_text(json.dumps(splits,indent=2))
    (output/'run_manifest.json').write_text(json.dumps({'dataset':'epn612','phase':phase,'source_package':str(source),
        'target_users':users,'budgets':[0,1,2,5],'dimension':32,'classifier_or_family_fit':False,
        'scope':'raw and relative spectral observation controls; source OOF probability temperature; target references from whole calibration trials only',
        'limitations':'relative baseline is signal context, not fatigue; source probability calibration simulates cal5; cal0 uses source population reference'},indent=2))
    print(json.dumps({'status':'ok','phase':phase,'rows':len(rows)}))


def replay_source(archive,source):
    manifest=json.loads((source/'run_manifest.json').read_text())
    data=load_epn612_windows(archive,users=manifest['source_users'])
    states=pickle.loads((source/'source_oof_states.pkl').read_bytes());before=pickle.dumps(states)
    splits=json.loads((source/'split_trial_ids.json').read_text());all_p={};all_y=[];all_trials=[];error=0.;checked=0
    for split in splits:
        if any(set(split[a])&set(split[b]) for a,b in (('train','calibration'),('train','evaluation'),('calibration','evaluation'))):
            raise AssertionError('Source OOF partition overlap')
        family,models,refs=states[split['fold']]
        x,y,u,trials,_=aggregate_trials(family.transform(data.batch),data)
        if set(split['train'])|set(split['calibration'])|set(split['evaluation'])!=set(trials):raise AssertionError('Source fold coverage mismatch')
        ev=np.array([np.flatnonzero(trials==t).item() for t in split['evaluation']])
        centered=x[ev].copy()
        for user in sorted(set(u[ev])):
            mask=u[ev]==user;cal=chosen(y,u,user,5)
            np.testing.assert_allclose(refs[user].reference_,x[cal].mean(0),atol=1e-6,rtol=1e-6)
            if not set(trials[cal])<=set(split['calibration']):raise AssertionError('Reference trial mismatch')
            centered[mask]=refs[user].transform(x[ev][mask])
        for name,features in (('raw_log_bands',x[ev]),('calibration_relative_log_bands',centered)):
            scaler,model=models[name];all_p.setdefault(name,[]).append(model.predict_proba(scaler.transform(features)));checked+=1
        all_y.append(y[ev]);all_trials.append(trials[ev])
    with np.load(source/'source_oof_predictions.npz',allow_pickle=False) as saved:
        np.testing.assert_array_equal(np.concatenate(all_y),saved['labels']);np.testing.assert_array_equal(np.concatenate(all_trials),saved['trials'])
        for name,ps in all_p.items():
            p=np.concatenate(ps);np.testing.assert_allclose(p,saved[name],atol=1e-10,rtol=1e-9)
            error=max(error,float(np.abs(p-saved[name]).max()))
            np.testing.assert_allclose(fit_temperature(p,saved['labels']),manifest['temperatures'][name],rtol=1e-8,atol=1e-8)
    if pickle.dumps(states)!=before:raise AssertionError('Source replay mutated state')
    audit={'status':'ok','source_oof_probability_blocks_checked':checked,'max_absolute_probability_error':error,
        'source_temperatures_recomputed':True,'classifier_or_family_fit':False,'target_data_opened':False,
        'source_state_sha256':hashlib.sha256((source/'source_fitted_states.pkl').read_bytes()).hexdigest(),
        'source_oof_sha256':hashlib.sha256((source/'source_oof_predictions.npz').read_bytes()).hexdigest()}
    (source/'replay_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit))


def replay(source,output):
    manifest=json.loads((source/'run_manifest.json').read_text());_,models,_,_=pickle.loads((source/'source_fitted_states.pkl').read_bytes())
    references=pickle.loads((output/'calibration_references.pkl').read_bytes());splits=json.loads((output/'split_trial_ids.json').read_text());checked=0;error=0.
    with np.load(output/'heldout_predictions.npz',allow_pickle=False) as saved:
        for split in splits:
            user=split['user'];shots=split['shots'];prefix=f'{user}_{shots}';x=saved[f'{prefix}_features']
            np.testing.assert_array_equal(saved[f'{prefix}_trials'],np.sort(split['evaluation']))
            for name,features in (('raw_log_bands',x),('calibration_relative_log_bands',references[(user,shots)].transform(x))):
                scaler,model=models[name];p=temperature_probability(model.predict_proba(scaler.transform(features)),manifest['temperatures'][name])
                np.testing.assert_allclose(p,saved[f'{prefix}_{name}'],atol=1e-10,rtol=1e-9)
                checked+=1;error=max(error,float(np.abs(p-saved[f'{prefix}_{name}']).max()))
    audit={'status':'ok','prediction_arrays_checked':checked,'max_absolute_probability_error':error,
        'classifier_or_family_fit':False,'scope':'all heldout spectral predictions rebuilt from saved trial features, source classifiers and frozen calibration references'}
    (output/'replay_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('source',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--phase',choices=('validation','final'));p.add_argument('--prepare',action='store_true');p.add_argument('--replay',action='store_true')
    p.add_argument('--replay-source',action='store_true')
    a=p.parse_args();prepare(a.archive,a.source) if a.prepare else replay_source(a.archive,a.source) if a.replay_source else replay(a.source,a.output) if a.replay else run(a.archive,a.source,a.output,a.phase)
