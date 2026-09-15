"""Concatenated-feature conditional Core comparisons using source-Ramp fits only."""
from pathlib import Path
import argparse
import hashlib
import json
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from emgimu.datasets.libemg_force import load_libemg_force_windows
from .force_full_fusion import IDS, aggregate
from .calibration_study import FROZEN_FAMILIES, CONDITIONS
from .screening import metrics, SEED
from .core_incremental_oof import write
from .unibo_temporal_complementarity import complementarity

CORE=FROZEN_FAMILIES
ADDED=('F3_Ring','F4_Spectral','F5_Temporal','F9_Quality')
SPECS={'Core':CORE,**{f'Core+{n}':CORE+(n,) for n in ADDED}}


def prepare(raw,old_source,output):
    if output.exists():raise FileExistsError(output)
    train=load_libemg_force_windows(raw,subjects=range(1,7),conditions=('Ramp',))
    previous=json.loads((old_source/'run_manifest.json').read_text())
    if set(train.trials)!=set(previous['source_trials']):raise AssertionError('Different family fitting trials')
    states,_=pickle.loads((old_source/'fitted_states.pkl').read_bytes());before=pickle.dumps(states)
    features={};dimensions={}
    for name in IDS:
        family,_,_=states[name]
        x,y,u,trials=aggregate(family.transform(train.batch),train)
        features[name]=x;dimensions[name]=x.shape[1]
    models={}
    for name,members in SPECS.items():
        x=np.concatenate([features[n] for n in members],1)
        scaler=StandardScaler().fit(x)
        model=LogisticRegression(C=1,class_weight='balanced',max_iter=2000,random_state=SEED).fit(scaler.transform(x),y)
        if int(model.n_iter_.max())>=2000:raise AssertionError('Core classifier did not converge')
        models[name]=(scaler,model,members)
        print(f'source fit {name}: {x.shape[1]} dimensions',flush=True)
    for name in ('F0',*ADDED):
        _,scaler,model=states[name];models[name]=(scaler,model,(name,))
    if pickle.dumps(states)!=before:raise AssertionError('Reused family states changed')
    output.mkdir(parents=True)
    (output/'fitted_states.pkl').write_bytes(pickle.dumps((states,models)))
    evidence={'dataset':'libemg_contraction_intensity','phase':'source_fit','source_users':list(range(1,7)),
        'source_condition':'Ramp','source_trials':trials.tolist(),'families':list(CORE),'specs':SPECS,
        'dimensions':dimensions,'classifier':'source-trial StandardScaler + balanced LogisticRegression C=1 max_iter=2000',
        'probabilities':'native logistic probabilities; no target temperature fit','source_family_run':old_source.name,
        'source_family_state_sha256':hashlib.sha256((old_source/'fitted_states.pkl').read_bytes()).hexdigest(),
        'seed':SEED,'target_users_opened':False,'core_selection':'existing frozen calibration_study.FROZEN_FAMILIES; no target-score selection'}
    (output/'run_manifest.json').write_text(json.dumps(evidence,indent=2))


def evaluate(raw,source,output,phase):
    if output.exists():raise FileExistsError(output)
    if phase not in ('validation','final'):raise ValueError('Independent validation/final phase required')
    manifest=json.loads((source/'run_manifest.json').read_text())
    if manifest['source_users']!=list(range(1,7)) or manifest['source_condition']!='Ramp':
        raise AssertionError('Requires source-only Ramp fits')
    users=(7,8) if phase=='validation' else (9,10)
    data=load_libemg_force_windows(raw,subjects=users,conditions=CONDITIONS)
    if set(data.trials)&set(manifest['source_trials']):raise AssertionError('Source/evaluation trial leakage')
    states,models=pickle.loads((source/'fitted_states.pkl').read_bytes());before=pickle.dumps((states,models))
    features={}
    for name in IDS:
        family,_,_=states[name]
        x,y,u,trials=aggregate(family.transform(data.batch),data);features[name]=x
    conditions=np.array([np.unique(data.conditions[data.trials==t]).item() for t in trials])
    probabilities={}
    for name,(scaler,model,members) in models.items():
        x=np.concatenate([features[n] for n in members],1)
        probabilities[name]=model.predict_proba(scaler.transform(x))
    if pickle.dumps((states,models))!=before:raise AssertionError('Frozen source fits changed')
    rows=[];increments=[];pairs=[]
    for user in (*users,'ALL'):
        for condition in ('ALL',*CONDITIONS):
            mask=(np.ones(len(y),bool) if user=='ALL' else u==user)&(np.ones(len(y),bool) if condition=='ALL' else conditions==condition)
            shared={'dataset':'libemg_contraction_intensity','phase':phase,'protocol':'source_fitted_concat_Core',
                'subject':user,'condition':condition,'calibration_budget':0,'aggregation':'pooled trials' if user=='ALL' else 'individual user'}
            scores={name:metrics(y[mask],p[mask],np.ones(mask.sum())) for name,p in probabilities.items()}
            for name,score in scores.items():
                members=models[name][2]
                rows.append({**shared,'feature_family':'|'.join(members),'model':name,
                    'feature_dimension':sum(manifest['dimensions'][n] for n in members),**score})
            for name in ADDED:
                base=scores['Core'];new=scores[f'Core+{name}']
                increments.append({**shared,'core_bank':'|'.join(CORE),'added_family':name,
                    'delta_logloss':base['log_loss']-new['log_loss'],'delta_brier':base['brier']-new['brier'],
                    'delta_macro_f1':new['macro_f1']-base['macro_f1']})
                pairs.append({**shared,'family_a':'Core','family_b':name,
                    **complementarity(y[mask],probabilities['Core'][mask],probabilities[name][mask],np.ones(mask.sum()))})
    output.mkdir(parents=True)
    for name,values in (('feature_family_results',rows),('conditional_incremental',increments),('error_complementarity',pairs)):
        write(output/f'{name}.csv',values)
    np.savez_compressed(output/'heldout_predictions.npz',labels=y,users=u,trials=trials,conditions=conditions,**probabilities,
        **{f'features_{n}':v for n,v in features.items()})
    (output/'split_trial_ids.json').write_text(json.dumps({'train':manifest['source_trials'],'calibration':[],'evaluation':trials.tolist()},indent=2))
    (output/'run_manifest.json').write_text(json.dumps({**manifest,'phase':phase,'target_users':list(users),
        'reused_source_run':source.name,'classifier_or_family_fit':False,'target_calibration':False,
        'source_fit_sha256':hashlib.sha256((source/'fitted_states.pkl').read_bytes()).hexdigest(),
        'source_manifest_sha256':hashlib.sha256((source/'run_manifest.json').read_bytes()).hexdigest(),
        'scope':'independent-user unseen-force, concatenated feature classifiers; source-selected Core frozen; no target selection; ALL pooled trials'},indent=2))
    print(json.dumps({'status':'ok','feature_rows':len(rows),'increment_rows':len(increments),'complementarity_rows':len(pairs)}))


def replay(source,output):
    manifest=json.loads((output/'run_manifest.json').read_text())
    for name,key in (('fitted_states.pkl','source_fit_sha256'),('run_manifest.json','source_manifest_sha256')):
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=manifest[key]:raise AssertionError('Changed source fit')
    _,models=pickle.loads((source/'fitted_states.pkl').read_bytes());error=0.
    with np.load(output/'heldout_predictions.npz',allow_pickle=False) as z:
        for name,(scaler,model,members) in models.items():
            x=np.concatenate([z[f'features_{n}'] for n in members],1)
            p=model.predict_proba(scaler.transform(x))
            np.testing.assert_allclose(p,z[name],atol=1e-10,rtol=1e-9)
            error=max(error,float(np.abs(p-z[name]).max()))
    audit={'status':'ok','source_hashes_checked':True,'classifier_probability_arrays_checked':len(models),
        'max_absolute_probability_error':error,'classifier_or_family_fit':False}
    (output/'replay_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--prepare',action='store_true');p.add_argument('--replay',action='store_true');p.add_argument('--phase',choices=('validation','final'))
    a=p.parse_args()
    if a.prepare:prepare(a.raw,a.source,a.output)
    elif a.replay:replay(a.source,a.output)
    else:evaluate(a.raw,a.source,a.output,a.phase)
