"""Source-session concatenated F0/SPD Core with temporal and real IMU additions."""
from pathlib import Path
import argparse
import hashlib
import json
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from emgimu.datasets.semg_manus import load_semg_manus_windows
from .manus_study import GESTURES, _aggregate, _metrics
from .manus_calibration import FROZEN_FAMILIES
from .screening import FAMILY_FACTORIES, SEED
from .core_incremental_oof import write
from .unibo_temporal_complementarity import complementarity

CORE=FROZEN_FAMILIES
ADDED=('F5_Temporal','F6_IMU')
SPECS={'Core':CORE,**{f'Core+{n}':CORE+(n,) for n in ADDED}}
USERS=(3,4,5,6,7,8)


def prepare(archive,old_source,output):
    if output.exists():raise FileExistsError(output)
    train=load_semg_manus_windows(archive,users=USERS,sessions=(1,),gestures=GESTURES)
    manifest=json.loads((old_source/'run_manifest.json').read_text())
    if manifest['dataset']!='semg_manus' or set(manifest['source_trials'])!=set(train.trials):
        raise AssertionError('Different source-session fitting trials')
    old,_=pickle.loads((old_source/'fitted_states.pkl').read_bytes());before=pickle.dumps(old)
    families={n:old[n][0] for n in ('F0',*ADDED)}
    families['F2c_SPD']=FAMILY_FACTORIES['F2c_SPD']().fit(train.batch,train.labels)
    features={};dimensions={}
    for name,family in families.items():
        x,y,u,s,speed,trials=_aggregate(family.transform(train.batch),train)
        features[name]=x;dimensions[name]=x.shape[1]
    models={}
    for name,members in SPECS.items():
        x=np.concatenate([features[n] for n in members],1);scaler=StandardScaler().fit(x)
        model=LogisticRegression(C=1,class_weight='balanced',max_iter=1000,random_state=SEED).fit(scaler.transform(x),y)
        if int(model.n_iter_.max())>=1000:raise AssertionError('Core classifier did not converge')
        models[name]=(scaler,model,members)
        print(f'source session1 {name}: {x.shape[1]} dimensions',flush=True)
    for name in ('F0',*ADDED):
        _,scaler,model=old[name];models[name]=(scaler,model,(name,))
    if pickle.dumps(old)!=before:raise AssertionError('Reused source families changed')
    output.mkdir(parents=True)
    (output/'fitted_states.pkl').write_bytes(pickle.dumps((families,models)))
    evidence={'dataset':'semg_manus','phase':'source_fit','source_users':list(USERS),'source_session':1,
        'source_trials':trials.tolist(),'families':list(CORE),'specs':SPECS,'dimensions':dimensions,'gestures':list(GESTURES),
        'source_family_run':old_source.name,'source_family_sha256':hashlib.sha256((old_source/'fitted_states.pkl').read_bytes()).hexdigest(),
        'classifier':'source-only trial StandardScaler and balanced logistic C=1 max_iter=1000',
        'probabilities':'native probabilities; no target temperature calibration','seed':SEED,'target_sessions_opened':False,
        'core_selection':'previous frozen manus_calibration.FROZEN_FAMILIES','scope':'six finger flexext classes; no rest; real IMU; source-session SPD reference fitted once'}
    (output/'run_manifest.json').write_text(json.dumps(evidence,indent=2))


def evaluate(archive,source,output,phase):
    if output.exists():raise FileExistsError(output)
    if phase not in ('validation','final'):raise ValueError('Independent target phase required')
    manifest=json.loads((source/'run_manifest.json').read_text())
    if manifest['source_session']!=1 or manifest['source_users']!=list(USERS):raise AssertionError('Source-session partition changed')
    session=2 if phase=='validation' else 3
    data=load_semg_manus_windows(archive,users=USERS,sessions=(session,),gestures=GESTURES)
    if set(data.trials)&set(manifest['source_trials']):raise AssertionError('Source/evaluation overlap')
    families,models=pickle.loads((source/'fitted_states.pkl').read_bytes());before=pickle.dumps((families,models));features={}
    for name,family in families.items():
        x,y,u,s,speed,trials=_aggregate(family.transform(data.batch),data);features[name]=x
    probabilities={name:model.predict_proba(scaler.transform(np.concatenate([features[n] for n in members],1)))
        for name,(scaler,model,members) in models.items()}
    if pickle.dumps((families,models))!=before:raise AssertionError('Target transformed source fits')
    rows=[];increments=[];pairs=[]
    for user in (*USERS,'ALL'):
        for condition in ('ALL','slow','medium','fast'):
            mask=(np.ones(len(y),bool) if user=='ALL' else u==user)&(np.ones(len(y),bool) if condition=='ALL' else speed==condition)
            shared={'dataset':'semg_manus','phase':phase,'protocol':'source_session_concat_Core','subject':user,
                'session/domain':f'session_{session}','condition':condition,'calibration_budget':0,'aggregation':'pooled trials' if user=='ALL' else 'individual user'}
            scores={name:_metrics(y[mask],p[mask]) for name,p in probabilities.items()}
            for name,score in scores.items():
                rows.append({**shared,'feature_family':'|'.join(models[name][2]),'model':name,
                    'feature_dimension':sum(manifest['dimensions'][n] for n in models[name][2]),**score})
            for name in ADDED:
                a=scores['Core'];b=scores[f'Core+{name}']
                increments.append({**shared,'core_bank':'|'.join(CORE),'added_family':name,
                    'delta_logloss':a['log_loss']-b['log_loss'],'delta_brier':a['brier']-b['brier'],'delta_macro_f1':b['macro_f1']-a['macro_f1']})
                pairs.append({**shared,'family_a':'Core','family_b':name,
                    **complementarity(y[mask],probabilities['Core'][mask],probabilities[name][mask],np.ones(mask.sum()))})
    output.mkdir(parents=True)
    for name,values in (('feature_family_results',rows),('conditional_incremental',increments),('error_complementarity',pairs)):
        write(output/f'{name}.csv',values)
    np.savez_compressed(output/'heldout_predictions.npz',labels=y,users=u,trials=trials,sessions=s,speeds=speed,
        **probabilities,**{f'features_{n}':x for n,x in features.items()})
    (output/'split_trial_ids.json').write_text(json.dumps({'train':manifest['source_trials'],'calibration':[],'evaluation':trials.tolist()},indent=2))
    (output/'run_manifest.json').write_text(json.dumps({**manifest,'phase':phase,'target_session':session,
        'source_fit_target_sessions_opened':False,'target_sessions_opened':True,
        'source_fit_run':source.name,'source_fit_sha256':hashlib.sha256((source/'fitted_states.pkl').read_bytes()).hexdigest(),
        'classifier_or_family_fit':False,'target_calibration':False,
        'limits':'same users across sessions; session and speed changes confounded; ALL pooled trial metrics; no rest/open/pinch task or hardware transfer claim'},indent=2))
    print(json.dumps({'status':'ok','models':len(models),'target_trials':len(y),'score_rows':len(rows),'increment_rows':len(increments)}))


def replay(source,output):
    manifest=json.loads((output/'run_manifest.json').read_text())
    if hashlib.sha256((source/'fitted_states.pkl').read_bytes()).hexdigest()!=manifest['source_fit_sha256']:raise AssertionError('Changed source fit')
    _,models=pickle.loads((source/'fitted_states.pkl').read_bytes());error=0.
    with np.load(output/'heldout_predictions.npz',allow_pickle=False) as z:
        for name,(scaler,model,members) in models.items():
            p=model.predict_proba(scaler.transform(np.concatenate([z[f'features_{n}'] for n in members],1)))
            np.testing.assert_allclose(p,z[name],atol=1e-10,rtol=1e-9);error=max(error,float(np.abs(p-z[name]).max()))
    audit={'status':'ok','models_checked':len(models),'max_absolute_probability_error':error,'source_hash_checked':True}
    (output/'replay_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--prepare',action='store_true');p.add_argument('--replay',action='store_true');p.add_argument('--phase',choices=('validation','final'))
    a=p.parse_args()
    if a.prepare:prepare(a.archive,a.source,a.output)
    elif a.replay:replay(a.source,a.output)
    else:evaluate(a.archive,a.source,a.output,a.phase)
