"""Whole-trial own-session anchors for frozen MANUS concatenated Core models."""
from pathlib import Path
import argparse
import hashlib
import json
import pickle
import numpy as np
from .manus_core_incremental import CORE, ADDED, USERS
from .manus_study import _metrics
from .calibration import PersonalAnchor
from .screening import SEED
from .core_incremental_oof import write
from .unibo_temporal_complementarity import complementarity

METRICS=('macro_f1','accuracy','log_loss','brier','ece')


def split(y,u,trials,user,shots):
    if shots not in (0,1,2):raise ValueError('Unsupported budget; one whole trial/class must remain')
    if len(set(trials))!=len(trials):raise AssertionError('Cache must contain unique whole trials')
    rng=np.random.default_rng(SEED+user);chosen=[]
    for label in range(6):
        index=np.flatnonzero((u==user)&(y==label))
        if len(index)<=shots:raise AssertionError('No held-out trial remains for a class')
        chosen.extend(rng.permutation(index)[:shots])
    cal=np.asarray(chosen,dtype=int);ev=np.flatnonzero((u==user)&~np.isin(np.arange(len(y)),cal))
    if set(trials[cal])&set(trials[ev]):raise AssertionError('Calibration/evaluation overlap')
    return cal,ev


def run(source,target,output):
    if output.exists():raise FileExistsError(output)
    manifest=json.loads((target/'run_manifest.json').read_text())
    if manifest['source_session']!=1 or manifest['target_session'] not in (2,3):raise AssertionError('Wrong session partition')
    if hashlib.sha256((source/'fitted_states.pkl').read_bytes()).hexdigest()!=manifest['source_fit_sha256']:raise AssertionError('Changed source fits')
    families,models=pickle.loads((source/'fitted_states.pkl').read_bytes());before=pickle.dumps((families,models))
    rows=[];increments=[];pairs=[];splits=[];saved={};anchors={}
    with np.load(target/'heldout_predictions.npz',allow_pickle=False) as z:
        y=z['labels'];u=z['users'];trials=z['trials'];speed=z['speeds']
        if set(trials)&set(manifest['source_trials']) or set(u)!=set(USERS):raise AssertionError('Source/target trial or user mismatch')
        for user in USERS:
            for shots in (0,1,2):
                cal,ev=split(y,u,trials,user,shots)
                branches={'without_anchor':{},'with_anchor':{}}
                for name,(scaler,model,members) in models.items():
                    base=z[name][ev];branches['without_anchor'][name]=base
                    p=base
                    if shots:
                        x=scaler.transform(np.concatenate([z[f'features_{n}'] for n in members],1))
                        anchor=PersonalAnchor(metric='standardized_euclidean').fit(x[cal],y[cal])
                        temperature=max(float(np.median(anchor.transform(x[cal])[:,:6])),1e-10)
                        logits=-anchor.transform(x[ev])[:,:6].astype(float)/temperature
                        logits-=logits.max(1,keepdims=True);q=np.exp(logits);q/=q.sum(1,keepdims=True)
                        alpha=shots/(shots+2);p=(1-alpha)*base+alpha*q
                        anchors[(user,shots,name)]=(anchor,temperature)
                        saved[f'{user}_{shots}_{name}::scaled']=x[ev]
                    branches['with_anchor'][name]=p
                for branch,probabilities in branches.items():
                    for name,p in probabilities.items():saved[f'{user}_{shots}_{name}::{branch}']=p
                    for condition in ('ALL','slow','medium','fast'):
                        mask=np.ones(len(ev),bool) if condition=='ALL' else speed[ev]==condition
                        shared={'dataset':'semg_manus','phase':manifest['phase'],'protocol':'whole_session_trial_concat_Core_anchor',
                            'subject':user,'session/domain':f"session_{manifest['target_session']}",'condition':condition,
                            'shots_per_class':shots,'method':branch,'calibration_trials':6*shots,'evaluation_trials':int(mask.sum()),
                            'classes_present':len(set(y[ev][mask])),'aggregation':'individual user; trial means'}
                        scores={n:_metrics(y[ev][mask],p[mask]) if mask.any() else {**{m:'' for m in METRICS},'per_class_f1_json':''} for n,p in probabilities.items()}
                        for name,score in scores.items():rows.append({**shared,'model':name,'feature_bank':'|'.join(models[name][2]),**score})
                        if not mask.any():continue
                        for name in ADDED:
                            a=scores['Core'];b=scores[f'Core+{name}']
                            increments.append({**shared,'core_bank':'|'.join(CORE),'added_family':name,
                                'delta_logloss':a['log_loss']-b['log_loss'],'delta_brier':a['brier']-b['brier'],'delta_macro_f1':b['macro_f1']-a['macro_f1']})
                            pairs.append({**shared,'family_a':'Core','family_b':name,
                                **complementarity(y[ev][mask],probabilities['Core'][mask],probabilities[name][mask],np.ones(mask.sum()))})
                saved[f'{user}_{shots}::indices']=ev
                splits.append({'user':user,'shots':shots,'source':manifest['source_trials'],'calibration':trials[cal].tolist(),'evaluation':trials[ev].tolist()})
        for shots in (0,1,2):
            for branch in ('without_anchor','with_anchor'):
                for condition in ('ALL','slow','medium','fast'):
                    for name in models:
                        subset=[r for r in rows if r['subject'] in USERS and r['shots_per_class']==shots and r['method']==branch and r['condition']==condition and r['model']==name]
                        available=[r for r in subset if r['evaluation_trials']]
                        rows.append({**subset[0],'subject':'ALL','evaluation_trials':sum(r['evaluation_trials'] for r in available),
                            'classes_present':'N/A','aggregation':f'mean over {len(available)} available users',
                            **{m:float(np.mean([r[m] for r in available])) if available else '' for m in METRICS},'per_class_f1_json':'mean per-user aggregation'})
                    for name in ADDED:
                        subset=[r for r in increments if r['subject'] in USERS and r['shots_per_class']==shots and r['method']==branch and r['condition']==condition and r['added_family']==name]
                        if subset:increments.append({**subset[0],'subject':'ALL','evaluation_trials':sum(r['evaluation_trials'] for r in subset),
                            'classes_present':'N/A','aggregation':f'mean over {len(subset)} available users',
                            **{m:float(np.mean([r[m] for r in subset])) for m in ('delta_logloss','delta_brier','delta_macro_f1')}})
        for user in (*USERS,'ALL'):
            for name in models:
                r=next(r for r in rows if r['subject']==user and r['model']==name and r['condition']=='ALL' and r['shots_per_class']==0)
                rows.append({**r,'shots_per_class':5,'method':'unsupported','calibration_trials':'','evaluation_trials':0,
                    **{m:'' for m in (*METRICS,'per_class_f1_json')}})
    if pickle.dumps((families,models))!=before:raise AssertionError('Source state changed')
    output.mkdir(parents=True)
    for name,values in (('calibration_curve',rows),('conditional_incremental',increments),('error_complementarity',pairs)):write(output/f'{name}.csv',values)
    np.savez_compressed(output/'calibration_predictions.npz',labels=y,users=u,trials=trials,**saved)
    (output/'anchors.pkl').write_bytes(pickle.dumps(anchors))
    (output/'split_trial_ids.json').write_text(json.dumps(splits,indent=2))
    (output/'run_manifest.json').write_text(json.dumps({**manifest,'source_fit_run':source.name,'target_prediction_run':target.name,
        'target_prediction_sha256':hashlib.sha256((target/'heldout_predictions.npz').read_bytes()).hexdigest(),
        'budgets':[0,1,2],'unsupported_5':'three complete trials/class/session; retain one for evaluation',
        'anchor':'calibration-only mean/MAD and median-distance temperature; fixed alpha shots/(shots+2)',
        'personal_anchor_fit':True,'target_calibration':True,'classifier_or_family_fit':False,
        'limits':'different evaluation trials per budget; matched without-anchor uses exactly same remaining trials; sparse speed cells may omit truth classes or users; no final-score selection'},indent=2))
    print(json.dumps({'status':'ok','curve_rows':len(rows),'increment_rows':len(increments),'pair_rows':len(pairs)}))


def replay(source,target,output):
    manifest=json.loads((output/'run_manifest.json').read_text())
    if hashlib.sha256((source/'fitted_states.pkl').read_bytes()).hexdigest()!=manifest['source_fit_sha256'] or hashlib.sha256((target/'heldout_predictions.npz').read_bytes()).hexdigest()!=manifest['target_prediction_sha256']:raise AssertionError('Changed provenance')
    anchors=pickle.loads((output/'anchors.pkl').read_bytes());count=0;error=0.
    with np.load(output/'calibration_predictions.npz',allow_pickle=False) as z,np.load(target/'heldout_predictions.npz',allow_pickle=False) as original:
        for key in ('labels','users','trials'):np.testing.assert_array_equal(z[key],original[key])
        for user in USERS:
            for shots in (0,1,2):
                _,ev=split(original['labels'],original['users'],original['trials'],user,shots)
                np.testing.assert_array_equal(ev,z[f'{user}_{shots}::indices'])
                for name in ('Core',*(f'Core+{n}' for n in ADDED),'F0',*ADDED):
                    key=f'{user}_{shots}_{name}';p=original[name][ev]
                    np.testing.assert_array_equal(p,z[f'{key}::without_anchor'])
                    if shots:
                        anchor,temperature=anchors[(user,shots,name)]
                        logits=-anchor.transform(z[f'{key}::scaled'])[:,:6].astype(float)/temperature
                        logits-=logits.max(1,keepdims=True);q=np.exp(logits);q/=q.sum(1,keepdims=True)
                        alpha=shots/(shots+2);p=(1-alpha)*p+alpha*q
                    np.testing.assert_allclose(p,z[f'{key}::with_anchor'],atol=1e-10,rtol=1e-9);error=max(error,float(np.abs(p-z[f'{key}::with_anchor']).max()));count+=1
    audit={'status':'ok','probability_arrays_checked':count,'matched_controls_checked':count,'max_absolute_probability_error':error,'source_target_hashes_checked':True}
    (output/'replay_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('target',type=Path);p.add_argument('output',type=Path);p.add_argument('--replay',action='store_true')
    a=p.parse_args();replay(a.source,a.target,a.output) if a.replay else run(a.source,a.target,a.output)
