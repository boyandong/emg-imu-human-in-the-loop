"""Ramp-only personal anchors for frozen concatenated-feature force Core models."""
from pathlib import Path
import argparse
import hashlib
import json
import pickle
import numpy as np
from emgimu.datasets.libemg_force import load_libemg_force_windows
from .force_full_fusion import IDS, aggregate
from .force_core_incremental import CORE, ADDED
from .calibration import PersonalAnchor
from .calibration_study import CONDITIONS
from .screening import metrics, SEED
from .core_incremental_oof import write
from .unibo_temporal_complementarity import complementarity
from .core_probability_oof import load_temperatures, apply_temperature, verify_calibration_audit


def choose(cy,cu,trials,user,shots):
    if shots not in (0,1,2):raise ValueError('Only supported frozen 0/1/2-shot budgets')
    rng=np.random.default_rng(SEED+user);chosen=[]
    for label in range(7):
        options=np.flatnonzero((cu==user)&(cy==label))
        if len(options)<shots:raise AssertionError('Insufficient whole calibration trials')
        chosen.extend(rng.permutation(options)[:shots])
    index=np.asarray(chosen,dtype=int)
    if len(set(trials[index]))!=7*shots:raise AssertionError('Repeated calibration trials')
    return index


def add_unsupported(rows):
    """Expose the unsupported 5-shot budget without inventing scores."""
    for user in dict.fromkeys(r['subject'] for r in rows):
        for name in dict.fromkeys(r['model'] for r in rows):
            template=next(r for r in rows if r['subject']==user and r['model']==name and r['condition']=='ALL' and str(r['shots_per_class'])=='0')
            rows.append({**template,'shots_per_class':5,'method':'unsupported_5_shot','calibration_trials':'',
                **{m:'' for m in ('macro_f1','accuracy','log_loss','brier','ece','per_class_f1_json')}})
    return rows


def run(raw,source,target_run,output,probability_source=None):
    if output.exists():raise FileExistsError(output)
    manifest=json.loads((target_run/'run_manifest.json').read_text())
    if manifest['phase'] not in ('validation','final') or manifest['source_users']!=list(range(1,7)):
        raise AssertionError('Requires frozen independent target run')
    users=(7,8) if manifest['phase']=='validation' else (9,10)
    if manifest['target_users']!=list(users):raise AssertionError('Target user partition changed')
    if hashlib.sha256((source/'fitted_states.pkl').read_bytes()).hexdigest()!=manifest['source_fit_sha256']:
        raise AssertionError('Source classifier changed')
    states,models=pickle.loads((source/'fitted_states.pkl').read_bytes());before=pickle.dumps((states,models))
    if probability_source:
        _,audit=load_temperatures(source,probability_source,models)
        manifest={**manifest,'probability_calibration':audit,'probabilities':'source-user OOF temperatures before frozen anchor mixing'}
    calibration=load_libemg_force_windows(raw,subjects=users,conditions=('Ramp',))
    features={}
    for name in IDS:
        family,_,_=states[name]
        x,cy,cu,cal_trials=aggregate(family.transform(calibration.batch),calibration);features[name]=x
    rows=[];increments=[];pairs=[];splits=[];anchors={};saved={};inputs={}
    with np.load(target_run/'heldout_predictions.npz',allow_pickle=False) as z:
        y=z['labels'];u=z['users'];trials=z['trials'];conditions=z['conditions']
        if set(cal_trials)&(set(trials)|set(manifest['source_trials'])):
            raise AssertionError('Calibration overlaps population fitting or unseen-force evaluation')
        for user in users:
            ev=u==user
            for shots in (0,1,2):
                cal=choose(cy,cu,cal_trials,user,shots)
                branches={'without_anchor':{},'with_anchor':{}}
                for name,(scaler,model,members) in models.items():
                    original=apply_temperature(z[name][ev],name,manifest);branches['without_anchor'][name]=original
                    if shots:
                        cx=scaler.transform(np.concatenate([features[n][cal] for n in members],1))
                        tx=scaler.transform(np.concatenate([z[f'features_{n}'][ev] for n in members],1))
                        anchor=PersonalAnchor(metric='standardized_euclidean').fit(cx,cy[cal])
                        if set(anchor.classes_)!=set(range(7)):raise AssertionError('Anchor class coverage')
                        scale=max(float(np.median(anchor.transform(cx)[:,:7])),1e-10)
                        logits=-anchor.transform(tx)[:,:7].astype(float)/scale
                        logits-=logits.max(1,keepdims=True);p=np.exp(logits);p/=p.sum(1,keepdims=True)
                        alpha=shots/(shots+2)
                        branches['with_anchor'][name]=(1-alpha)*original+alpha*p
                        anchors[(user,shots,name)]=(anchor,scale)
                        inputs[f'{user}_{shots}_{name}::target_scaled']=tx
                        inputs[f'{user}_{shots}_{name}::population']=original
                    else:branches['with_anchor'][name]=original
                for branch,probabilities in branches.items():
                    for condition in ('ALL',*CONDITIONS):
                        mask=np.ones(ev.sum(),bool) if condition=='ALL' else conditions[ev]==condition
                        shared={'dataset':'libemg_contraction_intensity','phase':manifest['phase'],'protocol':'Ramp_only_concat_Core_anchor',
                            'subject':user,'condition':condition,'shots_per_class':shots,'method':branch,
                            'calibration_force':'Ramp','target_force_calibration':False,'calibration_trials':7*shots}
                        scores={name:metrics(y[ev][mask],p[mask],np.ones(mask.sum())) for name,p in probabilities.items()}
                        for name,score in scores.items():
                            rows.append({**shared,'feature_bank':'|'.join(models[name][2]),'model':name,
                                'feature_dimension':sum(manifest['dimensions'][n] for n in models[name][2]),**score})
                        for name in ADDED:
                            base=scores['Core'];new=scores[f'Core+{name}']
                            increments.append({**shared,'core_bank':'|'.join(CORE),'added_family':name,
                                'delta_logloss':base['log_loss']-new['log_loss'],'delta_brier':base['brier']-new['brier'],
                                'delta_macro_f1':new['macro_f1']-base['macro_f1']})
                            pairs.append({**shared,'family_a':'Core','family_b':name,
                                **complementarity(y[ev][mask],probabilities['Core'][mask],probabilities[name][mask],np.ones(mask.sum()))})
                    for name,p in probabilities.items():saved[f'{user}_{shots}_{name}::{branch}']=p
                splits.append({'user':user,'shots':shots,'source':manifest['source_trials'],'calibration':cal_trials[cal].tolist(),'evaluation':trials[ev].tolist()})
        # ALL metrics explicitly average per-user scores, matching calibrated product comparisons.
        for shots in (0,1,2):
            for branch in ('without_anchor','with_anchor'):
                for condition in ('ALL',*CONDITIONS):
                    for name in models:
                        subset=[r for r in rows if r['subject'] in users and r['shots_per_class']==shots and r['method']==branch and r['condition']==condition and r['model']==name]
                        rows.append({**subset[0],'subject':'ALL',**{m:float(np.mean([r[m] for r in subset])) for m in ('macro_f1','accuracy','log_loss','brier','ece')},'per_class_f1_json':'mean per-user aggregation'})
                    for name in ADDED:
                        subset=[r for r in increments if r['subject'] in users and r['shots_per_class']==shots and r['method']==branch and r['condition']==condition and r['added_family']==name]
                        increments.append({**subset[0],'subject':'ALL',**{m:float(np.mean([r[m] for r in subset])) for m in ('delta_logloss','delta_brier','delta_macro_f1')}})
        if pickle.dumps((states,models))!=before:raise AssertionError('Source states changed during calibration')
    output.mkdir(parents=True)
    add_unsupported(rows)
    for name,values in (('calibration_curve',rows),('conditional_incremental',increments),('error_complementarity',pairs)):
        write(output/f'{name}.csv',values)
    np.savez_compressed(output/'calibration_predictions.npz',labels=y,users=u,trials=trials,conditions=conditions,**saved,**inputs)
    (output/'anchors.pkl').write_bytes(pickle.dumps(anchors))
    (output/'split_trial_ids.json').write_text(json.dumps(splits,indent=2))
    evidence={**manifest,'protocol':'Ramp_only_concat_Core_anchor','source_fit_run':source.name,'target_prediction_run':target_run.name,
        'target_prediction_sha256':hashlib.sha256((target_run/'heldout_predictions.npz').read_bytes()).hexdigest(),
        'target_calibration':'whole own-user Ramp trials only','budgets':[0,1,2],'unsupported_5':'only four Ramp trials/class',
        'anchor':'mean prototypes; calibration MAD distances; calibration-distance median temperature; alpha shots/(shots+2)',
        'aggregation':'ALL mean per-user metrics; individual native trial means; same unseen-force evaluation trials for all budgets',
        'classifier_or_family_fit':False,'personal_anchor_fit':True,'rule_selection':'fixed existing anchor protocol; no target-score selection'}
    (output/'run_manifest.json').write_text(json.dumps(evidence,indent=2));print(json.dumps({'status':'ok','curve_rows':len(rows),'increments':len(increments),'pairs':len(pairs)}))


def replay(source,target_run,output):
    manifest=json.loads((output/'run_manifest.json').read_text())
    verify_calibration_audit(manifest)
    if hashlib.sha256((source/'fitted_states.pkl').read_bytes()).hexdigest()!=manifest['source_fit_sha256'] or hashlib.sha256((target_run/'heldout_predictions.npz').read_bytes()).hexdigest()!=manifest['target_prediction_sha256']:
        raise AssertionError('Changed source or target provenance')
    anchors=pickle.loads((output/'anchors.pkl').read_bytes());error=0.;count=0
    with np.load(output/'calibration_predictions.npz',allow_pickle=False) as z,np.load(target_run/'heldout_predictions.npz',allow_pickle=False) as original:
        for key in ('labels','users','trials','conditions'):np.testing.assert_array_equal(z[key],original[key])
        for user in manifest['target_users']:
            for shots in (0,1,2):
                for name in ('Core',*(f'Core+{n}' for n in ADDED),'F0',*ADDED):
                    key=f'{user}_{shots}_{name}';p=apply_temperature(original[name][original['users']==user],name,manifest)
                    np.testing.assert_array_equal(z[f'{key}::without_anchor'],p)
                    if shots:
                        anchor,scale=anchors[(user,shots,name)]
                        logits=-anchor.transform(z[f'{key}::target_scaled'])[:,:7].astype(float)/scale
                        logits-=logits.max(1,keepdims=True);q=np.exp(logits);q/=q.sum(1,keepdims=True)
                        alpha=shots/(shots+2);p=(1-alpha)*p+alpha*q
                    np.testing.assert_allclose(p,z[f'{key}::with_anchor'],atol=1e-10,rtol=1e-9)
                    error=max(error,float(np.abs(p-z[f'{key}::with_anchor']).max()));count+=1
    audit={'status':'ok','anchor_probability_arrays_checked':count,'matching_no_anchor_arrays_checked':count,'max_absolute_probability_error':error,'source_hashes_checked':True}
    (output/'replay_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('source',type=Path);p.add_argument('target_run',type=Path);p.add_argument('output',type=Path);p.add_argument('--replay',action='store_true')
    p.add_argument('--probability-source',type=Path)
    a=p.parse_args();replay(a.source,a.target_run,a.output) if a.replay else run(a.raw,a.source,a.target_run,a.output,a.probability_source)
