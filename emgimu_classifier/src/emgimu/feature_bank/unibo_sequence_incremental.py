"""Held-day conditional value of DTW distances given validated bout-level G5."""
from pathlib import Path
import argparse
import hashlib
import json
import pickle
import numpy as np
from .core import FeatureBatch
from .unibo_sequence_temporal import load_bouts, g5_features, bout_weights, complete_paths
from .unibo_full_fusion import classifier
from .unibo_study import _metrics
from .unibo_temporal_complementarity import complementarity
from .force_nested_oof import fit_temperature,temperature_probability
from .core_incremental_oof import write
from emgimu.datasets.unibo_baseline import HAND_NAMES


PARENT_FILES=('fitted_states.pkl','run_manifest.json','bout_metadata.json',
    'heldout_predictions.npz','source_calibration_predictions.npz','split_trial_ids.json')


def check_parent(manifest,splits):
    if (manifest.get('train_days') != [1,2,3,4,5] or manifest.get('target_days') != [6]
            or manifest.get('final_days_opened') is not False):
        raise ValueError('Requires frozen source Days1-5 and independent validation Day6')
    users=[]
    for split in splits:
        users.append(split['user'])
        partitions=[set(split[k]) for k in ('train','calibration','evaluation')]
        if any(a&b for i,a in enumerate(partitions) for b in partitions[i+1:]):
            raise ValueError('Parent source/calibration/evaluation trial leakage')
    if len(set(users))!=len(users):
        raise ValueError('Duplicate parent source user')


def features(state,bouts):
    before=pickle.dumps(state)
    family,_,_=state['g5'];template,_=state['dtw']
    values=np.concatenate([g5_features(family,bouts),
        template.transform(complete_paths(bouts))],1)
    if before!=pickle.dumps(state):
        raise AssertionError('Parent source feature state changed')
    return values


def run(dataset,parent,output):
    if output.exists():raise FileExistsError(output)
    manifest=json.loads((parent/'run_manifest.json').read_text())
    splits=json.loads((parent/'split_trial_ids.json').read_text())
    check_parent(manifest,splits)
    parent_hashes={name:hashlib.sha256((parent/name).read_bytes()).hexdigest() for name in PARENT_FILES}
    print('[1/3] reusing native full-bout G5 and source-only DTW templates',flush=True)
    bouts=load_bouts(dataset)
    metadata=[{k:v for k,v in b.items() if k not in ('windows','path')} for b in bouts]
    if metadata!=json.loads((parent/'bout_metadata.json').read_text()):
        raise AssertionError('Parent bout data or boundaries changed')
    states,base_t=pickle.loads((parent/'fitted_states.pkl').read_bytes())
    added_states={};arrays={};source_arrays={};probs=[];base_probs=[];ordered=[];temperatures={};dims={}
    for user in sorted(states):
        print(f'[2/3] {user}: fit only G5+DTW classifiers on source 1-4 and 1-5',flush=True)
        inner=[b for b in bouts if b['user']==user and b['day']<=4]
        cal=[b for b in bouts if b['user']==user and b['day']==5]
        train=inner+cal;ev=[b for b in bouts if b['user']==user and b['day']==6]
        split=next(s for s in splits if s['user']==user)
        for key,selection in (('train',inner),('calibration',cal),('evaluation',ev)):
            if set(split[key])!={b['trial'] for b in selection}:
                raise AssertionError('Parent split coverage changed')
        inner_values=features(states[user][0],inner);cal_values=features(states[user][0],cal)
        source_values=features(states[user][1],train);target_values=features(states[user][1],ev)
        iscaler,imodel=classifier(inner_values,np.array([b['label'] for b in inner]),bout_weights(inner))
        cp=imodel.predict_proba(iscaler.transform(cal_values))
        temperature=fit_temperature(cp,np.array([b['label'] for b in cal]))
        scaler,model=classifier(source_values,np.array([b['label'] for b in train]),bout_weights(train))
        p=temperature_probability(model.predict_proba(scaler.transform(target_values)),temperature)
        added_states[user]=(iscaler,imodel,scaler,model)
        temperatures[user]=temperature;dims[user]=source_values.shape[1]
        arrays[f'{user}_target_features']=target_values
        arrays[f'{user}_target_bout_ids']=np.array([b['id'] for b in ev])
        arrays[f'{user}_G5_plus_DTW']=p
        source_arrays[f'{user}_cal_features']=cal_values
        source_arrays[f'{user}_cal_labels']=np.array([b['label'] for b in cal])
        source_arrays[f'{user}_cal_raw_probability']=cp
        source_arrays[f'{user}_fit_bout_ids']=np.array([b['id'] for b in train])
        # Independently reproduce the fixed matched baseline from parent states.
        family,bs,bm=states[user][1]['g5']
        bp=temperature_probability(bm.predict_proba(bs.transform(g5_features(family,ev))),base_t[user]['G5'])
        probs.append(p);base_probs.append(bp);ordered.extend(ev)
    y=np.array([b['label'] for b in ordered]);u=np.array([b['user'] for b in ordered])
    posture=np.array([b['posture'] for b in ordered]);w=bout_weights(ordered)
    probabilities={'G5':np.concatenate(base_probs),'G5_plus_DTW':np.concatenate(probs)}
    with np.load(parent/'heldout_predictions.npz',allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['bout_ids'],[b['id'] for b in ordered])
        np.testing.assert_allclose(saved['G5'],probabilities['G5'],atol=1e-12,rtol=0)
    cells=[('ALL','ALL',np.ones(len(y),bool))]
    cells.extend((user,'ALL',u==user) for user in sorted(states))
    cells.extend(('ALL',f'posture_{p}',posture==p) for p in sorted(set(posture)))
    cells.extend(('ALL',f'class_{HAND_NAMES[h]}',y==h) for h in range(4))
    rows=[];increments=[];pairs=[]
    for user,condition,mask in cells:
        if not mask.any():continue
        shared=dict(dataset='unibo_inail',phase='validation',subject=user,condition=condition,
            calibration_budget=0,protocol='matched personal historical Days1-5; complete oracle-labelled bouts',
            evaluation_bouts=int(mask.sum()))
        scores={name:_metrics(y[mask],p[mask],w[mask]) for name,p in probabilities.items()}
        for name,score in scores.items():
            rows.append({**shared,'method':name,'feature_family':name,
                'feature_dimension':next(iter(dims.values()))-(4 if name=='G5' else 0),**score})
        a,b=scores['G5'],scores['G5_plus_DTW']
        increments.append({**shared,'core_bank':'validated_G5_bout_mean','added_family':'full_bout_DTW_distances',
            'core_scope':'one-family specialist conditional control; not multi-family Core',
            'delta_logloss':a['log_loss']-b['log_loss'],'delta_macro_f1':b['macro_f1']-a['macro_f1'],
            'delta_brier':a['brier']-b['brier']})
        pairs.append({**shared,'family_a':'validated_G5_bout_mean','family_b':'validated_G5_plus_full_bout_DTW',
            **complementarity(y[mask],probabilities['G5'][mask],probabilities['G5_plus_DTW'][mask],w[mask])})
    print('[3/3] saving held-day conditional evidence and immutable source provenance',flush=True)
    output.mkdir(parents=True)
    for filename,values in (('feature_family_results',rows),('conditional_incremental',increments),('error_complementarity',pairs)):
        write(output/f'{filename}.csv',values)
    with (output/'fitted_states.pkl').open('wb') as handle:pickle.dump(added_states,handle)
    np.savez_compressed(output/'source_feature_arrays.npz',**source_arrays)
    np.savez_compressed(output/'target_feature_arrays.npz',**arrays)
    np.savez_compressed(output/'heldout_predictions.npz',**probabilities,labels=y,subjects=u,posture=posture,
        weights=w,bout_ids=np.array([b['id'] for b in ordered]))
    (output/'split_trial_ids.json').write_text(json.dumps(splits,indent=2))
    (output/'run_manifest.json').write_text(json.dumps(dict(dataset='unibo_inail',phase='validation',
        parent_run=parent.name,parent_artifacts_sha256=parent_hashes,train_days=[1,2,3,4,5],target_days=[6],
        final_days_opened=False,source_temperatures=temperatures,source_feature_dimensions=dims,
        baseline_prediction_max_error=0.,parent_families_refit=False,
        model='source-weighted StandardScaler and balanced LogisticRegression C1, max_iter1000',
        scope='Conditional predictive control given standalone G5, not multi-family Core; oracle full-bout boundaries, not streaming. No target-day tuning or calibration.'),indent=2))
    replay(parent,output)


def replay(parent,output):
    manifest=json.loads((output/'run_manifest.json').read_text())
    for name,digest in manifest['parent_artifacts_sha256'].items():
        if hashlib.sha256((parent/name).read_bytes()).hexdigest()!=digest:
            raise AssertionError('Parent source or evaluation provenance changed')
    states=pickle.loads((output/'fitted_states.pkl').read_bytes());arrays=0
    with np.load(output/'source_feature_arrays.npz',allow_pickle=False) as source, np.load(output/'target_feature_arrays.npz',allow_pickle=False) as target:
        for user,state in states.items():
            iscaler,imodel,scaler,model=state
            cp=imodel.predict_proba(iscaler.transform(source[f'{user}_cal_features']))
            np.testing.assert_allclose(cp,source[f'{user}_cal_raw_probability'],atol=1e-12,rtol=0)
            t=fit_temperature(cp,source[f'{user}_cal_labels'])
            np.testing.assert_allclose(t,manifest['source_temperatures'][user],atol=1e-12,rtol=0)
            p=temperature_probability(model.predict_proba(scaler.transform(target[f'{user}_target_features'])),t)
            np.testing.assert_allclose(p,target[f'{user}_G5_plus_DTW'],atol=1e-12,rtol=0)
            arrays+=2
    result=dict(status='ok',source_and_target_prediction_arrays=arrays,source_temperatures_recomputed=len(states),
        parent_artifact_hashes_verified=True,scope='Saved source classifiers, source temperatures and target predictions; native parent bout replay is separate')
    (output/'replay_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


def run_final(dataset,parent,frozen,output):
    if output.exists():raise FileExistsError(output)
    # Verify immutable source states and their Day-6 evidence before opening
    # explicitly requested final days. No source family or classifier is refit.
    replay(parent,frozen)
    manifest=json.loads((frozen/'run_manifest.json').read_text())
    if manifest['target_days']!=[6] or manifest['train_days']!=[1,2,3,4,5]:
        raise ValueError('Requires the frozen Day6 source-only conditional control')
    frozen_hashes={name:hashlib.sha256((frozen/name).read_bytes()).hexdigest() for name in
        ('fitted_states.pkl','run_manifest.json','source_feature_arrays.npz','target_feature_arrays.npz')}
    states=pickle.loads((frozen/'fitted_states.pkl').read_bytes())
    base_states,base_t=pickle.loads((parent/'fitted_states.pkl').read_bytes())
    before=pickle.dumps((states,base_states))
    print('[1/2] frozen conditional models; opening only final Days7-8',flush=True)
    target=load_bouts(dataset,days=(7,8))
    if {b['user'] for b in target}!=set(states):
        raise ValueError('Final personal-source user coverage changed')
    ordered=[];p=[];base=[];arrays={}
    for user in sorted(states):
        ev=[b for b in target if b['user']==user]
        values=features(base_states[user][1],ev)
        _,_,scaler,model=states[user]
        probability=temperature_probability(model.predict_proba(scaler.transform(values)),
            manifest['source_temperatures'][user])
        family,bs,bm=base_states[user][1]['g5']
        bp=temperature_probability(bm.predict_proba(bs.transform(g5_features(family,ev))),base_t[user]['G5'])
        arrays[f'{user}_target_features']=values
        arrays[f'{user}_G5_plus_DTW']=probability
        p.append(probability);base.append(bp);ordered.extend(ev)
    if before!=pickle.dumps((states,base_states)):
        raise AssertionError('Final evaluation changed a source state')
    y=np.array([b['label'] for b in ordered]);u=np.array([b['user'] for b in ordered])
    posture=np.array([b['posture'] for b in ordered]);day=np.array([b['day'] for b in ordered])
    w=bout_weights(ordered);probs={'G5':np.concatenate(base),'G5_plus_DTW':np.concatenate(p)}
    cells=[('ALL','ALL',np.ones(len(y),bool))]
    cells.extend((user,'ALL',u==user) for user in sorted(states))
    cells.extend(('ALL',f'posture_{v}',posture==v) for v in sorted(set(posture)))
    cells.extend(('ALL',f'day_{v}',day==v) for v in sorted(set(day)))
    cells.extend(('ALL',f'class_{HAND_NAMES[h]}',y==h) for h in range(4))
    rows=[];increments=[];pairs=[]
    for user,condition,mask in cells:
        if not mask.any():continue
        shared=dict(dataset='unibo_inail',phase='final',subject=user,condition=condition,
            calibration_budget=0,protocol='frozen personal historical Days1-5; complete oracle-labelled final bouts',
            evaluation_bouts=int(mask.sum()))
        scores={name:_metrics(y[mask],prob[mask],w[mask]) for name,prob in probs.items()}
        for name,score in scores.items():
            dimension=next(iter(manifest['source_feature_dimensions'].values()))-(4 if name=='G5' else 0)
            rows.append({**shared,'method':name,'feature_family':name,'feature_dimension':dimension,**score})
        a,b=scores['G5'],scores['G5_plus_DTW']
        increments.append({**shared,'core_bank':'validated_G5_bout_mean','added_family':'full_bout_DTW_distances',
            'core_scope':'one-family specialist conditional control; not multi-family Core',
            'delta_logloss':a['log_loss']-b['log_loss'],'delta_macro_f1':b['macro_f1']-a['macro_f1'],
            'delta_brier':a['brier']-b['brier']})
        pairs.append({**shared,'family_a':'validated_G5_bout_mean','family_b':'validated_G5_plus_full_bout_DTW',
            **complementarity(y[mask],probs['G5'][mask],probs['G5_plus_DTW'][mask],w[mask])})
    print('[2/2] saving independent final scores; no model fitting',flush=True)
    output.mkdir(parents=True)
    for filename,values in (('feature_family_results',rows),('conditional_incremental',increments),('error_complementarity',pairs)):
        write(output/f'{filename}.csv',values)
    np.savez_compressed(output/'heldout_predictions.npz',**probs,labels=y,subjects=u,posture=posture,days=day,
        weights=w,bout_ids=np.array([b['id'] for b in ordered]))
    np.savez_compressed(output/'target_feature_arrays.npz',**arrays)
    parent_splits=json.loads((frozen/'split_trial_ids.json').read_text())
    splits=[]
    for user in sorted(states):
        old=next(s for s in parent_splits if s['user']==user)
        source=set(old['train'])|set(old['calibration'])
        test={b['trial'] for b in ordered if b['user']==user}
        if source&test:raise AssertionError('Final source/test trial leakage')
        splits.append(dict(user=user,train=sorted(source),test=sorted(test)))
    (output/'split_trial_ids.json').write_text(json.dumps(splits,indent=2))
    evidence=dict(dataset='unibo_inail',phase='final',train_days=[1,2,3,4,5],target_days=[7,8],
        frozen_run=frozen.name,frozen_artifacts_sha256=frozen_hashes,parent_run=parent.name,
        parent_artifacts_sha256=manifest['parent_artifacts_sha256'],source_temperatures=manifest['source_temperatures'],
        classifier_or_family_fit=False,source_states_immutable=True,
        target_sha256={b['trial']:b['raw_sha256'] for b in ordered},
        scope='Frozen specialist conditional control, complete oracle-labelled bouts; not multi-family Core or streaming accuracy')
    (output/'run_manifest.json').write_text(json.dumps(evidence,indent=2))
    # Saved target inputs independently reproduce all final added-model arrays.
    with np.load(output/'target_feature_arrays.npz',allow_pickle=False) as saved:
        for user,state in states.items():
            _,_,scaler,model=state
            probability=temperature_probability(model.predict_proba(scaler.transform(saved[f'{user}_target_features'])),
                manifest['source_temperatures'][user])
            np.testing.assert_allclose(probability,saved[f'{user}_G5_plus_DTW'],atol=1e-12,rtol=0)
    result=dict(status='ok',source_frozen=True,classifier_or_family_fit=False,
        target_bouts=len(y),final_prediction_arrays_replayed=len(states),score_rows=len(rows),increment_rows=len(increments))
    (output/'replay_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


def replay_final(parent,frozen,output):
    evidence=json.loads((output/'run_manifest.json').read_text())
    if evidence['target_days']!=[7,8] or evidence['classifier_or_family_fit'] is not False:
        raise ValueError('Requires the frozen independent final protocol')
    for root,items in ((parent,evidence['parent_artifacts_sha256']),
            (frozen,evidence['frozen_artifacts_sha256'])):
        for name,digest in items.items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
                raise AssertionError('Immutable final source provenance changed')
    states=pickle.loads((frozen/'fitted_states.pkl').read_bytes())
    base_states,base_t=pickle.loads((parent/'fitted_states.pkl').read_bytes())
    before=pickle.dumps((states,base_states))
    maximum_error=0.;arrays=0
    with np.load(output/'target_feature_arrays.npz',allow_pickle=False) as inputs, np.load(output/'heldout_predictions.npz',allow_pickle=False) as saved:
        for user,state in states.items():
            values=inputs[f'{user}_target_features'];mask=saved['subjects']==user
            if int(mask.sum())!=len(values):raise AssertionError('Final user coverage changed')
            _,_,scaler,model=state
            p=temperature_probability(model.predict_proba(scaler.transform(values)),
                evidence['source_temperatures'][user])
            _,bs,bm=base_states[user][1]['g5']
            bp=temperature_probability(bm.predict_proba(bs.transform(values[:,:bs.n_features_in_])),base_t[user]['G5'])
            for name,probability in (('G5',bp),('G5_plus_DTW',p)):
                reference=saved[name][mask]
                maximum_error=max(maximum_error,float(np.max(np.abs(probability-reference))))
                np.testing.assert_allclose(probability,reference,atol=1e-12,rtol=0)
                arrays+=1
    if before!=pickle.dumps((states,base_states)):raise AssertionError('Final replay mutated source')
    result=dict(status='ok',final_prediction_arrays_replayed=arrays,
        maximum_absolute_probability_error=maximum_error,source_hashes_verified=True,
        classifier_or_family_fit=False,source_states_immutable=True,
        scope='Both baseline and added-model final predictions replayed from saved inputs and immutable source states; complete-bout offline protocol')
    (output/'replay_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('dataset',type=Path)
    parser.add_argument('parent',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--replay',action='store_true')
    parser.add_argument('--frozen',type=Path,help='Evaluate fixed validation-source classifiers on Days7-8 without fitting')
    args=parser.parse_args()
    if args.frozen:
        if args.replay:
            replay_final(args.parent,args.frozen,args.output)
        else:
            run_final(args.dataset,args.parent,args.frozen,args.output)
    elif args.replay:
        replay(args.parent,args.output)
    else:
        run(args.dataset,args.parent,args.output)
