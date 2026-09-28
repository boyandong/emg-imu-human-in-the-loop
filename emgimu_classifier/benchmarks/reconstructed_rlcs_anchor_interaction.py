"""Source-only reconstructed ring-lag RLCS by Personal Anchor on native EPN.

This is a new, versioned method. It cannot establish identity with historical
RLCS. Validation/final users never fit the population model or temperature.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from emgimu.datasets.epn612 import load_epn612_windows
from emgimu.feature_bank.calibration import PersonalAnchor
from emgimu.feature_bank.epn_study import _metrics, aggregate_trials
from emgimu.feature_bank.force_nested_oof import fit_temperature, temperature_probability
from emgimu.feature_bank.reconstructed_ring import ReconstructedRlcs
from emgimu.feature_bank.screening import SEED
from emgimu.feature_bank.unibo_temporal_complementarity import complementarity


SOURCE_USERS=tuple(range(1,16))
OOF_GROUPS=(tuple(range(1,6)),tuple(range(6,11)),tuple(range(11,16)))
PHASE_USERS={'validation':(16,17,18),'final':(19,20,21)}
BUDGETS=(1,2,5)
ARMS=('B','B_plus_RLCS','B_plus_Anchor','B_plus_RLCS_plus_Anchor')
ARCHIVE_SHA256='4ee8db037385e7bee1e6ac6f9e9eea4f0869e25f7825f5eb7e5be0dff4f93c21'
ARCHIVE_BYTES=5483385161


def sha(path:Path)->str:
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path:Path,rows:list[dict])->None:
    with path.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def classifier(x:np.ndarray,y:np.ndarray):
    scaler=StandardScaler().fit(x)
    model=LogisticRegression(C=1,class_weight='balanced',max_iter=1000,random_state=SEED).fit(scaler.transform(x),y)
    np.testing.assert_array_equal(model.classes_,np.arange(6))
    return scaler,model


def anchor_probability(anchor:PersonalAnchor,temperature:float,x:np.ndarray)->np.ndarray:
    if not np.isfinite(temperature) or temperature<=0:
        raise ValueError('calibration-only anchor temperature must be positive')
    logits=-anchor.transform(x)[:,:6].astype(np.float64)/temperature
    logits-=logits.max(axis=1,keepdims=True)
    p=np.exp(logits)
    return p/p.sum(axis=1,keepdims=True)


def four_arms(f0:np.ndarray,ring:np.ndarray,f0_personal:np.ndarray,ring_personal:np.ndarray)->dict[str,np.ndarray]:
    arrays=(f0,ring,f0_personal,ring_personal)
    if any(a.shape!=arrays[0].shape or a.ndim!=2 or not np.isfinite(a).all() for a in arrays):
        raise ValueError('four aligned finite probability providers required')
    return {'B':f0,'B_plus_RLCS':(f0+ring)/2,
        'B_plus_Anchor':f0_personal,'B_plus_RLCS_plus_Anchor':(f0_personal+ring_personal)/2}


def interaction(scores:dict[str,dict])->dict[str,float]:
    b,r,a,both=(scores[key] for key in ARMS)
    return {'S_macro_f1':both['macro_f1']-r['macro_f1']-a['macro_f1']+b['macro_f1'],
        'S_negative_logloss':-both['log_loss']+r['log_loss']+a['log_loss']-b['log_loss'],
        'S_negative_brier':-both['brier']+r['brier']+a['brier']-b['brier']}


def export_delivery(output:Path)->None:
    """Map immutable four-arm scores into the canonical delivery vocabulary."""
    with (output/'arm_scores.csv').open(newline='',encoding='utf-8') as handle:
        rows=list(csv.DictReader(handle))
    if len(rows)!=48 or {r['arm'] for r in rows}!=set(ARMS):
        raise ValueError('Expected the complete 3-user plus pooled, 3-budget, four-arm grid')
    features=[];increments=[]
    names={'B':'F0_frozen','B_plus_RLCS':'F0_plus_RLCS_reconstructed',
        'B_plus_Anchor':'F0_PersonalAnchor',
        'B_plus_RLCS_plus_Anchor':'F0_plus_RLCS_reconstructed_PersonalAnchor'}
    for row in rows:
        compact={key:value for key,value in row.items() if key not in ('ring_family','base','arm','calibration_trials')}
        features.append({**compact,'condition':'cross_user','calibration_budget':row['shots_per_class'],
            'feature_family':names[row['arm']],'method':row['arm']})
    cells=sorted({(r['phase'],r['subject'],r['shots_per_class']) for r in rows})
    for phase,subject,shots in cells:
        scores={r['arm']:r for r in rows if (r['phase'],r['subject'],r['shots_per_class'])==(phase,subject,shots)}
        if set(scores)!=set(ARMS):raise AssertionError('Incomplete matched delivery cell')
        for core,added,family in (('B','B_plus_RLCS','RLCS_reconstructed_v1'),
                ('B','B_plus_Anchor','PersonalAnchor'),
                ('B_plus_Anchor','B_plus_RLCS_plus_Anchor','RLCS_reconstructed_v1_after_PersonalAnchor')):
            before,after=scores[core],scores[added]
            increments.append(dict(dataset='epn612',phase=phase,subject=subject,condition='cross_user',
                calibration_budget=shots,core_bank=core,added_family=family,
                delta_macro_f1=float(after['macro_f1'])-float(before['macro_f1']),
                delta_logloss=float(before['log_loss'])-float(after['log_loss']),
                delta_brier=float(before['brier'])-float(after['brier']),
                scope='new reconstructed RLCS and calibration-only anchor; not historical identity'))
    write_csv(output/'feature_family_results.csv',features)
    write_csv(output/'conditional_incremental.csv',increments)
    for name in ('interaction_results.csv','error_complementarity.csv'):
        with (output/name).open(newline='',encoding='utf-8') as handle:
            source_rows=list(csv.DictReader(handle))
        compact=[]
        for row in source_rows:
            item={key:value for key,value in row.items() if key!='ring_family' and
                (name=='interaction_results.csv' or key!='base')}
            if name=='interaction_results.csv':
                item.update(family_a='RLCS_reconstructed_v1',family_b='PersonalAnchor')
            compact.append(item)
        write_csv(output/name,compact)
    manifest_path=output/'run_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    for name in ('feature_family_results.csv','conditional_incremental.csv',
            'interaction_results.csv','error_complementarity.csv'):
        manifest['output_hashes'][name]=sha(output/name)
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')


def prepare(archive:Path,frozen_f0:Path,output:Path,protocol:Path)->None:
    if output.exists():raise FileExistsError(output)
    if archive.stat().st_size!=ARCHIVE_BYTES or sha(archive)!=ARCHIVE_SHA256:
        raise ValueError('EPN source archive differs from verified public release')
    frozen_manifest=json.loads((frozen_f0/'run_manifest.json').read_text())
    if frozen_manifest.get('dataset')!='epn612' or frozen_manifest.get('phase')!='validation':
        raise ValueError('Expected the frozen EPN source-user validation provider')
    print('[1/3] loading source users 1-15 and fixed reconstructed RLCS',flush=True)
    source=load_epn612_windows(archive,users=SOURCE_USERS)
    family=ReconstructedRlcs().fit(source.batch,source.labels)
    x,y,u,trials,_=aggregate_trials(family.transform(source.batch),source)
    if x.shape[1]!=8 or set(trials)!=set(frozen_manifest['source_trials']) or set(u)!=set(SOURCE_USERS):
        raise AssertionError('Source trials, users, or reconstructed feature shape changed')
    print('[2/3] fitting three source-user OOF folds and one full-source classifier',flush=True)
    oof=np.full((len(y),6),np.nan,dtype=float)
    for held_out in OOF_GROUPS:
        validation=np.isin(u,held_out)
        scaler,model=classifier(x[~validation],y[~validation])
        oof[validation]=model.predict_proba(scaler.transform(x[validation]))
    if not np.isfinite(oof).all():raise AssertionError('Incomplete source OOF coverage')
    temperature=float(fit_temperature(oof,y))
    scaler,model=classifier(x,y)
    print('[3/3] saving source state; target users not opened',flush=True)
    output.mkdir(parents=True)
    with (output/'fitted_state.pkl').open('wb') as handle:pickle.dump((family,scaler,model,temperature),handle)
    np.savez_compressed(output/'source_oof_predictions.npz',probability=oof,labels=y,users=u,trials=trials)
    manifest=dict(dataset='epn612',family=family.family_id,feature_dimension=8,
        source_users=list(SOURCE_USERS),source_oof_groups=[list(v) for v in OOF_GROUPS],
        source_trial_ids=trials.tolist(),source_archive_sha256=ARCHIVE_SHA256,
        source_archive_size=archive.stat().st_size,source_temperature=temperature,
        source_classifier='balanced LogisticRegression C1 plus StandardScaler',
        validation_or_final_users_opened=False,protocol_sha256=sha(protocol),
        frozen_f0_manifest_sha256=sha(frozen_f0/'run_manifest.json'),
        scope='New reconstructed RLCS family; no historical algorithm identity')
    (output/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'status':'ok','source_trials':len(y),'temperature':temperature}))


def evaluate(archive:Path,source_run:Path,frozen_f0:Path,output:Path,phase:str,protocol:Path)->None:
    if output.exists():raise FileExistsError(output)
    if phase not in PHASE_USERS:raise ValueError(phase)
    source_manifest=json.loads((source_run/'run_manifest.json').read_text())
    f0_manifest=json.loads((frozen_f0/'run_manifest.json').read_text())
    f0_calibration=json.loads((frozen_f0/'probability_calibration.json').read_text())
    if (source_manifest['protocol_sha256']!=sha(protocol) or f0_manifest.get('phase')!=phase
            or f0_manifest.get('classifier_or_family_fit') is not False
            or f0_manifest.get('anchor_alpha')!='shots/(shots+2)'
            or set(source_manifest['source_trial_ids'])!=set(f0_manifest['source_trials'])
            or tuple(f0_calibration['fit_users'])!=SOURCE_USERS):
        raise ValueError('Frozen source providers or protocol changed')
    splits=json.loads((frozen_f0/'split_trial_ids.json').read_text())
    users=PHASE_USERS[phase]
    expected={(user,shots) for user in users for shots in (0,*BUDGETS)}
    if {(int(s['user']),int(s['shots'])) for s in splits}!=expected:
        raise ValueError('Frozen target split grid changed')
    print(f'[{phase} 1/3] loading native target users {users}',flush=True)
    target=load_epn612_windows(archive,users=users)
    f0_states,f0_anchors=pickle.loads((frozen_f0/'fitted_states.pkl').read_bytes())
    family,ring_scaler,ring_model,ring_t=pickle.loads((source_run/'fitted_state.pkl').read_bytes())
    before=pickle.dumps((f0_states,f0_anchors,family,ring_scaler,ring_model,ring_t))
    f0_family,f0_scaler,f0_model=f0_states['F0']
    f0_x,y,u,trials,_=aggregate_trials(f0_family.transform(target.batch),target)
    ring_x,ry,ru,rt,_=aggregate_trials(family.transform(target.batch),target)
    for actual,expected_values in ((ry,y),(ru,u),(rt,trials)):
        np.testing.assert_array_equal(actual,expected_values)
    f0_x=f0_scaler.transform(f0_x);ring_x=ring_scaler.transform(ring_x)
    f0=temperature_probability(f0_model.predict_proba(f0_x),float(f0_calibration['temperatures']['F0']))
    ring=temperature_probability(ring_model.predict_proba(ring_x),ring_t)
    with np.load(frozen_f0/'heldout_predictions.npz',allow_pickle=False) as previous:
        for name,current in (('labels',y),('users',u),('trials',trials)):
            np.testing.assert_array_equal(previous[name],current)
    if pickle.dumps((f0_states,f0_anchors,family,ring_scaler,ring_model,ring_t))!=before:
        raise AssertionError('Target transforms mutated frozen source states')
    print(f'[{phase} 2/3] four matched arms with saved whole-trial 1/2/5-shot splits',flush=True)
    score_rows=[];interaction_rows=[];error_rows=[];prediction_arrays={};split_rows=[]
    by_cell={(int(s['user']),int(s['shots'])):s for s in splits}
    for shots in BUDGETS:
        pooled_truth=[];pooled={name:[] for name in ARMS}
        pooled_ring=[];pooled_ring_personal=[]
        for user in users:
            split=by_cell[(user,shots)]
            cal_ids=np.asarray(split['calibration'],dtype=trials.dtype)
            ev_ids=np.asarray(split['evaluation'],dtype=trials.dtype)
            cal=np.flatnonzero(np.isin(trials,cal_ids));ev=np.flatnonzero(np.isin(trials,ev_ids))
            if (len(cal)!=6*shots or len(ev)!=len(ev_ids) or not len(ev)
                    or set(cal_ids)&set(ev_ids) or np.any(u[cal]!=user) or np.any(u[ev]!=user)
                    or set(y[cal])!=set(range(6))):
                raise ValueError(f'Invalid frozen calibration/evaluation trials: {user}/{shots}')
            alpha=shots/(shots+2)
            f0_anchor,f0_anchor_t=f0_anchors[(user,shots,'F0')]
            expected_prototypes=np.stack([f0_x[cal][y[cal]==label].mean(0) for label in range(6)])
            np.testing.assert_allclose(f0_anchor.prototypes_,expected_prototypes,rtol=1e-6,atol=1e-7)
            f0_personal=(1-alpha)*f0[ev]+alpha*anchor_probability(f0_anchor,f0_anchor_t,f0_x[ev])
            ring_anchor=PersonalAnchor().fit(ring_x[cal],y[cal])
            ring_anchor_t=max(float(np.median(ring_anchor.transform(ring_x[cal])[:,:6])),1e-10)
            ring_personal=(1-alpha)*ring[ev]+alpha*anchor_probability(ring_anchor,ring_anchor_t,ring_x[ev])
            arms=four_arms(f0[ev],ring[ev],f0_personal,ring_personal)
            scores={name:_metrics(y[ev],p,np.ones(len(ev))) for name,p in arms.items()}
            common=dict(dataset='epn612',phase=phase,subject=user,shots_per_class=shots,
                calibration_trials=len(cal),evaluation_trials=len(ev),
                ring_family='RLCS_reconstructed_v1',base='frozen_F0')
            score_rows.extend({**common,'arm':name,**scores[name]} for name in ARMS)
            interaction_rows.append({**common,**interaction(scores)})
            error_rows.append({**common,'family_a':'reconstructed_RLCS_population',
                'family_b':'reconstructed_RLCS_personalized',
                **complementarity(y[ev],ring[ev],ring_personal,np.ones(len(ev)))})
            split_rows.append(dict(user=user,shots=shots,calibration=cal_ids.tolist(),evaluation=ev_ids.tolist()))
            prediction_arrays[f'{user}_{shots}_labels']=y[ev]
            prediction_arrays[f'{user}_{shots}_trials']=trials[ev]
            prediction_arrays[f'{user}_{shots}_ring_population']=ring[ev]
            prediction_arrays[f'{user}_{shots}_ring_personalized']=ring_personal
            for name,p in arms.items():
                prediction_arrays[f'{user}_{shots}_{name}']=p;pooled[name].append(p)
            pooled_truth.append(y[ev]);pooled_ring.append(ring[ev]);pooled_ring_personal.append(ring_personal)
        truth=np.concatenate(pooled_truth)
        scores={name:_metrics(truth,np.concatenate(parts),np.ones(len(truth))) for name,parts in pooled.items()}
        common=dict(dataset='epn612',phase=phase,subject='ALL',shots_per_class=shots,
            calibration_trials=6*shots*len(users),evaluation_trials=len(truth),
            ring_family='RLCS_reconstructed_v1',base='frozen_F0')
        score_rows.extend({**common,'arm':name,**scores[name]} for name in ARMS)
        interaction_rows.append({**common,**interaction(scores)})
        error_rows.append({**common,'family_a':'reconstructed_RLCS_population',
            'family_b':'reconstructed_RLCS_personalized',
            **complementarity(truth,np.concatenate(pooled_ring),np.concatenate(pooled_ring_personal),np.ones(len(truth)))})
    print(f'[{phase} 3/3] saving scores, paired errors, and exact predictions',flush=True)
    output.mkdir(parents=True)
    for name,rows in (('arm_scores',score_rows),('interaction_results',interaction_rows),('error_complementarity',error_rows)):
        write_csv(output/f'{name}.csv',rows)
    np.savez_compressed(output/'heldout_predictions.npz',**prediction_arrays)
    (output/'split_trial_ids.json').write_text(json.dumps(split_rows,indent=2)+'\n')
    manifest=dict(dataset='epn612',phase=phase,ring_family='RLCS_reconstructed_v1',
        source_users=list(SOURCE_USERS),target_users=list(users),budgets=list(BUDGETS),
        protocol_sha256=sha(protocol),source_run=source_run.name,frozen_f0_run=frozen_f0.name,
        source_hashes={name:sha(source_run/name) for name in ('fitted_state.pkl','run_manifest.json','source_oof_predictions.npz')},
        frozen_f0_hashes={name:sha(frozen_f0/name) for name in ('fitted_states.pkl','run_manifest.json','probability_calibration.json','split_trial_ids.json','heldout_predictions.npz')},
        source_model_or_temperature_fit=False,target_anchor_fit_on_calibration_only=True,
        target_split_selection=False,historical_equivalence=False,
        output_hashes={name:sha(output/name) for name in ('arm_scores.csv','interaction_results.csv','error_complementarity.csv','heldout_predictions.npz','split_trial_ids.json')},
        boundary='Reconstructed RLCS and target-calibration-only anchors; previously inspected final users are descriptive, not newly untouched')
    (output/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    export_delivery(output)
    print(json.dumps({'status':'ok','phase':phase,'score_rows':len(score_rows),'interaction_rows':len(interaction_rows)}))


def summarize(validation:Path,final:Path,output:Path)->None:
    result={'pair':'RLCS_reconstructed_v1 x PersonalAnchor','historical_equivalence':False,
        'completion_proven':False,'phases':{},'pooled_interactions':[]}
    for phase,root in (('validation',validation),('final',final)):
        manifest=json.loads((root/'run_manifest.json').read_text())
        if manifest['phase']!=phase or manifest['source_model_or_temperature_fit'] is not False:
            raise ValueError('Phase manifest changed')
        for name,digest in manifest['output_hashes'].items():
            if sha(root/name)!=digest:raise AssertionError(f'Output changed: {phase}/{name}')
        replay=json.loads((root/'replay_audit.json').read_text())
        if (replay.get('status')!='ok' or replay.get('prediction_arrays')!=54
                or replay.get('maximum_absolute_probability_error')!=0.
                or replay.get('source_model_or_temperature_refit') is not False):
            raise AssertionError(f'Frozen native replay incomplete: {phase}')
        with (root/'interaction_results.csv').open(newline='',encoding='utf-8') as handle:
            rows=list(csv.DictReader(handle))
        result['pooled_interactions'].extend({**row,'phase':phase} for row in rows if row['subject']=='ALL')
        result['phases'][phase]={'run_manifest_sha256':sha(root/'run_manifest.json'),
            'replay_audit_sha256':sha(root/'replay_audit.json'),
            'output_hashes':manifest['output_hashes']}
    result['boundary']='New reconstructed RLCS only; EPN final users previously examined; no old-method or own-device claim'
    output.write_text(json.dumps(result,indent=2)+'\n')


def verify(archive:Path,source_run:Path,frozen_f0:Path,output:Path,phase:str,protocol:Path)->None:
    """Recompute all target probabilities from native trials and saved states."""
    manifest=json.loads((output/'run_manifest.json').read_text())
    if manifest['phase']!=phase or manifest['protocol_sha256']!=sha(protocol):
        raise ValueError('Run phase or frozen protocol changed')
    for root,key in ((source_run,'source_hashes'),(frozen_f0,'frozen_f0_hashes')):
        for name,digest in manifest[key].items():
            if sha(root/name)!=digest:raise AssertionError(f'Frozen input changed: {name}')
    for name,digest in manifest['output_hashes'].items():
        if sha(output/name)!=digest:raise AssertionError(f'Saved result changed: {name}')
    target=load_epn612_windows(archive,users=PHASE_USERS[phase])
    f0_states,f0_anchors=pickle.loads((frozen_f0/'fitted_states.pkl').read_bytes())
    ring_family,ring_scaler,ring_model,ring_t=pickle.loads((source_run/'fitted_state.pkl').read_bytes())
    f0_family,f0_scaler,f0_model=f0_states['F0']
    f0_features,y,u,trials,_=aggregate_trials(f0_family.transform(target.batch),target)
    ring_features,ry,ru,rt,_=aggregate_trials(ring_family.transform(target.batch),target)
    for actual,reference in ((ry,y),(ru,u),(rt,trials)):
        np.testing.assert_array_equal(actual,reference)
    f0_x=f0_scaler.transform(f0_features);ring_x=ring_scaler.transform(ring_features)
    calibration=json.loads((frozen_f0/'probability_calibration.json').read_text())
    f0=temperature_probability(f0_model.predict_proba(f0_x),float(calibration['temperatures']['F0']))
    ring=temperature_probability(ring_model.predict_proba(ring_x),ring_t)
    splits=json.loads((output/'split_trial_ids.json').read_text())
    maximum_error=0.;prediction_arrays=0
    with np.load(output/'heldout_predictions.npz',allow_pickle=False) as saved:
        for split in splits:
            user,shots=int(split['user']),int(split['shots'])
            cal=np.flatnonzero(np.isin(trials,np.asarray(split['calibration'],dtype=trials.dtype)))
            ev=np.flatnonzero(np.isin(trials,np.asarray(split['evaluation'],dtype=trials.dtype)))
            if (len(cal)!=6*shots or len(ev)!=len(split['evaluation'])
                    or set(y[cal])!=set(range(6)) or np.any(u[ev]!=user)):
                raise AssertionError('Native target split changed')
            alpha=shots/(shots+2)
            f0_anchor,f0_t=f0_anchors[(user,shots,'F0')]
            f0_personal=(1-alpha)*f0[ev]+alpha*anchor_probability(f0_anchor,f0_t,f0_x[ev])
            ring_anchor=PersonalAnchor().fit(ring_x[cal],y[cal])
            ring_anchor_t=max(float(np.median(ring_anchor.transform(ring_x[cal])[:,:6])),1e-10)
            ring_personal=(1-alpha)*ring[ev]+alpha*anchor_probability(ring_anchor,ring_anchor_t,ring_x[ev])
            expected={**four_arms(f0[ev],ring[ev],f0_personal,ring_personal),
                'ring_population':ring[ev],'ring_personalized':ring_personal}
            for name,value in expected.items():
                reference=saved[f'{user}_{shots}_{name}']
                maximum_error=max(maximum_error,float(np.max(np.abs(value-reference))))
                np.testing.assert_allclose(value,reference,atol=1e-12,rtol=0)
                prediction_arrays+=1
            np.testing.assert_array_equal(saved[f'{user}_{shots}_labels'],y[ev])
            np.testing.assert_array_equal(saved[f'{user}_{shots}_trials'],trials[ev])
    result=dict(status='ok',phase=phase,native_target_reloaded=True,
        source_model_or_temperature_refit=False,target_anchors_recomputed_from_saved_calibration_trials=True,
        source_and_output_hashes_verified=True,prediction_arrays=prediction_arrays,
        maximum_absolute_probability_error=maximum_error,
        boundary='Reconstructed RLCS, not historical RLCS or own-device validation')
    (output/'replay_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest='mode',required=True)
    source=sub.add_parser('prepare');source.add_argument('archive',type=Path)
    source.add_argument('frozen_f0',type=Path);source.add_argument('output',type=Path)
    target=sub.add_parser('evaluate');target.add_argument('archive',type=Path)
    target.add_argument('source_run',type=Path);target.add_argument('frozen_f0',type=Path)
    target.add_argument('output',type=Path);target.add_argument('phase',choices=tuple(PHASE_USERS))
    summary=sub.add_parser('summarize');summary.add_argument('validation',type=Path)
    summary.add_argument('final',type=Path);summary.add_argument('output',type=Path)
    check=sub.add_parser('verify');check.add_argument('archive',type=Path)
    check.add_argument('source_run',type=Path);check.add_argument('frozen_f0',type=Path)
    check.add_argument('output',type=Path);check.add_argument('phase',choices=tuple(PHASE_USERS))
    delivery=sub.add_parser('export');delivery.add_argument('output',type=Path)
    parser.add_argument('--protocol',type=Path,default=Path(__file__).with_name('reconstructed_rlcs_anchor_protocol.json'))
    args=parser.parse_args()
    if args.mode=='prepare':prepare(args.archive,args.frozen_f0,args.output,args.protocol)
    elif args.mode=='evaluate':evaluate(args.archive,args.source_run,args.frozen_f0,args.output,args.phase,args.protocol)
    elif args.mode=='verify':verify(args.archive,args.source_run,args.frozen_f0,args.output,args.phase,args.protocol)
    elif args.mode=='export':export_delivery(args.output)
    else:summarize(args.validation,args.final,args.output)
