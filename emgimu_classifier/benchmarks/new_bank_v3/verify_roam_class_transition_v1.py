"""Independent source-grid and target state-machine/logit/metric verification.

No production detector feed, training, optimizer or classifier predict calls.
Frozen representation transforms are reused; logits and temporal/anchor
geometry are evaluated independently. No native experiment is rerun.
"""
import argparse
import json
import pickle
import zipfile
from pathlib import Path
import numpy as np
from emgimu.datasets.roam_cued_intervals_v1 import load_roam_cued_intervals,CLASSES,CHANNELS,PREPROCESSING
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.personal_temporal_bouts_v1 import TemporalBoutBatchV1
from benchmarks.new_bank_v3.roam_class_transition_v1 import ROOT,HERE,PROTOCOL,SOURCE,TARGET,OUT,INPUT,SOURCE_RECEIPT,TARGET_RECEIPT
from benchmarks.new_bank_v3.roam_native_joint_v1 import sha,write,ARCHIVE,build,GROUPS
from benchmarks.new_bank_v3.verify_roam_native_joint_v1 import softmax,temporal_oracle
from benchmarks.new_bank_v3.verify_roam_native_continuous_v2 import independent_metrics,window_oracle,close
from benchmarks.new_bank_v3.verify_calibration_rest_continuous_unibo_v1 import independent_matches


def read(path):return json.loads(path.read_text(encoding='utf8'))


def probability_oracle(bank,windows):
    batch=FeatureBatch(np.asarray(windows,np.float32),200.);raw={}
    for g in GROUPS:
        x=np.concatenate([f.transform(batch) for f in bank.families_[g]],axis=1)
        scaler,model=bank.models_[g]
        z=x.copy();z-=scaler.mean_.astype(z.dtype);z/=scaler.scale_.astype(z.dtype)
        q=softmax(z@model.coef_.T+model.intercept_)
        raw[g]=softmax(np.log(np.maximum(q,1e-15))/bank.temperatures_[g])
    q=sum(bank.policy_.population[i]*raw[g] for i,g in enumerate(GROUPS))
    return q/q.sum(1,keepdims=True)


def interval_oracle(q,ends,config):
    # Index-based trailing means and a separate finite-state implementation.
    state=None;candidate=None;run=0;first=None;start=None;known=False;result=[];discarded=0
    for i,end in enumerate(ends):
        average=np.mean(q[max(0,i-config['smoothing_windows']+1):i+1],axis=0)
        c=CLASSES[int(np.argmax(average))] if np.max(average)>=config['confidence'] else 'Unknown'
        if c==state:candidate=None;run=0
        else:
            if candidate!=c:candidate=c;run=0;first=int(end)-20
            run+=1
            if run==config['confirmations']:
                previous=state
                if state not in (None,'relax','Unknown'):
                    if c!='Unknown' and known and 200<=first-start<=6000:
                        result.append(dict(start=start,end=first,estimated_class=state,algorithmic_available_at_sample_index=int(end)))
                    else:discarded+=1
                state=c;start=None if c in ('relax','Unknown') else first;known=previous is not None
                candidate=None;run=0
        if start is not None and end-start>6040:known=False
    return result,state not in (None,'relax','Unknown') or run>0,discarded


def pairs(bounds,refs):
    return [{k:v for k,v in m.items() if k!='label'} for m in independent_matches(
        [[e['start'],e['end']] for e in bounds],[dict(r,native_label=r['class_name']) for r in refs])]


def source_counts(records):
    reference_counts=np.zeros(2,int);prediction_counts=np.zeros(2,int);true_positive=np.zeros(2,int)
    detections=matched=0
    for r in records:
        detections+=len(r['events']);matched+=len(r['matches'])
        for ref in r['references']:reference_counts[('close','open').index(ref['class_name'])]+=1
        for e in r['events']:prediction_counts[('close','open').index(e['estimated_class'])]+=1
        for m in r['matches']:
            c=r['references'][m['reference_index']]['class_name']
            if r['events'][m['event_index']]['estimated_class']==c:true_positive[('close','open').index(c)]+=1
    total=int(reference_counts.sum());den=reference_counts+prediction_counts
    f=np.divide(2*true_positive,den,out=np.zeros(2,float),where=den>0)
    return dict(references=total,detections=detections,matched=matched,unmatched_detections=detections-matched,
        detection_f1=2*matched/(total+detections) if total+detections else 0.,
        stable_class_proxy_macro_f1=float(f.mean()),stable_class_proxy_correct=int(true_positive.sum()))


def checked():
    p=read(PROTOCOL);assert sha(ARCHIVE)==p['archive_sha256']
    for rel,h in {**p['source_sha256'],**p['source_artifact_sha256']}.items():assert sha(ROOT/rel)==h,rel
    return p


def source_verify():
    p=checked();r=read(SOURCE);policy=read(OUT/'source/policy.json');inputs=read(INPUT)
    assert r['protocol_sha256']==sha(PROTOCOL)==policy['source_protocol_sha256']
    for rel,h in r['artifacts_sha256'].items():assert sha(ROOT/rel)==h
    assert policy['source_probability_cache_sha256']==sha(OUT/'source/window_scores.npz')
    assert policy['inference_bank_id']==inputs['inference_bank_id'] and policy['classes']==list(CLASSES)
    assert len(r['source_recordings'])==72 and len({v['member'] for v in r['source_recordings']})==72
    assert p['source_users']==list(range(1,19)) and len(p['source_candidates'])==18
    banks={i:pickle.loads((HERE/f'roam_source_fusion_v1/source/bank_fold{i}.pkl').read_bytes()) for i in range(3)}
    frozen={i:pickle.dumps(b) for i,b in banks.items()};receipts={};refs_seen=set();max_error=0.
    cache=np.load(OUT/'source/window_scores.npz',allow_pickle=False);evaluated={i:[] for i in range(18)}
    with zipfile.ZipFile(ARCHIVE) as z:
        for row in r['source_recordings']:
            u=row['user'];assert u in p['source_users'];bank=banks[row['fold']]
            assert bank.bank_id_==row['bank_id'] and bank.sensor_contract_==(200.,40,8)
            assert u in inputs['source_folds'][row['fold']]['held_users']
            assert set(bank.policy_.source_trials)==set(inputs['source_folds'][row['fold']]['fit_ids'])
            assert all(f'/s{u}/' not in t for t in bank.policy_.source_trials)
            d=load_roam_cued_intervals(z,row['member']);receipts[row['member']]=d.receipt;x=np.concatenate(d.batch.sequences)
            assert len(x)==d.receipt['native_samples']==row['native_samples'] and not d.receipt['excluded_intervals']
            refs=[v for v in d.receipt['cue_intervals'] if v['class_name']!='relax'];assert refs==row['references']
            refs_seen.update(v['trial_id'] for v in refs);key=row['key'];ends=cache[key+'_ends'];q=cache[key+'_q']
            assert np.array_equal(ends,np.arange(40,len(x)+1,8)) and q.shape==(len(ends),3)
            windows=np.stack([x[e-40:e] for e in ends]);expected=probability_oracle(bank,windows)
            max_error=max(max_error,float(abs(q-expected).max()))
            for i,config in enumerate(p['source_candidates']):
                events,_,_=interval_oracle(q,ends,config)
                evaluated[i].append(dict(user=u,events=events,references=refs,matches=pairs(events,refs)))
    assert len(refs_seen)==360 and json.loads(json.dumps([receipts[v['member']] for v in p['native_source_recordings']]))==p['native_source_recordings']
    assert len(cache.files)==144 and max_error<1e-12 and all(pickle.dumps(banks[i])==frozen[i] for i in banks)
    assert len(r['candidates'])==18
    for i,candidate in enumerate(r['candidates']):
        assert candidate['index']==i and candidate['config']==p['source_candidates'][i]
        per_user=[]
        for u in p['source_users']:
            scores=source_counts([b for b in evaluated[i] if b['user']==u]);assert scores['references']==20
            saved=next(v for v in candidate['per_user'] if v['user']==u)
            close({k:v for k,v in saved.items() if k!='user'},scores);per_user.append(scores)
        assert abs(candidate['mean_detection_f1']-np.mean([v['detection_f1'] for v in per_user]))<1e-12
        assert abs(candidate['mean_stable_class_proxy_macro_f1']-np.mean([v['stable_class_proxy_macro_f1'] for v in per_user]))<1e-12
        assert candidate['total_unmatched_detections']==sum(v['unmatched_detections'] for v in per_user)
    selected=sorted(r['candidates'],key=lambda c:(-c['mean_detection_f1'],-c['mean_stable_class_proxy_macro_f1'],c['total_unmatched_detections'],c['index']))[0]
    assert selected['index']==r['selected_candidate']==policy['selected_candidate']
    assert selected['config']==r['selected_config']==policy['config']
    assert not r['existing_classifiers_refitted'] and not r['target_arrays_read'] and not r['training_scores_are_unbiased']
    return dict(schema='roam_class_transition_v1_source_acceptance',protocol_sha256=sha(PROTOCOL),source_result_sha256=sha(SOURCE),
        policy_sha256=sha(OUT/'source/policy.json'),verifier_sha256=sha(Path(__file__)),
        native_source_recordings=72,independent_source_active_references=360,candidates_verified=18,
        maximum_independent_source_window_probability_error=max_error,selected_config=policy['config'],
        source_query_users_excluded_from_representation_fit=True,source_grid_scores_are_unbiased=False,
        target_arrays_read=False,classifiers_refitted=False,independent_native_logits_FSM_matching_and_selection=True,
        read_only_no_fitting=True,native_experiment_rerun=False,scope=r['scope'])


def target_verify():
    p=checked();r=read(TARGET);policy=read(OUT/'source/policy.json');a=read(SOURCE_RECEIPT)
    assert a['protocol_sha256']==r['protocol_sha256']==sha(PROTOCOL)
    assert a['source_result_sha256']==r['source_result_sha256']==sha(SOURCE)
    assert a['policy_sha256']==r['policy_sha256']==sha(OUT/'source/policy.json')
    assert r['source_acceptance_sha256']==sha(SOURCE_RECEIPT) and a['verifier_sha256']==sha(Path(__file__))
    for rel,h in {**p['target_artifact_sha256'],**r['artifacts_sha256']}.items():assert sha(ROOT/rel)==h
    records=read(OUT/'target/recordings.json');assert len(records)==60
    assert len({(v['user'],v['shots'],v['member']) for v in records})==60
    bank=pickle.loads((HERE/'roam_native_joint_v1/source_bank.pkl').read_bytes());w=build(bank);before=pickle.dumps(w)
    assert policy['inference_bank_id']==bank.bank_id_;max_error=0.;unique_refs=set();base_shapes={};checked_events=0
    with zipfile.ZipFile(ARCHIVE) as z:
        for row in records:
            u=row['user'];user=f'ROAM_s{u}';d=load_roam_cued_intervals(z,row['member']);x=np.concatenate(d.batch.sequences)
            assert row['member_sha256']==d.receipt['member_sha256'] and row['native_samples']==len(x)==d.receipt['native_samples']
            refs=[v for v in d.receipt['cue_intervals'] if v['class_name']!='relax'];assert refs==row['references']
            unique_refs.update(v['trial_id'] for v in refs)
            if row['member'] not in base_shapes:
                ends=np.arange(40,len(x)+1,8);windows=np.stack([x[e-40:e] for e in ends])
                q=probability_oracle(bank,windows);base_shapes[row['member']]=interval_oracle(q,ends,policy['config'])
            expected,censored,discarded=base_shapes[row['member']]
            assert row['censored_end']==censored and row['discarded_candidates']==discarded and row['detector_target_calibration_trials']==0
            assert len(expected)==len(row['events']) and pairs(expected,refs)==row['matches']
            personal=w.load_profile(ROOT/row['personal_path'],user_id=user)
            session=None if row['session_path'] is None else w.load_profile(ROOT/row['session_path'],user_id=user,session_id='hanging',personal=personal)
            immutable=pickle.dumps((personal,session))
            for pf in (personal,session):
                if pf is not None:assert row['member'] not in pf.calibration.recording_ids
            for i,(event,bounds) in enumerate(zip(row['events'],expected)):
                checked_events+=1
                for k,v in bounds.items():assert event[k]==v
                assert event['trial_id']==row['member']+f':class_estimated{i}' and event['recording_id']==row['member']
                assert event['boundary_kind']=='estimated' and event['returned_after_sample_index']==len(x)
                assert event['end']<=event['algorithmic_available_at_sample_index']<=len(x)
                start,end=event['start'],event['end'];batch=TemporalBoutBatchV1((x[start:end],),(event['trial_id'],),(row['member'],),(start,),200.,CHANNELS,PREPROCESSING,'estimated')
                source,window=window_oracle(bank,batch,personal,session)
                ld,ls=temporal_oracle(batch,personal.temporal);dtw,sig=ld,ls
                if session is not None:
                    sd,ss=temporal_oracle(batch,session.temporal);dtw=.5*(ld+sd);sig=.5*(ls+ss)
                probabilities=dict(source_window=source[0],window_full=window[0],joint_full=(.75*window+.125*dtw+.125*sig)[0])
                for arm,q in probabilities.items():
                    q/=q.sum();saved=np.asarray(event['probabilities'][arm]);max_error=max(max_error,float(abs(q-saved).max()))
                    assert saved.shape==(3,) and np.isfinite(saved).all() and np.all(saved>=0)
                    assert event['labels'][arm]==CLASSES[int(q.argmax())]
            assert pickle.dumps((personal,session))==immutable
    assert len(unique_refs)==100 and len(base_shapes)==20 and pickle.dumps(w)==before and max_error<1e-12
    assert len(r['cells'])==90 and len(r['aggregates'])==27
    for cell in r['cells']:
        close(cell['metrics'],independent_metrics([v for v in records if v['user']==cell['user'] and v['shots']==cell['shots']],cell['arm']))
        assert cell['target_calibration_trials']==(0 if cell['arm']=='source_window' else 6+3*cell['shots'])
    for cell in r['aggregates']:
        users=range(19,24) if cell['phase']=='validation' else range(24,29) if cell['phase']=='descriptive_final' else range(19,29)
        close(cell['metrics'],independent_metrics([v for v in records if v['user'] in users and v['shots']==cell['shots']],cell['arm']))
    old=read(HERE/'ROAM_NATIVE_CONTINUOUS_V1_RESULTS.json')
    ag={(v['phase'],v['shots'],v['arm']):v['metrics'] for v in r['aggregates']}
    og={(v['phase'],v['shots'],v['arm']):v['metrics'] for v in old['aggregates']}
    cu={(v['user'],v['shots'],v['arm']):v['metrics'] for v in r['cells']}
    ou={(v['user'],v['shots'],v['arm']):v['metrics'] for v in old['cells']}
    full,baseline,source=ag['validation',2,'joint_full'],og['validation',2,'joint_full'],ag['validation',2,'source_window']
    wins=sum(cu[u,2,'joint_full']['correct']>ou[u,2,'joint_full']['correct'] for u in range(19,24))
    loss_wins=sum(cu[u,2,'joint_full']['conditional_log_loss'] is not None and cu[u,2,'joint_full']['conditional_log_loss']<cu[u,2,'source_window']['conditional_log_loss'] for u in range(19,24))
    primary=dict(higher_end_to_end_success=full['correct']>baseline['correct'],higher_detection_recall=full['matched']>baseline['matched'],
        nonworse_active_f1=full['active_event_macro_f1']>=baseline['active_event_macro_f1'],
        nonworse_both_active_recalls=all(full['active_recall'][c]>=baseline['active_recall'][c] for c in ('close','open')),
        no_more_unmatched_detections=full['unmatched_detections']<=baseline['unmatched_detections'],at_least3_user_success_wins=wins>=3)
    secondary=dict(lower_log_loss=full['conditional_log_loss'] is not None and full['conditional_log_loss']<source['conditional_log_loss'],
        lower_brier=full['conditional_brier'] is not None and full['conditional_brier']<source['conditional_brier'],
        nonworse_end_to_end_success=full['correct']>=source['correct'],nonworse_active_f1=full['active_event_macro_f1']>=source['active_event_macro_f1'],
        nonworse_both_active_recalls=all(full['active_recall'][c]>=source['active_recall'][c] for c in ('close','open')),at_least3_user_loss_wins=loss_wins>=3)
    assert primary==r['detector_primary_guards'] and all(primary.values())==r['detector_primary_pass'] and wins==r['validation_user_success_wins']
    assert secondary==r['within_detector_joint_guards'] and all(secondary.values())==r['within_detector_joint_pass'] and loss_wins==r['validation_user_loss_wins']
    assert r['source_policy_committed_before_target_inference'] and not r['target_fitted_or_selected']
    assert not r['default_promoted'] and not r['physical_validation_proven'] and not r['completion_proven']
    return dict(schema='roam_class_transition_v1_acceptance',protocol_sha256=sha(PROTOCOL),source_result_sha256=sha(SOURCE),
        source_acceptance_sha256=sha(SOURCE_RECEIPT),target_result_sha256=sha(TARGET),policy_sha256=sha(OUT/'source/policy.json'),
        verifier_sha256=sha(Path(__file__)),selected_config=policy['config'],distinct_query_recordings=20,
        independent_active_references=100,recording_budget_blocks=60,cells=90,checked_event_budget_rows=checked_events,
        maximum_independent_probability_error=max_error,independent_native_logits_FSM_geometry_metrics=True,
        detector_primary_guards=primary,detector_primary_pass=all(primary.values()),within_detector_joint_guards=secondary,
        within_detector_joint_pass=all(secondary.values()),validation_user_success_wins=wins,validation_user_loss_wins=loss_wins,
        target_fitted_or_selected=False,source_policy_committed_before_target_inference=True,
        detector_target_calibration_trials=0,source_window_target_calibration_trials=0,
        conditional_loss_across_detectors_is_paired=False,read_only_no_fitting=True,native_experiment_rerun=False,
        two_shot_summary=[v for v in r['aggregates'] if v['shots']==2],default_promoted=False,
        physical_validation_proven=False,completion_proven=False,scope=r['scope'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-only',action='store_true');parser.add_argument('--write-receipt',action='store_true');args=parser.parse_args()
    result=source_verify() if args.source_only else target_verify()
    if args.write_receipt:write(SOURCE_RECEIPT if args.source_only else TARGET_RECEIPT,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('scope','two_shot_summary')}))
