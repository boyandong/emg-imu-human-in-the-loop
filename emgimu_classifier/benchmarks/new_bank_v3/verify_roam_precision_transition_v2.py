"""Read-only independent cached-source selection and native target verification.

No production detector, model.predict, training or experiment phase is called.
Source native-window correctness is inherited from its immutable independent
receipt; new source FSM/metrics/selection and target logits are checked here.
"""
import argparse
import json
import pickle
import zipfile
from pathlib import Path
import numpy as np
from emgimu.datasets.roam_cued_intervals_v1 import load_roam_cued_intervals,CLASSES,CHANNELS,PREPROCESSING
from emgimu.feature_bank.personal_temporal_bouts_v1 import TemporalBoutBatchV1
from benchmarks.new_bank_v3.roam_precision_transition_v2 import ROOT,HERE,PROTOCOL,SOURCE,TARGET,OUT,PREVIOUS_SOURCE,CACHE,SOURCE_RECEIPT,TARGET_RECEIPT
from benchmarks.new_bank_v3.roam_native_joint_v1 import sha,write,ARCHIVE,build
from benchmarks.new_bank_v3.verify_roam_native_joint_v1 import temporal_oracle
from benchmarks.new_bank_v3.verify_roam_native_continuous_v2 import independent_metrics,window_oracle,close
from benchmarks.new_bank_v3.verify_roam_class_transition_v1 import probability_oracle,interval_oracle,pairs,source_counts


def read(path):return json.loads(path.read_text(encoding='utf8'))


def same(actual,expected):
    if isinstance(expected,dict):
        assert actual.keys()==expected.keys()
        for key in expected:same(actual[key],expected[key])
    elif isinstance(expected,list):
        assert len(actual)==len(expected)
        for a,b in zip(actual,expected):same(a,b)
    elif isinstance(expected,(str,int,bool)) or expected is None:assert actual==expected
    else:assert abs(actual-expected)<1e-12


def checked():
    p=read(PROTOCOL)
    for rel,h in {**p['source_sha256'],**p['source_artifact_sha256']}.items():assert sha(ROOT/rel)==h,rel
    return p


def scores(index,config,users):
    # Algebraically separate calculation of equal-user beta=.5 detection F.
    values=[];recalls=[]
    for u in users:
        total=u['references'];hits=u['matched'];false=u['unmatched_detections']
        values.append(5*hits/(total+4*(hits+false)));recalls.append(hits/total)
    return dict(index=index,config=config,per_user=users,mean_detection_fbeta_half=float(sum(values)/len(values)),
        mean_detection_recall=float(sum(recalls)/len(recalls)),total_unmatched_detections=sum(u['unmatched_detections'] for u in users))


def source_verify():
    p=checked();r=read(SOURCE);policy=read(OUT/'source/policy.json');previous=read(PREVIOUS_SOURCE)
    prior_receipt=read(ROOT/'feature_bank/ROAM_CLASS_TRANSITION_V1_SOURCE_ACCEPTANCE.json')
    assert prior_receipt['source_result_sha256']==sha(PREVIOUS_SOURCE)
    assert prior_receipt['verifier_sha256']==sha(HERE/'verify_roam_class_transition_v1.py')
    assert prior_receipt['independent_native_logits_FSM_matching_and_selection']
    assert prior_receipt['maximum_independent_source_window_probability_error']<1e-12
    assert previous['artifacts_sha256'][CACHE.relative_to(ROOT).as_posix()]==sha(CACHE)==policy['source_probability_cache_sha256']
    assert policy['source_protocol_sha256']==r['protocol_sha256']==sha(PROTOCOL)
    for rel,h in r['artifacts_sha256'].items():assert sha(ROOT/rel)==h
    rows=r['source_recordings'];assert rows==previous['source_recordings'] and len(rows)==72
    assert len({v['member'] for v in rows})==72 and len({v['trial_id'] for row in rows for v in row['references']})==360
    assert p['source_users']==list(range(1,19)) and len(p['source_candidates'])==36 and len(r['candidates'])==36
    cache=np.load(CACHE,allow_pickle=False);assert len(cache.files)==144
    rebuilt=[]
    for index,config in enumerate(p['source_candidates']):
        events=[]
        for row in rows:
            key=row['key'];q=cache[key+'_q'];ends=cache[key+'_ends']
            assert np.array_equal(ends,np.arange(40,row['native_samples']+1,8)) and q.shape==(len(ends),3)
            assert np.isfinite(q).all() and np.all(q>=0) and np.allclose(q.sum(1),1.,atol=1e-12,rtol=0)
            bounds,_,_=interval_oracle(q,ends,config)
            events.append(dict(user=row['user'],references=row['references'],events=bounds,matches=pairs(bounds,row['references'])))
        users=[dict(user=u,**source_counts([v for v in events if v['user']==u])) for u in p['source_users']]
        summary=scores(index,config,users);same(summary,r['candidates'][index]);rebuilt.append(summary)
    old=previous['candidates'][previous['selected_candidate']]
    baseline=scores(-1,old['config'],old['per_user']);same(baseline,r['baseline'])
    assert old['config']==p['source_candidates'][0]
    same({k:v for k,v in baseline.items() if k!='index'},{k:v for k,v in rebuilt[0].items() if k!='index'})
    eligible=[v for v in rebuilt if v['mean_detection_recall']>=.9*baseline['mean_detection_recall'] and v['total_unmatched_detections']<=baseline['total_unmatched_detections']]
    chosen=sorted(eligible,key=lambda v:(-v['mean_detection_fbeta_half'],-v['mean_detection_recall'],v['total_unmatched_detections'],v['index']))[0]
    assert chosen['index']==r['selected_candidate']==policy['selected_candidate'] and chosen['config']==r['selected_config']==policy['config']
    assert policy['source_query_recordings']==[v['member'] for v in rows] and policy['classes']==list(CLASSES) and policy['rest_label']=='relax'
    assert policy['inference_bank_id']==read(HERE/'ROAM_SOURCE_FUSION_V1_SOURCE_INPUT.json')['inference_bank_id']
    assert not any(r[k] for k in ('target_arrays_read','existing_classifiers_refitted','native_source_inference_repeated','training_scores_are_unbiased'))
    return dict(schema='roam_precision_transition_v2_source_acceptance',protocol_sha256=sha(PROTOCOL),source_result_sha256=sha(SOURCE),
        policy_sha256=sha(OUT/'source/policy.json'),verifier_sha256=sha(Path(__file__)),
        inherited_native_source_receipt_sha256=sha(ROOT/'feature_bank/ROAM_CLASS_TRANSITION_V1_SOURCE_ACCEPTANCE.json'),
        source_probability_cache_sha256=sha(CACHE),source_recordings=72,independent_active_references=360,candidates_verified=36,
        selected_config=policy['config'],source_recall_retention=chosen['mean_detection_recall']/baseline['mean_detection_recall'],
        old_source_unmatched_detections=baseline['total_unmatched_detections'],new_source_unmatched_detections=chosen['total_unmatched_detections'],
        inherited_source_logits_verified=True,independent_cached_FSM_matching_metrics_and_selection=True,
        native_source_inference_repeated=False,target_arrays_read=False,classifiers_refitted=False,source_scores_are_unbiased=False,
        read_only_no_fitting=True,native_experiment_rerun=False,scope=r['scope'])


def target_verify():
    p=checked();assert sha(ARCHIVE)==p['archive_sha256'];r=read(TARGET);policy=read(OUT/'source/policy.json');a=read(SOURCE_RECEIPT)
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
    old=read(HERE/'ROAM_CLASS_TRANSITION_V1_TARGET_RESULTS.json')
    ag={(v['phase'],v['shots'],v['arm']):v['metrics'] for v in r['aggregates']}
    og={(v['phase'],v['shots'],v['arm']):v['metrics'] for v in old['aggregates']}
    cu={(v['user'],v['shots'],v['arm']):v['metrics'] for v in r['cells']}
    ou={(v['user'],v['shots'],v['arm']):v['metrics'] for v in old['cells']}
    full,baseline,source=ag['validation',2,'joint_full'],og['validation',2,'joint_full'],ag['validation',2,'source_window']
    wins=sum(cu[u,2,'joint_full']['unmatched_detections']<ou[u,2,'joint_full']['unmatched_detections'] and cu[u,2,'joint_full']['correct']>=ou[u,2,'joint_full']['correct'] for u in range(19,24))
    loss_wins=sum(cu[u,2,'joint_full']['conditional_log_loss'] is not None and cu[u,2,'joint_full']['conditional_log_loss']<cu[u,2,'source_window']['conditional_log_loss'] for u in range(19,24))
    primary=dict(fewer_unmatched_detections=full['unmatched_detections']<baseline['unmatched_detections'],
        nonworse_end_to_end_success=full['correct']>=baseline['correct'],nonworse_detection_recall=full['matched']>=baseline['matched'],
        nonworse_active_f1=full['active_event_macro_f1']>=baseline['active_event_macro_f1'],
        nonworse_both_active_recalls=all(full['active_recall'][c]>=baseline['active_recall'][c] for c in ('close','open')),
        at_least3_user_false_event_wins=wins>=3)
    secondary=dict(lower_log_loss=full['conditional_log_loss'] is not None and full['conditional_log_loss']<source['conditional_log_loss'],
        lower_brier=full['conditional_brier'] is not None and full['conditional_brier']<source['conditional_brier'],
        nonworse_end_to_end_success=full['correct']>=source['correct'],nonworse_active_f1=full['active_event_macro_f1']>=source['active_event_macro_f1'],
        nonworse_both_active_recalls=all(full['active_recall'][c]>=source['active_recall'][c] for c in ('close','open')),at_least3_user_loss_wins=loss_wins>=3)
    assert primary==r['detector_primary_guards'] and all(primary.values())==r['detector_primary_pass'] and wins==r['validation_user_false_event_wins']
    assert secondary==r['within_detector_joint_guards'] and all(secondary.values())==r['within_detector_joint_pass'] and loss_wins==r['validation_user_loss_wins']
    assert r['source_policy_committed_before_target_inference'] and not r['target_fitted_or_selected']
    assert not r['default_promoted'] and not r['physical_validation_proven'] and not r['completion_proven']
    return dict(schema='roam_precision_transition_v2_acceptance',protocol_sha256=sha(PROTOCOL),source_result_sha256=sha(SOURCE),
        source_acceptance_sha256=sha(SOURCE_RECEIPT),target_result_sha256=sha(TARGET),policy_sha256=sha(OUT/'source/policy.json'),
        verifier_sha256=sha(Path(__file__)),selected_config=policy['config'],distinct_query_recordings=20,
        independent_active_references=100,recording_budget_blocks=60,cells=90,checked_event_budget_rows=checked_events,
        maximum_independent_probability_error=max_error,independent_native_logits_FSM_geometry_metrics=True,
        detector_primary_guards=primary,detector_primary_pass=all(primary.values()),within_detector_joint_guards=secondary,
        within_detector_joint_pass=all(secondary.values()),validation_user_false_event_wins=wins,validation_user_loss_wins=loss_wins,
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
