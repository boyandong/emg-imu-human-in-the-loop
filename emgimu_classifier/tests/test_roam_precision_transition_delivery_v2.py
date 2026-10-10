import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'benchmarks/new_bank_v3'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def test_precision_policy_source_grid_preserves_frozen_inputs_and_selection_guards():
    p=read(HERE/'ROAM_PRECISION_TRANSITION_V2_PROTOCOL.json')
    s=read(HERE/'ROAM_PRECISION_TRANSITION_V2_SOURCE_RESULTS.json')
    t=read(HERE/'ROAM_PRECISION_TRANSITION_V2_TARGET_RESULTS.json')
    for rel,h in {**p['source_sha256'],**p['source_artifact_sha256'],**p['target_artifact_sha256'],**s['artifacts_sha256'],**t['artifacts_sha256']}.items():assert sha(ROOT/rel)==h
    assert s['protocol_sha256']==t['protocol_sha256']==sha(HERE/'ROAM_PRECISION_TRANSITION_V2_PROTOCOL.json')
    assert len(s['candidates'])==36 and len(s['source_recordings'])==72 and s['independent_active_references']==360
    chosen=s['candidates'][s['selected_candidate']];baseline=s['baseline']
    assert chosen['mean_detection_recall']>=.9*baseline['mean_detection_recall']
    assert chosen['total_unmatched_detections']<=baseline['total_unmatched_detections']
    assert not any(s[k] for k in ('existing_classifiers_refitted','native_source_inference_repeated','target_arrays_read','training_scores_are_unbiased'))


def test_target_reference_denominators_and_zero_calibration_controls_remain_identical():
    records=read(HERE/'roam_precision_transition_v2/target/recordings.json')
    old=read(HERE/'roam_class_transition_v1/target/recordings.json')
    lookup={(r['user'],r['shots'],r['member']):r for r in old}
    result=read(HERE/'ROAM_PRECISION_TRANSITION_V2_TARGET_RESULTS.json')
    assert len(records)==60 and len(result['cells'])==90
    for r in records:
        prior=lookup[r['user'],r['shots'],r['member']]
        assert r['references']==prior['references'] and r['member_sha256']==prior['member_sha256']
        assert r['detector_target_calibration_trials']==0
        assert all(e['algorithmic_available_at_sample_index']-e['end']==172 for e in r['events'])
    for member in {r['member'] for r in records}:
        signatures=[[(e['start'],e['end'],e['probabilities']['source_window']) for e in r['events']] for r in records if r['member']==member]
        assert len(signatures)==3 and signatures[0]==signatures[1]==signatures[2]
    for c in result['cells']:
        m=c['metrics'];assert m['references']==10 and m['matched']+m['missed']==10
        assert m['matched']+m['unmatched_detections']==m['detections']
        assert c['target_calibration_trials']==(0 if c['arm']=='source_window' else 6+3*c['shots'])


def test_independent_receipts_preserve_actual_efficacy_guards_and_scope():
    a=read(ROOT/'feature_bank/ROAM_PRECISION_TRANSITION_V2_ACCEPTANCE.json')
    s=read(ROOT/'feature_bank/ROAM_PRECISION_TRANSITION_V2_SOURCE_ACCEPTANCE.json')
    t=read(HERE/'ROAM_PRECISION_TRANSITION_V2_TARGET_RESULTS.json')
    assert a['verifier_sha256']==s['verifier_sha256']==sha(HERE/'verify_roam_precision_transition_v2.py')
    assert s['inherited_source_logits_verified'] and s['independent_cached_FSM_matching_metrics_and_selection']
    assert a['maximum_independent_probability_error']<1e-12 and a['independent_native_logits_FSM_geometry_metrics']
    assert a['detector_primary_guards']==t['detector_primary_guards']
    assert a['detector_primary_pass']==all(a['detector_primary_guards'].values())
    assert a['within_detector_joint_pass']==all(a['within_detector_joint_guards'].values())
    assert a['source_policy_committed_before_target_inference']
    assert not any(a[k] for k in ('target_fitted_or_selected','conditional_loss_across_detectors_is_paired','native_experiment_rerun','default_promoted','physical_validation_proven','completion_proven'))


def test_precision_policy_current_delivery_preserves_costs_and_tradeoff():
    a=read(ROOT/'feature_bank/ROAM_PRECISION_TRANSITION_V2_ACCEPTANCE.json')
    current=read(ROOT/'feature_bank/CURRENT_SCIENTIFIC_CONCLUSIONS_V1.json')['roam_precision_transition']
    assert all(current[k]==v for k,v in a.items())
    index=read(ROOT/'feature_bank/delivery/INDEX.json')['roam_precision_transition_acceptance']
    assert index['sha256']==sha(ROOT/'feature_bank/ROAM_PRECISION_TRANSITION_V2_ACCEPTANCE.json')
    report=(ROOT/'feature_bank/REPORT.md').read_text(encoding='utf8')
    assert 'Source-only precision-biased transition policy' in report
    assert 'cue-offset MAE' in report and '0.86s after its estimated boundary' in report
    with (ROOT/'feature_bank/delivery/new_bank_v3/precision_transition_detection.csv').open(encoding='utf8',newline='') as f:rows=list(csv.DictReader(f))
    assert len(rows)==90
    native=read(HERE/'ROAM_PRECISION_TRANSITION_V2_TARGET_RESULTS.json')
    lookup={(c['user'],c['shots'],c['arm']):c for c in native['cells']}
    for r in rows:
        c=lookup[int(r['subject']),int(r['current_shots_per_class']),r['arm']]
        assert int(r['total_calibration_trials'])==c['target_calibration_trials']
        assert int(r['unmatched_detections'])==c['metrics']['unmatched_detections']
        assert int(r['correct'])==c['metrics']['correct']
        assert r['conditional_loss_comparable_across_detectors']=='False'
