import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'benchmarks/new_bank_v3'
def read(path):return json.loads(path.read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def test_source_policy_protocol_and_native_source_grid_are_bound():
    p=read(HERE/'ROAM_CLASS_TRANSITION_V1_PROTOCOL.json');s=read(HERE/'ROAM_CLASS_TRANSITION_V1_SOURCE_RESULTS.json')
    t=read(HERE/'ROAM_CLASS_TRANSITION_V1_TARGET_RESULTS.json')
    for rel,h in {**p['source_sha256'],**p['source_artifact_sha256'],**p['target_artifact_sha256'],**s['artifacts_sha256'],**t['artifacts_sha256']}.items():assert sha(ROOT/rel)==h
    assert s['protocol_sha256']==t['protocol_sha256']==sha(HERE/'ROAM_CLASS_TRANSITION_V1_PROTOCOL.json')
    assert len(s['source_recordings'])==72 and len(s['candidates'])==18 and s['independent_active_references']==360
    assert s['selected_config']==t['selected_config']=={'confirmations':5,'smoothing_windows':1,'confidence':0.}
    assert not s['target_arrays_read'] and not s['existing_classifiers_refitted'] and not s['training_scores_are_unbiased']
    assert t['source_policy_committed_before_target_inference'] and not t['target_fitted_or_selected']


def test_new_detector_is_calibration_independent_and_all_misses_and_false_events_remain():
    r=read(HERE/'ROAM_CLASS_TRANSITION_V1_TARGET_RESULTS.json');records=read(HERE/'roam_class_transition_v1/target/recordings.json')
    assert len(records)==60 and len({v['member'] for v in records})==20
    for member in {v['member'] for v in records}:
        blocks=[v for v in records if v['member']==member];assert {v['shots'] for v in blocks}=={0,1,2}
        signatures=[]
        for b in blocks:
            assert len(b['references'])==5 and b['detector_target_calibration_trials']==0
            signatures.append([(e['start'],e['end'],e['estimated_class'],e['algorithmic_available_at_sample_index'],e['probabilities']['source_window']) for e in b['events']])
            for e in b['events']:
                for q in e['probabilities'].values():
                    assert len(q)==3 and min(q)>=0 and abs(sum(q)-1)<1e-12
        assert signatures[0]==signatures[1]==signatures[2]
    assert len(r['cells'])==90
    for c in r['cells']:
        m=c['metrics'];assert m['references']==10 and m['matched']+m['missed']==10
        assert m['matched']+m['unmatched_detections']==m['detections']
        assert c['target_calibration_trials']==(0 if c['arm']=='source_window' else 6+3*c['shots'])


def test_independent_acceptance_keeps_failed_false_detection_and_fusion_guards():
    a=read(ROOT/'feature_bank/ROAM_CLASS_TRANSITION_V1_ACCEPTANCE.json');s=read(ROOT/'feature_bank/ROAM_CLASS_TRANSITION_V1_SOURCE_ACCEPTANCE.json')
    t=read(HERE/'ROAM_CLASS_TRANSITION_V1_TARGET_RESULTS.json')
    assert a['verifier_sha256']==s['verifier_sha256']==sha(HERE/'verify_roam_class_transition_v1.py')
    assert s['maximum_independent_source_window_probability_error']<1e-12 and a['maximum_independent_probability_error']<1e-12
    assert s['independent_native_logits_FSM_matching_and_selection'] and a['independent_native_logits_FSM_geometry_metrics']
    assert a['detector_primary_guards']==t['detector_primary_guards'] and not a['detector_primary_pass']
    assert a['within_detector_joint_guards']==t['within_detector_joint_guards'] and not a['within_detector_joint_pass']
    assert not a['detector_primary_guards']['no_more_unmatched_detections']
    assert a['validation_user_success_wins']==5 and a['validation_user_loss_wins']==0
    assert not any(a[k] for k in ('target_fitted_or_selected','conditional_loss_across_detectors_is_paired','native_experiment_rerun','default_promoted','physical_validation_proven','completion_proven'))


def test_current_report_and_table_bind_tradeoff_and_actual_calibration_costs():
    a=read(ROOT/'feature_bank/ROAM_CLASS_TRANSITION_V1_ACCEPTANCE.json')
    c=read(ROOT/'feature_bank/CURRENT_SCIENTIFIC_CONCLUSIONS_V1.json')['roam_class_transition']
    assert all(c[k]==v for k,v in a.items())
    index=read(ROOT/'feature_bank/delivery/INDEX.json')['roam_class_transition_acceptance']
    assert index['sha256']==sha(ROOT/'feature_bank/ROAM_CLASS_TRANSITION_V1_ACCEPTANCE.json')
    report=(ROOT/'feature_bank/REPORT.md').read_text(encoding='utf8')
    assert 'Source-selected active-to-active segmentation' in report and 'unmatched detections increase' in report
    path=ROOT/'feature_bank/delivery/new_bank_v3/class_transition_detection.csv'
    with path.open(encoding='utf8',newline='') as f:rows=list(csv.DictReader(f))
    assert len(rows)==90 and len({(r['subject'],r['current_shots_per_class'],r['arm']) for r in rows})==90
    native=read(HERE/'ROAM_CLASS_TRANSITION_V1_TARGET_RESULTS.json');lookup={(c['user'],c['shots'],c['arm']):c for c in native['cells']}
    for row in rows:
        cell=lookup[int(row['subject']),int(row['current_shots_per_class']),row['arm']]
        assert int(row['total_calibration_trials'])==cell['target_calibration_trials']
        assert int(row['missed'])==cell['metrics']['missed'] and float(row['end_to_end_success'])==cell['metrics']['end_to_end_success']
        assert row['conditional_loss_comparable_across_detectors']=='False'
