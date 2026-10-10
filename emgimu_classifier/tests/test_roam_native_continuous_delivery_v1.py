"""Frozen native outputs, complete denominators and current register integrity."""
import csv
import hashlib
import json
from pathlib import Path
import pytest
from benchmarks.new_bank_v3.audit_roam_native_continuous_v1 import audit

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'benchmarks/new_bank_v3'


def read(path):return json.loads(path.read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def test_native_continuous_protocol_artifacts_and_independent_receipt():
    p=read(HERE/'ROAM_NATIVE_CONTINUOUS_V1_PROTOCOL.json');r=read(HERE/'ROAM_NATIVE_CONTINUOUS_V1_RESULTS.json')
    a=read(ROOT/'feature_bank/ROAM_NATIVE_CONTINUOUS_V1_ACCEPTANCE.json')
    for rel,h in {**p['source_sha256'],**p['artifact_sha256'],**r['artifacts_sha256']}.items():assert sha(ROOT/rel)==h
    assert a['protocol_sha256']==r['protocol_sha256']==sha(HERE/'ROAM_NATIVE_CONTINUOUS_V1_PROTOCOL.json')
    assert a['result_sha256']==sha(HERE/'ROAM_NATIVE_CONTINUOUS_V1_RESULTS.json')
    assert a['verifier_sha256']==sha(HERE/'verify_roam_native_continuous_v2.py')
    assert a['protocol_verifier_sha256']==sha(HERE/'verify_roam_native_continuous_v1.py')
    assert a['independent_native_energy_FSM'] and a['independent_logits_anchor_routing_path_and_metric_equations']
    assert a['all_misses_and_unmatched_detections_retained'] and a['calibration_query_recordings_disjoint']
    assert a['maximum_independent_probability_error']<1e-12 and a['maximum_rest_threshold_error']==0
    assert a['recording_budget_blocks']==60 and a['independent_active_references']==100 and a['cells']==90
    assert a['detected_event_budget_rows']==88 and a['csv_probability_rows']==264
    assert a['primary_guards']==r['primary_guards'] and a['primary_pass']==all(r['primary_guards'].values())
    assert not any(a[k] for k in ('native_experiment_rerun','classifier_refitted','detector_refitted','physiological_boundary_ground_truth','physical_validation_proven','default_promoted','completion_proven'))


def test_geometry_diagnostic_partitions_all_references_without_rerunning_inference():
    d=read(HERE/'ROAM_NATIVE_CONTINUOUS_V1_DIAGNOSTIC.json')
    records=read(HERE/'roam_native_continuous_v1/recordings.json')
    rows,merged,summaries=audit(records)
    assert d['rows']==rows and d['multi_cue_intervals']==merged and d['summaries']==summaries
    assert len(rows)==300 and len({r['reference_id'] for r in rows})==100
    assert d['source_sha256']==sha(HERE/'audit_roam_native_continuous_v1.py')
    assert d['result_sha256']==sha(HERE/'ROAM_NATIVE_CONTINUOUS_V1_RESULTS.json')
    assert d['recordings_sha256']==sha(HERE/'roam_native_continuous_v1/recordings.json')
    for s in summaries:assert sum(s['categories'].values())==s['references']
    assert not d['native_inference_rerun'] and not d['thresholds_changed']


def test_geometry_distinguishes_assignment_conflict_long_merge_and_no_detection():
    refs=[dict(trial_id='a',start=100,end=500,class_name='open'),dict(trial_id='b',start=500,end=900,class_name='close')]
    def event(a,b):return dict(trial_id='e',start=a,end=b,labels={g:'open' for g in ('source_window','window_full','joint_full')})
    record=dict(user=19,shots=2,member='fixture',references=refs,events=[event(100,900)],matches=[dict(reference_index=0,event_index=0)])
    rows,merged,_=audit([record]);assert [r['category'] for r in rows]==['matched','miss_one_to_one_assignment_conflict']
    assert len(merged)==1 and all(r['touches_another_active_cue'] for r in rows)
    record.update(events=[event(50,1200)],matches=[])
    rows,_,_=audit([record]);assert all(r['category']=='miss_overlap_below_iou_threshold' for r in rows)
    record.update(events=[])
    rows,_,_=audit([record]);assert all(r['category']=='miss_no_detected_interval_overlap' for r in rows)


def test_current_report_index_and_event_summary_bind_native_continuous_scope():
    a=read(ROOT/'feature_bank/ROAM_NATIVE_CONTINUOUS_V1_ACCEPTANCE.json')
    current=read(ROOT/'feature_bank/CURRENT_SCIENTIFIC_CONCLUSIONS_V1.json')['roam_native_continuous']
    assert all(current[k]==v for k,v in a.items())
    index=read(ROOT/'feature_bank/delivery/INDEX.json')['roam_native_continuous_acceptance']
    assert index['sha256']==sha(ROOT/'feature_bank/ROAM_NATIVE_CONTINUOUS_V1_ACCEPTANCE.json')
    report=(ROOT/'feature_bank/REPORT.md').read_text(encoding='utf8')
    assert 'Native eight-channel autonomous cue-reference evaluation' in report
    assert 'conditional classification accuracy is not end-to-end success' in report
    table=ROOT/'feature_bank/delivery/new_bank_v3/native_continuous_detection.csv'
    with table.open(encoding='utf8',newline='') as f:rows=list(csv.DictReader(f))
    assert len(rows)==90 and len({(r['subject'],r['current_shots_per_class'],r['arm']) for r in rows})==90
    result=read(HERE/'ROAM_NATIVE_CONTINUOUS_V1_RESULTS.json')
    lookup={(c['user'],c['shots'],c['arm']):c['metrics'] for c in result['cells']}
    for row in rows:
        key=int(row['subject']),int(row['current_shots_per_class']),row['arm'];m=lookup[key]
        assert int(row['total_calibration_trials'])==6+3*key[1]
        assert row['reference_unit']=='active_gt_cue_interval_not_physiological_bout'
        assert int(row['missed'])==m['missed'] and int(row['references'])==m['references']
        assert float(row['end_to_end_success'])==m['end_to_end_success']
        assert row['conditional_log_loss']==('N/A' if m['conditional_log_loss'] is None else str(m['conditional_log_loss']))
