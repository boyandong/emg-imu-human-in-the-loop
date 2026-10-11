import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'benchmarks/new_bank_v3'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def test_document_reliability_source_selection_is_bound_and_has_no_target_fitting():
    p=read(HERE/'ROAM_DOCUMENT_RELIABILITY_V3_PROTOCOL.json');s=read(HERE/'ROAM_DOCUMENT_RELIABILITY_V3_SOURCE_RESULTS.json');t=read(HERE/'ROAM_DOCUMENT_RELIABILITY_V3_TARGET_RESULTS.json')
    for rel,h in {**p['source_sha256'],**p['source_artifact_sha256'],**p['target_artifact_sha256'],**s['artifacts_sha256'],**t['artifacts_sha256']}.items():assert sha(ROOT/rel)==h
    assert s['protocol_sha256']==t['protocol_sha256']==sha(HERE/'ROAM_DOCUMENT_RELIABILITY_V3_PROTOCOL.json')
    assert len(s['blocks'])==54 and len(s['candidates'])==16 and s['source_query_trials']==324 and s['source_budget_rows']==972
    selected=sorted(s['candidates'],key=lambda c:(c['mean_log_loss'],c['mean_brier'],c['index']))[0]
    assert selected['config']==s['selected_config'] and selected['index']==s['selected_candidate']
    assert not any(s[k] for k in ('existing_classifiers_refitted','target_arrays_read','source_scores_are_unbiased'))
    assert t['source_policy_committed_before_target'] and not t['target_fitted_or_selected'] and not t['target_query_feature_inference_repeated']


def test_exact_reliability_composition_retains_same180_native_queries_and_calibration_costs():
    r=read(HERE/'ROAM_DOCUMENT_RELIABILITY_V3_TARGET_RESULTS.json');old=read(HERE/'ROAM_NATIVE_JOINT_V1_RESULTS.json')
    lookup={s['user']:s for s in old['subjects']}
    assert len(r['blocks'])==30 and len(r['cells'])==210 and len(r['aggregates'])==63
    for b in r['blocks']:
        prior=lookup[b['user']]
        assert b['query_ids']==prior['query_ids'] and b['query_labels']==prior['query_labels'] and b['query_recording_ids']==prior['query_recordings']
        assert b['long_calibration_trials']==6 and b['current_calibration_trials']==3*b['shots']
        assert len(b['weights'])==len(b['routed_weights'])==7
        assert abs(sum(b['weights'])-1)<1e-12 and min(b['weights'])>=0
    for c in r['cells']:
        assert c['trials']==18 and c['target_calibration_trials']==(0 if c['arm']=='source_window' else 6+3*c['shots'])


def test_independent_exact_weight_and_composition_acceptance_retains_actual_guards():
    a=read(ROOT/'feature_bank/ROAM_DOCUMENT_RELIABILITY_V3_ACCEPTANCE.json');s=read(ROOT/'feature_bank/ROAM_DOCUMENT_RELIABILITY_V3_SOURCE_ACCEPTANCE.json')
    t=read(HERE/'ROAM_DOCUMENT_RELIABILITY_V3_TARGET_RESULTS.json')
    assert a['verifier_sha256']==s['verifier_sha256']==sha(HERE/'verify_roam_document_reliability_v3.py')
    assert s['independent_exact_hierarchy_and_source_selection'] and a['independent_calibration_hierarchy_routing_composition_and_metrics']
    assert s['maximum_independent_calibration_feature_error']<1e-12 and a['maximum_independent_probability_error']<1e-12
    assert a['primary_guards']==t['primary_guards'] and a['primary_pass']==all(all(v.values()) for v in a['primary_guards'].values())
    assert not any(a[k] for k in ('target_query_feature_inference_repeated','target_fitted_or_selected','native_experiment_rerun','default_promoted','physical_validation_proven','completion_proven'))


def test_current_document_reliability_delivery_preserves_all210_cells():
    a=read(ROOT/'feature_bank/ROAM_DOCUMENT_RELIABILITY_V3_ACCEPTANCE.json')
    c=read(ROOT/'feature_bank/CURRENT_SCIENTIFIC_CONCLUSIONS_V1.json')['roam_document_reliability']
    assert all(c[k]==v for k,v in a.items())
    index=read(ROOT/'feature_bank/delivery/INDEX.json')['roam_document_reliability_acceptance']
    assert index['sha256']==sha(ROOT/'feature_bank/ROAM_DOCUMENT_RELIABILITY_V3_ACCEPTANCE.json')
    assert 'Native document-exact reliability with source-user CV' in (ROOT/'feature_bank/REPORT.md').read_text(encoding='utf8')
    with (ROOT/'feature_bank/delivery/new_bank_v3/document_reliability.csv').open(encoding='utf8',newline='') as f:rows=list(csv.DictReader(f))
    assert len(rows)==210 and len({(r['subject'],r['current_shots_per_class'],r['arm']) for r in rows})==210
    native=read(HERE/'ROAM_DOCUMENT_RELIABILITY_V3_TARGET_RESULTS.json');lookup={(c['user'],c['shots'],c['arm']):c for c in native['cells']}
    for row in rows:
        cell=lookup[int(row['subject']),int(row['current_shots_per_class']),row['arm']]
        assert int(row['total_calibration_trials'])==cell['target_calibration_trials'] and float(row['log_loss'])==cell['log_loss']
