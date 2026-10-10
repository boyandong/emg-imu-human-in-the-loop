"""Signed per-trial metrics, true calibration costs and strict source bindings."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'benchmarks/song_real8'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def test_integrated_fixed_native_cells_guards_and_calibration_splits():
    p=json.loads((HERE/'SONG_INTEGRATED_DECISION_V1_PROTOCOL.json').read_text(encoding='utf8'))
    r=json.loads((HERE/'SONG_INTEGRATED_DECISION_V1_RESULTS.json').read_text(encoding='utf8'))
    assert sha(HERE/'SONG_INTEGRATED_DECISION_V1_PROTOCOL.json')==r['protocol_sha256']
    for name,digest in {**p['source_sha256'],**p['artifact_sha256'],**r['artifact_sha256']}.items():assert sha(ROOT/name)==digest
    previous=json.loads((HERE/'SONG_PERSONAL_SESSION_V1_RESULTS.json').read_text(encoding='utf8'))
    for field in ('evaluation_ids','personal_calibration_ids','reserved_current_ids'):assert r[field]==previous[field]
    assert not set(r['evaluation_ids'])&(set(r['personal_calibration_ids'])|set(r['reserved_current_ids']))
    last=set()
    for shots in (1,2,5):
        current=set(r['profiles'][str(shots)]['calibration_ids']);assert len(current)==4*shots and last<=current
        last=current
    with (HERE/'song_integrated_decision_v1/predictions.csv').open(encoding='utf8',newline='') as f:rows=list(csv.DictReader(f))
    assert len(rows)==6944 and len(r['cells'])==56
    for cell in r['cells']:
        selected=[v for v in rows if int(v['shots'])==cell['shots'] and v['arm']==cell['arm']]
        assert [v['trial_id'] for v in selected]==r['evaluation_ids'] and [v['label'] for v in selected]==r['evaluation_labels']
        classes=['fist','index_pinch','neutral','open_hand'];q=np.array([[float(v['p_'+c]) for c in classes] for v in selected])
        y=np.array([classes.index(v['label']) for v in selected]);pred=q.argmax(1)
        f1=[];recall={}
        for i,c in enumerate(classes):
            tp=np.sum((y==i)&(pred==i));support=np.sum(y==i);f1.append(2*tp/(support+np.sum(pred==i)));recall[c]=float(tp/support)
        np.testing.assert_allclose(cell['macro_f1'],np.mean(f1),rtol=0,atol=1e-12)
        np.testing.assert_allclose(cell['log_loss'],-np.log(np.maximum(q[np.arange(len(y)),y],1e-15)).mean(),rtol=0,atol=1e-12)
        np.testing.assert_allclose(cell['brier'],((q-np.eye(4)[y])**2).mean(),rtol=0,atol=1e-12)
        assert cell['recall']==recall and cell['long_term_calibration_trials']==20 and cell['current_calibration_trials']==4*cell['shots']
        assert cell['session_routing_enabled']==(cell['shots']>0 and cell['arm'] not in ('baseline','F7_long','F7_local','F7_blended','F7_standalone'))
    a=next(c for c in r['cells'] if c['shots']==5 and c['arm']=='baseline');b=next(c for c in r['cells'] if c['shots']==5 and c['arm']=='F7_F8_F9_structural')
    expected=dict(lower_log_loss=b['log_loss']<a['log_loss'],lower_brier=b['brier']<a['brier'],nonworse_macro_f1=b['macro_f1']>=a['macro_f1'],nonworse_all_class_recall=all(b['recall'][c]>=a['recall'][c] for c in classes))
    assert expected==r['primary_guards'] and all(expected.values())==r['primary_pass']
    assert not r['default_promoted'] and not r['physical_validation_proven'] and not r['completion_proven']


def test_no_fit_integrated_acceptance_binds_every_artifact_and_arithmetic_oracle():
    a=json.loads((ROOT/'feature_bank/SONG_INTEGRATED_DECISION_ACCEPTANCE_V1.json').read_text(encoding='utf8'))
    r=json.loads((HERE/'SONG_INTEGRATED_DECISION_V1_RESULTS.json').read_text(encoding='utf8'))
    for name,digest in a['source_sha256'].items():assert sha(ROOT/name)==digest
    assert a['artifact_sha256']==r['artifact_sha256'] and a['checked_cells']==56 and a['checked_trial_probabilities']==6944
    assert a['maximum_probability_error']<1e-12
    assert all(a[k] for k in ('independent_covariance_prototype_oracle','independent_generalized_eigenvalue_oracle','independent_session_routing_oracle','previous_baseline_parity','source_and_profiles_immutable'))
    assert a['primary_pass']==r['primary_pass'] and not a['completion_proven']


def test_canonical_integrated_cells_and_provider_branch_removals():
    r=json.loads((HERE/'SONG_INTEGRATED_DECISION_V1_RESULTS.json').read_text(encoding='utf8'))
    def rows(name):
        with (ROOT/'feature_bank/delivery/new_bank_v3'/name).open(encoding='utf8',newline='') as f:
            return [v for v in csv.DictReader(f) if v['run_id']=='song_integrated_decision_v1']
    family=rows('feature_family_results.csv');ablations=rows('ablation_full_bank.csv');pairs=rows('conditional_incremental.csv')
    assert len(family)==56 and len(ablations)==36 and len(pairs)==64
    lookup={(int(v['calibration_budget']),v['feature_family']):v for v in family}
    for c in r['cells']:
        row=lookup[(c['shots'],c['arm'])]
        for metric in ('macro_f1','log_loss','brier'):np.testing.assert_allclose(float(row[metric]),c[metric],rtol=0,atol=1e-12)
        assert json.loads(row['metadata_notes_json'])['actual_total_target_calibration_trials'][c['arm']]==20+4*c['shots']
    for row in ablations:
        shots=int(row['calibration_budget']);full=lookup[(shots,row['full_bank'])];alt=lookup[(shots,row['remaining_bank'])]
        assert int(row['full_target_calibration_trials_per_user'])==int(row['remaining_target_calibration_trials_per_user'])==20+4*shots
        np.testing.assert_allclose(float(row['delta_logloss']),float(alt['log_loss'])-float(full['log_loss']),rtol=0,atol=1e-12)
    assert {v['removed_provider'] for v in ablations}>={'F7_anchor','F8_router','F9_raw'}
