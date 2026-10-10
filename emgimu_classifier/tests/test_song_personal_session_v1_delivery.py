"""Independent arithmetic and provenance checks of native lifecycle artifacts."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'benchmarks/song_real8'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def test_native_workflow_cells_metrics_costs_and_provenance():
    protocol=HERE/'SONG_PERSONAL_SESSION_V1_PROTOCOL.json';p=json.loads(protocol.read_text(encoding='utf8'))
    r=json.loads((HERE/'SONG_PERSONAL_SESSION_V1_RESULTS.json').read_text(encoding='utf8'))
    acceptance=json.loads((ROOT/'feature_bank/SONG_PERSONAL_SESSION_ACCEPTANCE_V1.json').read_text(encoding='utf8'))
    assert sha(protocol)==r['protocol_sha256'] and len(r['source_trial_ids'])==285
    for name,digest in p['source_sha256'].items():assert sha(ROOT/name)==digest
    for block in ('source_sha256','artifact_sha256'):
        for name,digest in acceptance[block].items():assert sha(ROOT/name)==digest
    assert acceptance['checked_native_cells']==44 and acceptance['checked_trial_probabilities']==5456
    assert acceptance['maximum_probability_error']<1e-12
    assert r['source_state_immutable'] and r['personal_state_immutable']
    assert not any(r[k] for k in ('default_promoted','physical_validation_proven','completion_proven'))
    source=set(r['source_trial_ids']);personal=set(r['personal_calibration_ids']);reserved=set(r['reserved_current_ids']);evaluation=set(r['evaluation_ids'])
    assert len(personal)==len(reserved)==20 and len(evaluation)==124
    assert not (source&personal or source&reserved or source&evaluation or personal&reserved or reserved&evaluation)
    assert len(r['source_oof'])==2
    for fold in r['source_oof']:
        assert set(fold['fit_ids'])|set(fold['validation_ids'])==source
        assert not set(fold['fit_ids'])&set(fold['validation_ids'])
    assert r['selected_source_policy']==min(r['source_policy_candidates'],key=lambda c:(c['mean_source_loss'],c['n0'],c['temperature']))
    with (ROOT/r['prediction_path']).open(encoding='utf8',newline='') as f:rows=list(csv.DictReader(f))
    classes=p['classes'];truth=np.array([classes.index(c) for c in r['evaluation_labels']]);saved={}
    for cell in r['cells']:
        selected=[v for v in rows if int(v['shots'])==cell['shots'] and v['arm']==cell['arm']]
        assert [v['trial_id'] for v in selected]==r['evaluation_ids']
        assert [v['label'] for v in selected]==r['evaluation_labels']
        q=np.array([[float(v['p_'+c]) for c in classes] for v in selected]);pred=q.argmax(1);cm=np.zeros((4,4))
        for t,v in zip(truth,pred):cm[t,v]+=1
        f1=2*np.diag(cm)/(cm.sum(0)+cm.sum(1));recall=np.diag(cm)/cm.sum(1)
        assert abs(cell['macro_f1']-f1.mean())<1e-12 and abs(cell['accuracy']-np.mean(truth==pred))<1e-12
        assert abs(cell['log_loss']+np.log(np.maximum(q[np.arange(124),truth],1e-15)).mean())<1e-12
        assert abs(cell['brier']-np.mean((q-np.eye(4)[truth])**2))<1e-12
        np.testing.assert_allclose([cell['per_class_f1'][c] for c in classes],f1,atol=1e-12,rtol=0)
        np.testing.assert_allclose([cell['recall'][c] for c in classes],recall,atol=1e-12,rtol=0)
        saved[(cell['shots'],cell['arm'])]=q
        source_only=cell['arm'] in ('F0','population','uniform')
        assert cell['long_term_calibration_trials']==(0 if source_only else 20)
        assert cell['current_calibration_trials']==(4*cell['shots'] if cell['arm'].startswith('session') else 0)
    for arm in ('F0','population','uniform','personal'):
        for shots in (1,2,5):np.testing.assert_array_equal(saved[(0,arm)],saved[(shots,arm)])
    np.testing.assert_array_equal(saved[(0,'session')],saved[(0,'personal')])
    assert all(c['maximum_probability_error']<1e-12 for c in r['session_roundtrips'])


def test_current_canonical_song_rows_match_native_metrics_and_total_costs():
    r=json.loads((HERE/'SONG_PERSONAL_SESSION_V1_RESULTS.json').read_text(encoding='utf8'))
    delivery=ROOT/'feature_bank/delivery/new_bank_v3'
    with (delivery/'feature_family_results.csv').open(encoding='utf8',newline='') as f:
        family=[v for v in csv.DictReader(f) if v['run_id']=='song_personal_session_v1']
    assert len(family)==64
    for cell in r['cells']:
        row=next(v for v in family if v['feature_family']==cell['arm'] and int(v['calibration_budget'])==cell['shots'])
        for k in ('macro_f1','accuracy','log_loss','brier'):assert abs(float(row[k])-cell[k])<1e-12
        notes=json.loads(row['metadata_notes_json'])
        assert notes['actual_total_target_calibration_trials'][cell['arm']]==cell['long_term_calibration_trials']+cell['current_calibration_trials']
    with (delivery/'ablation_full_bank.csv').open(encoding='utf8',newline='') as f:
        ablations=[v for v in csv.DictReader(f) if v['run_id']=='song_personal_session_v1']
    assert len(ablations)==24
    for row in ablations:
        shots=int(row['calibration_budget']);full=next(c for c in r['cells'] if c['shots']==shots and c['arm']=='session')
        alternative=next(c for c in r['cells'] if c['shots']==shots and c['arm']==row['remaining_bank'])
        assert abs(float(row['delta_logloss'])-(alternative['log_loss']-full['log_loss']))<1e-12
        assert int(row['full_target_calibration_trials_per_user'])==int(row['remaining_target_calibration_trials_per_user'])==20+4*shots
