"""Independently recover every cell and signed guard from per-trial predictions."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'benchmarks/song_real8'
RESULT=HERE/'SONG_MATCHED_NORMALIZATION_V1_RESULTS.json'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(labels,q,classes):
    truth=np.array([classes.index(c) for c in labels]);prediction=q.argmax(1)
    confusion=np.zeros((len(classes),len(classes)),int);np.add.at(confusion,(truth,prediction),1)
    tp=np.diag(confusion);denominator=confusion.sum(0)+confusion.sum(1)
    f1=np.divide(2*tp,denominator,out=np.zeros(len(classes),float),where=denominator>0)
    return dict(macro_f1=float(f1.mean()),accuracy=float((truth==prediction).mean()),
        log_loss=float(-np.log(np.maximum(q[np.arange(len(truth)),truth],1e-15)).mean()),
        brier=float(((q-np.eye(len(classes))[truth])**2).mean()),
        recall=dict(zip(classes,np.divide(tp,confusion.sum(1),out=np.zeros(len(classes),float),where=confusion.sum(1)>0))))


def test_protocol_source_reservations_and_independent_native_metrics():
    r=json.loads(RESULT.read_text(encoding='utf8'));p=json.loads((HERE/'SONG_MATCHED_NORMALIZATION_V1_PROTOCOL.json').read_text(encoding='utf8'))
    assert sha(HERE/'SONG_MATCHED_NORMALIZATION_V1_PROTOCOL.json')==r['protocol_sha256']
    for name,digest in p['source_sha256'].items():assert sha(ROOT/name)==digest
    for name,digest in r['artifact_sha256'].items():assert sha(ROOT/name)==digest
    fit=set(r['source_model_fit_ids']);cal=set(t for ids in r['source_normalization_ids'].values() for t in ids)
    assert len(fit)==245 and len(cal)==40 and not fit&cal and fit|cal==set(r['source_all_ids'])
    for mode,folds in r['source_oof'].items():
        assert len(folds)==2
        for f in folds:
            assert not set(f['fit_ids'])&set(f['long_calibration_ids'])
            assert not (set(f['fit_ids'])|set(f['long_calibration_ids']))&set(f['evaluation_ids'])
            previous=set()
            for n in (1,2,5):
                current=set(f['current_calibration_ids'][str(n)])
                assert len(current)==4*n and previous<=current and not current&set(f['evaluation_ids'])
                previous=current
            for values in f['probabilities'].values():
                for q in values.values():np.testing.assert_allclose(np.asarray(q).sum(1),1.,rtol=0,atol=1e-12)
        metadata=r['source'][mode]
        assert metadata['selected_policy']==min(metadata['policy_candidates'],key=lambda c:(c['mean_source_loss'],c['n0'],c['temperature']))
        assert len(metadata['policy_candidates'])==9
    old=json.loads((HERE/'SONG_PERSONAL_SESSION_V1_RESULTS.json').read_text(encoding='utf8'))
    for field in ('evaluation_ids','personal_calibration_ids','reserved_current_ids'):assert r[field]==old[field]
    assert not set(r['evaluation_ids'])&(set(r['personal_calibration_ids'])|set(r['reserved_current_ids']))
    with (HERE/'song_matched_normalization_v1/predictions.csv').open(encoding='utf8',newline='') as f:rows=list(csv.DictReader(f))
    assert len(r['cells'])==80 and len(rows)==9920
    classes=list(p['classes']);computed={}
    for c in r['cells']:
        records=[v for v in rows if v['mode']==c['mode'] and int(v['shots'])==c['shots'] and v['arm']==c['arm']]
        assert [v['trial_id'] for v in records]==r['evaluation_ids']
        assert [v['label'] for v in records]==r['evaluation_labels']
        q=np.array([[float(v['p_'+name]) for name in classes] for v in records])
        m=metrics(r['evaluation_labels'],q,classes);computed[(c['mode'],c['shots'],c['arm'])]=m
        for key in ('macro_f1','accuracy','log_loss','brier'):np.testing.assert_allclose(m[key],c[key],rtol=0,atol=1e-12)
        for name in classes:assert m['recall'][name]==c['recall'][name]
        assert c['trials']==124
        assert c['long_term_calibration_trials']==(0 if c['mode']=='raw' and c['arm'] in ('F0','population') else 20)
        expected=4*c['shots'] if c['arm'].startswith('session') or (c['mode']=='normalized' and c['arm'] in ('F0','population')) else 0
        assert c['current_calibration_trials']==expected
    a=computed[('raw',5,'session')];b=computed[('normalized',5,'session')]
    guards=dict(lower_log_loss=b['log_loss']<a['log_loss'],lower_brier=b['brier']<a['brier'],
        nonworse_macro_f1=b['macro_f1']>=a['macro_f1'],nonworse_all_class_recall=all(b['recall'][c]>=a['recall'][c] for c in classes))
    assert guards==r['primary_guards'] and all(guards.values())==r['primary_pass']
    assert not r['default_promoted'] and not r['physical_validation_proven'] and not r['completion_proven']
    assert len(r['roundtrips'])==6 and all(v['maximum_probability_error']==0. for v in r['roundtrips'])


def test_no_fit_acceptance_binds_complete_matched_artifacts():
    r=json.loads(RESULT.read_text(encoding='utf8'))
    a=json.loads((ROOT/'feature_bank/SONG_MATCHED_NORMALIZATION_ACCEPTANCE_V1.json').read_text(encoding='utf8'))
    for name,digest in a['source_sha256'].items():assert sha(ROOT/name)==digest
    assert a['artifact_sha256']==r['artifact_sha256']
    assert a['checked_cells']==80 and a['checked_trial_probabilities']==9920
    assert a['maximum_probability_error']<1e-12 and a['maximum_normalization_error']==0.
    assert a['source_model_fit_trials']==245 and a['source_normalization_trials']==40
    assert a['source_state_immutable'] and a['personal_state_immutable']
    assert a['primary_pass']==r['primary_pass'] and not a['completion_proven']


def test_canonical_comparisons_retain_every_matched_cell_and_true_calibration_cost():
    r=json.loads(RESULT.read_text(encoding='utf8'));delivery=ROOT/'feature_bank/delivery/new_bank_v3'
    def rows(name):
        with (delivery/name).open(encoding='utf8',newline='') as f:
            return [v for v in csv.DictReader(f) if v['run_id']=='song_matched_normalization_v1']
    family=rows('feature_family_results.csv');ablations=rows('ablation_full_bank.csv');pairs=rows('conditional_incremental.csv')
    assert len(family)==120 and len(ablations)==48 and len(pairs)==216
    lookup={(int(v['calibration_budget']),v['feature_family']):v for v in family}
    for cell in r['cells']:
        name=cell['mode']+'_'+cell['arm'];row=lookup[(cell['shots'],name)]
        assert int(row['evaluation_trials'])==124
        for metric in ('macro_f1','accuracy','log_loss','brier'):np.testing.assert_allclose(float(row[metric]),cell[metric],rtol=0,atol=1e-12)
        notes=json.loads(row['metadata_notes_json'])
        assert notes['actual_total_target_calibration_trials'][name]==cell['long_term_calibration_trials']+cell['current_calibration_trials']
    for row in ablations:
        shots=int(row['calibration_budget']);full=lookup[(shots,row['full_bank'])];remaining=lookup[(shots,row['remaining_bank'])]
        np.testing.assert_allclose(float(row['delta_logloss']),float(remaining['log_loss'])-float(full['log_loss']),rtol=0,atol=1e-12)
        assert int(row['full_target_calibration_trials_per_user'])==int(row['remaining_target_calibration_trials_per_user'])==20+4*shots
    cross=[v for v in pairs if v['core_bank'].startswith('raw_') and v['added_family'].startswith('normalized_')]
    assert len(cross)==16
