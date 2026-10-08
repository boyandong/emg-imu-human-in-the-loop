import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT/'benchmarks/new_bank_v3'

def test_cost_curve_preserves_frozen_predictions_and_realistic_requirements():
    curve = json.loads((HERE/'MAHALANOBIS_EPN_HOLDOUT_V2_COST_CURVE.json').read_text())
    result = json.loads((HERE/'MAHALANOBIS_EPN_HOLDOUT_V2_RESULTS.json').read_text())
    for path,digest in curve['source_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
    for name,digest in curve['figure_sha256'].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest
    rows = curve['rows']
    keys = [(r['subject'],r['shots_per_class'],r['method']) for r in rows]
    assert len(keys)==len(set(keys))==40
    for block in result['blocks']:
        for method in ('euclidean','mahalanobis'):
            row = next(r for r in rows if (r['subject'],r['shots_per_class'],r['method'])==(block['user'],block['shots'],method))
            probability=np.asarray(block['probabilities'][method]); y=np.asarray(block['labels'])
            assert np.isclose(row['macro_f1'],f1_score(y,probability.argmax(1),labels=range(6),average='macro',zero_division=0),atol=1e-12)
            assert np.isclose(row['log_loss'],log_loss(y,probability,labels=range(6)),atol=1e-12)
            assert row['calibration_trials']==6*block['shots']
            assert row['signal_seconds']==4.8*block['shots']
            assert row['full_recording_seconds']>6*row['signal_seconds']
    for shots in (10,20):
        blocks=[b for b in result['blocks'] if b['shots']==shots]
        y=np.concatenate([b['labels'] for b in blocks])
        for method in ('euclidean','mahalanobis'):
            p=np.concatenate([b['probabilities'][method] for b in blocks])
            assert np.isclose(curve['pooled_scores'][str(shots)][method]['macro_f1'],f1_score(y,p.argmax(1),labels=range(6),average='macro',zero_division=0))
    requirements=curve['calibration_requirements']
    assert requirements['all_task_gestures_required'] and requirements['required_native_gesture_classes']==6
    for key in ('each_target_session','target_force_required','target_posture_required','physical_session_seconds'):
        assert requirements[key].startswith('N/A:')
    assert not any(curve[k] for k in ('few_second_calibration_proven','device_efficacy_proven','default_promoted'))
