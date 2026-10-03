"""Frozen paired-trial readback for the new document-exact F5 candidate."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmarks.grabmyo_crossday import run as grab


ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'benchmarks' / 'new_bank_v3'
PARENT = ROOT / 'benchmarks' / 'new_bank_v2'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_f5_temporal_saved_predictions_and_no_default_promotion():
    protocol_path = HERE / 'F5_TEMPORAL_GRAB_PROTOCOL.json'
    protocol = json.loads(protocol_path.read_text())
    result = json.loads((HERE / 'F5_TEMPORAL_GRAB_RESULTS.json').read_text())
    path = HERE / 'F5_TEMPORAL_GRAB_PREDICTIONS.csv'
    assert result['protocol_sha256'] == sha(protocol_path)
    assert result['prediction_sha256'] == sha(path)
    assert protocol['parent_prediction_sha256'] == sha(PARENT / 'GRAB_USER_PREDICTIONS.csv')
    assert result['f0v2_parent_replay_max_abs_error'] == 0
    assert result['feature_dimensions'] == {'F0v2': 48, 'F5temporal': 57}
    assert not set(result['source_trial_ids']) & set(result['validation_trial_ids'])
    assert not set(result['source_trial_ids']) & set(result['final_trial_ids'])
    with path.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == result['prediction_rows'] == 224
    classes = np.asarray(grab.GESTURES)
    for arm in protocol['arms']:
        for phase in ('validation', 'final'):
            selected = [row for row in rows if row['arm'] == arm and row['phase'] == phase]
            assert len(selected) == 56 == len({row['trial_id'] for row in selected})
            probabilities = np.array([[float(row[f'p_{label}']) for label in classes]
                                      for row in selected])
            score = grab.score(np.array([int(row['gesture']) for row in selected]),
                               probabilities, classes,
                               np.array([int(row['subject']) for row in selected]))
            np.testing.assert_allclose(probabilities.sum(axis=1), 1., atol=1e-12)
            for metric in ('macro_f1', 'log_loss', 'brier', 'minimum_subject_macro_f1'):
                np.testing.assert_allclose(score[metric], result['scores'][arm][phase][metric],
                                           atol=1e-12)
    for phase in ('validation', 'final'):
        base = result['scores']['F0v2'][phase]
        candidate = result['scores']['F0v2+F5temporal'][phase]
        assert candidate['macro_f1'] < base['macro_f1']
        assert candidate['log_loss'] > base['log_loss']
