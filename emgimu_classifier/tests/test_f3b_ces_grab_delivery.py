"""Read back the isolated public CES comparison without refitting a model."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmarks.grabmyo_crossday import run as grab


ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'benchmarks' / 'new_bank_v3'
PARENT = ROOT / 'benchmarks' / 'new_bank_v2'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_f3b_ces_saved_probabilities_have_frozen_parent_and_recomputed_scores():
    protocol = json.loads((HERE / 'F3B_CES_GRAB_PROTOCOL.json').read_text())
    result = json.loads((HERE / 'F3B_CES_GRAB_RESULTS.json').read_text())
    path = HERE / 'F3B_CES_GRAB_PREDICTIONS.csv'
    assert result['protocol_sha256'] == digest(HERE / 'F3B_CES_GRAB_PROTOCOL.json')
    assert result['prediction_sha256'] == digest(path)
    assert protocol['parent_prediction_sha256'] == digest(PARENT / 'GRAB_USER_PREDICTIONS.csv')
    assert result['f0v2_parent_replay_max_abs_error'] == 0
    assert not (set(result['source_trial_ids']) & set(result['validation_trial_ids']))
    assert not (set(result['source_trial_ids']) & set(result['final_trial_ids']))
    classes = np.asarray(grab.GESTURES)
    with path.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == result['prediction_rows'] == 224
    for arm in protocol['arms']:
        for phase in ('validation', 'final'):
            selected = [row for row in rows if row['arm'] == arm and row['phase'] == phase]
            assert len(selected) == 56
            assert len({row['trial_id'] for row in selected}) == 56
            y = np.array([int(row['gesture']) for row in selected])
            users = np.array([int(row['subject']) for row in selected])
            probabilities = np.array([[float(row[f'p_{label}']) for label in classes]
                                      for row in selected])
            np.testing.assert_allclose(probabilities.sum(axis=1), 1., atol=1e-12)
            rescored = grab.score(y, probabilities, classes, users)
            for metric in ('macro_f1', 'log_loss', 'brier', 'minimum_subject_macro_f1'):
                np.testing.assert_allclose(rescored[metric], result['scores'][arm][phase][metric],
                                           atol=1e-12)
    for phase in ('validation', 'final'):
        for metric in ('macro_f1', 'log_loss'):
            old = result['scores']['F0v2'][phase][metric]
            candidate = result['scores']['F0v2+F3bCES'][phase][metric]
            assert (candidate < old) if metric == 'macro_f1' else (candidate > old)
