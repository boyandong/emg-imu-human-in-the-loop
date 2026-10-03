"""Read back the single fresh-user result from saved trial probabilities."""

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, log_loss

from emgimu.datasets.epn612 import GESTURES


ROOT = Path(__file__).resolve().parents[1] / 'benchmarks/new_bank_v3/F7_AFFINE_FRESH'


def _read(name):
    with (ROOT / name).open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def test_fresh_artifact_hashes_trial_isolation_and_primary_metrics():
    report = json.loads((ROOT / 'results.json').read_text(encoding='utf-8'))
    assert tuple(report['fresh_users']) == tuple(range(22, 32))
    assert report['validation_replay_maximum_probability_error'] <= 1e-10
    for name, expected in report['output_sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
    assert hashlib.sha256(Path(__file__).resolve().parents[1].joinpath(
        'benchmarks/new_bank_v3/f7_affine_fresh_epn.py').read_bytes()).hexdigest() == report['script_sha256']

    selections = _read('calibration_trial_ids.csv')
    predictions = _read('trial_predictions.csv')
    assert len(predictions) == 4020
    by_cal = defaultdict(set)
    by_eval = defaultdict(set)
    by_user = defaultdict(set)
    for row in selections:
        key = (int(row['subject']), int(row['shots_per_class']))
        assert row['trial_id'] not in by_cal[key]
        by_cal[key].add(row['trial_id'])
        by_user[int(row['subject'])].add(row['trial_id'])
    for row in predictions:
        key = (int(row['subject']), int(row['shots_per_class']))
        assert row['trial_id'] not in by_eval[key]
        by_eval[key].add(row['trial_id'])
        by_user[int(row['subject'])].add(row['trial_id'])
        for prefix in ('p_core_', 'p_f7_'):
            p = np.asarray([float(row[prefix + name]) for name in GESTURES])
            assert np.isfinite(p).all() and np.all(p >= 0)
            np.testing.assert_allclose(p.sum(), 1., atol=1e-9)
    assert all(len(by_user[user]) == 150 for user in range(22, 32))
    for user in range(22, 32):
        for budget in (1, 2, 5):
            key = user, budget
            assert len(by_cal[key]) == 6 * budget
            assert len(by_eval[key]) == 150 - 6 * budget
            assert not by_cal[key] & by_eval[key]
            assert by_cal[key] | by_eval[key] == by_user[user]
        assert by_cal[user, 1] < by_cal[user, 2] < by_cal[user, 5]

    heldout = [r for r in predictions if int(r['shots_per_class']) == 5]
    y = np.asarray([int(r['true_label']) for r in heldout])
    core = np.asarray([[float(r['p_core_' + name]) for name in GESTURES] for r in heldout])
    f7 = np.asarray([[float(r['p_f7_' + name]) for name in GESTURES] for r in heldout])
    mix = (core + f7) / 2
    uniform = (core + 1. / len(GESTURES)) / 2
    primary = report['primary_five_shot']
    np.testing.assert_allclose(primary['delta_logloss'], log_loss(y, core, labels=range(6)) -
                               log_loss(y, mix, labels=range(6)), atol=1e-12)
    np.testing.assert_allclose(primary['delta_logloss_over_uniform'],
                               log_loss(y, uniform, labels=range(6)) -
                               log_loss(y, mix, labels=range(6)), atol=1e-12)
    np.testing.assert_allclose(primary['delta_macro_f1'],
                               f1_score(y, mix.argmax(axis=1), labels=range(6), average='macro') -
                               f1_score(y, core.argmax(axis=1), labels=range(6), average='macro'),
                               atol=1e-12)
    assert primary['conjunction_passed'] and primary['user_logloss_wins'] == 7
