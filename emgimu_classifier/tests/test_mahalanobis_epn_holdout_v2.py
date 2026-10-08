import json
from pathlib import Path
import numpy as np
from scipy.special import softmax
from benchmarks.new_bank_v3.mahalanobis_epn_budget_v1 import sha
from emgimu.feature_bank.epn_study import _metrics

HERE = Path(__file__).resolve().parents[1] / 'benchmarks/new_bank_v3'


def test_frozen_holdout_provenance_nested_trials_and_no_source_leakage():
    p = json.loads((HERE / 'MAHALANOBIS_EPN_HOLDOUT_V2_PROTOCOL.json').read_text())
    r = json.loads((HERE / 'MAHALANOBIS_EPN_HOLDOUT_V2_RESULTS.json').read_text())
    assert sha(HERE / 'MAHALANOBIS_EPN_HOLDOUT_V2_PROTOCOL.json') == r['protocol_sha256']
    for name, expected in p['source_sha256'].items():
        assert sha(HERE.parents[1] / name) == expected
    assert p['target_users'] == list(range(32, 42))
    assert not set(p['source_users']) & set(p['target_users'])
    assert len(r['blocks']) == 20 and r['source_state_immutable'] and not r['default_promoted']
    assert len(r['source_trial_ids']) == len(set(r['source_trial_ids']))
    for user in p['target_users']:
        a, b = [block for block in r['blocks'] if block['user'] == user]
        assert (a['shots'], b['shots']) == (10, 20)
        assert a['evaluation_ids'] == b['evaluation_ids']
        assert set(a['calibration_ids']) <= set(b['calibration_ids'])
        for block in (a, b):
            cal, evaluation = block['calibration_ids'], block['evaluation_ids']
            assert len(cal) == 6 * block['shots'] == len(set(cal))
            assert len(evaluation) == len(set(evaluation)) == 30
            assert not set(cal) & set(evaluation)
            assert not (set(cal) | set(evaluation)) & set(r['source_trial_ids'])
            assert all(f':user{user}:' in trial for trial in cal + evaluation)


def test_independent_shrunk_quadratic_distances_probabilities_and_pooled_scores():
    r = json.loads((HERE / 'MAHALANOBIS_EPN_HOLDOUT_V2_RESULTS.json').read_text())
    for block in r['blocks']:
        cal, labels, target = map(np.asarray, (block['calibration_features'], block['calibration_labels'], block['evaluation_features']))
        direct = []
        euclidean = []
        for label in range(6):
            points = cal[labels == label]
            center = points.mean(0)
            centered = points - center
            covariance = centered.T @ centered / (len(points) - 1)
            scale = np.trace(covariance) / 8
            covariance = .8 * covariance + (.2 * scale + max(scale * 1e-8, 1e-10)) * np.eye(8)
            delta = target - center
            direct.append(np.sqrt(np.maximum(np.einsum('ni,ij,nj->n', delta, np.linalg.inv(covariance), delta), 0)))
            euclidean.append(np.linalg.norm(delta, axis=1))
        for name, values in (('mahalanobis', direct), ('euclidean', euclidean)):
            distance = np.stack(values, axis=1)
            np.testing.assert_allclose(block['distances'][name], distance, atol=1e-5, rtol=2e-6)
            np.testing.assert_allclose(block['probabilities'][name],
                                       softmax(-np.asarray(block['distances'][name]) / block['scales'][name], axis=1), atol=1e-12)
    for shots in (10, 20):
        blocks = [b for b in r['blocks'] if b['shots'] == shots]
        truth = np.concatenate([b['labels'] for b in blocks])
        for name in ('euclidean', 'mahalanobis'):
            probability = np.concatenate([b['probabilities'][name] for b in blocks])
            actual = _metrics(truth, probability, np.ones(len(truth)))
            for metric in ('macro_f1', 'log_loss', 'brier', 'accuracy'):
                assert abs(actual[metric] - r['scores'][str(shots)][name][metric]) < 1e-12


def test_calibration_exposure_costs_preserve_unknown_physical_times():
    burden = json.loads((HERE/'MAHALANOBIS_EPN_HOLDOUT_V2_BURDEN.json').read_text())
    result = json.loads((HERE/'MAHALANOBIS_EPN_HOLDOUT_V2_RESULTS.json').read_text())
    assert not burden['few_second_calibration_proven'] and not burden['physical_wall_time_proven']
    for path, expected in burden['source_sha256'].items():
        assert sha(HERE.parents[1]/path) == expected
    assert len(burden['records']) == 20
    assert {(r['subject'],r['shots_per_class']) for r in burden['records']} == {(b['user'],b['shots']) for b in result['blocks']}
    for row in burden['records']:
        budget = row['shots_per_class']
        assert row['used_calibration_trials'] == budget*6
        assert row['used_calibration_windows'] == row['used_calibration_trials']*4
        assert row['window_samples'] == 40 and row['sample_rate_hz'] == 200
        assert row['used_signal_seconds'] == row['used_calibration_windows']*40/200 == budget*4.8
        assert row['used_trials_full_recording_seconds'] >= row['used_signal_seconds']
        assert row['reserved_calibration_trials'] == 120 and row['reserved_signal_seconds'] == 96
        assert all(row[key] == 'N/A' for key in ('hardware_setup_seconds','guided_prompt_rest_seconds','device_wall_time_seconds'))
