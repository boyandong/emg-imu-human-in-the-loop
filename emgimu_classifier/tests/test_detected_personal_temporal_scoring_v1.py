import numpy as np
import pytest
from benchmarks.new_bank_v3.detected_personal_temporal_unibo_v1 import conditional_weights, score, end_to_end


def test_active_conditional_probability_metrics_and_rest_errors_are_explicit():
    y = np.array([1, 1, 2, 3])
    weights = conditional_weights(y)
    np.testing.assert_allclose(weights, [1/6, 1/6, 1/3, 1/3])
    q = np.array([[.7, .1, .1, .1], [.1, .7, .1, .1], [.1, .1, .7, .1], [.1, .1, .1, .7]])
    r = score(y, q, weights)
    assert r['predicted_rest_count'] == 1
    assert r['accuracy'] == pytest.approx(5/6)
    assert r['active_macro_f1'] == pytest.approx((2/3+1+1)/3)
    assert r['log_loss'] == pytest.approx(-np.dot(weights, np.log(q[np.arange(4), y])))
    # Brier in this repository is the mean over all four output coordinates.
    truth = np.eye(4)[y]
    assert r['brier'] == pytest.approx(np.dot(weights, np.mean((q-truth)**2, axis=1)))
    with pytest.raises(ValueError):
        conditional_weights([1, 2])
    with pytest.raises(ValueError):
        score([0, 1, 2, 3], q, weights)


def test_end_to_end_keeps_missed_references_in_denominator():
    labels = [1, 2, 3]
    q = np.eye(4)[[1, 0, 3]]
    r = end_to_end(labels, q, [1, 1, 2, 2, 3, 3])
    assert r['success'] == pytest.approx(2/6)
    assert r['per_class']['fist'] == dict(references=2, matched=1, misses=1, correct=0, success=0.)
    assert sum(v['misses'] for v in r['per_class'].values()) == 3
    with pytest.raises(ValueError):
        end_to_end([1, 1, 2, 3], np.eye(4)[[1, 1, 2, 3]], [1, 2, 3])
