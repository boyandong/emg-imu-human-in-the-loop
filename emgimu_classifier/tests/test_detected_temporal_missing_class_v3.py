import numpy as np
import pytest
from benchmarks.new_bank_v3.detected_personal_temporal_unibo_v3 import conditional_weights, score, end_to_end


def test_missing_matched_class_is_explicit_and_remains_end_to_end_miss():
    y = np.array([1, 1, 3])
    q = np.eye(4)[y]
    weights = conditional_weights(y)
    np.testing.assert_array_equal(weights, [.25, .25, .5])
    r = score(y, q, weights)
    assert r['observed_active_classes'] == [1, 3]
    assert not r['all_active_classes_observed']
    assert r['active_macro_f1'] == pytest.approx(2/3)
    totals = end_to_end(y, q, [1, 1, 2, 2, 3])
    assert totals['per_class']['fist']['matched'] == 0
    assert totals['per_class']['fist']['misses'] == 2
    assert totals['success'] == pytest.approx(3/5)
    with pytest.raises(ValueError):
        conditional_weights([])
