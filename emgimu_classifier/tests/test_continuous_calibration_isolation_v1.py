import numpy as np
import pytest
from emgimu.feature_bank.continuous_calibration_isolation_v1 import (
    numbered_trial_intervals, projected_exclusion_mask, isolate_fixed_pairs,
)


def test_counter_identity_uses_original_label_and_preserves_full_rest_edges():
    counter = np.array([0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0])
    label = np.array([6, 1, 1, 2, 2, 1, 1, 1, 6, 1, 3])
    assert numbered_trial_intervals(counter, label, user=1, day=6, posture=2) == {
        'unibo-u01-d06-p02-g02-r01': (2, 6), 'unibo-u01-d06-p02-g06-r01': (7, 10)}
    with pytest.raises(ValueError, match='Ambiguous'):
        numbered_trial_intervals([1, 1, 0, 1, 1], [1, 2, 1, 2, 1], user=1, day=6, posture=1)
    with pytest.raises(ValueError, match='exactly one'):
        numbered_trial_intervals([1, 1], [2, 6], user=1, day=6, posture=1)
    with pytest.raises(ValueError):
        numbered_trial_intervals([1, 1.5], [2, 2], user=1, day=6, posture=1)


def test_projection_keeps_fractional_centers_and_half_open_guard_boundaries():
    mask = projected_exclusion_mask(12, [(10, 15)], source_rate=500., target_rate=200., guard_source_samples=2)
    # Centers 0,2.5,... ; guarded source interval [8,17).
    np.testing.assert_array_equal(np.flatnonzero(mask), [4, 5, 6])
    exact = projected_exclusion_mask(12, [(10, 15)], source_rate=500., target_rate=200., guard_source_samples=0)
    np.testing.assert_array_equal(np.flatnonzero(exact), [4, 5])


def test_pair_exclusion_is_symmetric_without_resegmenting_or_rematching():
    mask = np.zeros(100, dtype=bool)
    mask[20:30] = True
    events = [(10, 25), (30, 40), (60, 70), (80, 90)]
    references = [(10, 20), (25, 35), (60, 70), (90, 99)]
    result = isolate_fixed_pairs(events, references, [(0, 0), (1, 1), (2, 2)], mask)
    assert result == dict(retained_events=[2, 3], retained_references=[2, 3], retained_pairs=[(2, 2)],
        direct_excluded_events=[0], direct_excluded_references=[1],
        excluded_events=[0, 1], excluded_references=[0, 1])
    # Genuine unmatched reference3 remains a miss; event3 remains a false trigger.
    with pytest.raises(ValueError, match='one-to-one'):
        isolate_fixed_pairs(events, references, [(0, 0), (0, 1)], mask)
    with pytest.raises(ValueError, match='outside'):
        isolate_fixed_pairs([(99, 101)], references, [], mask)
