"""Independent known-path checks for the optional complete-bout F5c block."""
import pickle

import numpy as np
import pytest

from emgimu.feature_bank import CompleteSequenceBatch, FeatureBatch, PathSignatureFamily


def complete(points):
    return CompleteSequenceBatch(np.asarray(points, dtype=float)[None], 2.,
        durations_seconds=np.array([2.]), full_coverage=True)


def test_three_point_polygon_order_two_and_no_time_channel():
    # L2-normalized envelope vertices: (1,0), (0,1), (s,s).
    s = np.sqrt(0.5)
    points = np.array([[4., 0.], [0., 4.], [4., 4.]])
    first = np.array([-1., 1.])
    second = np.array([s, s-1.])
    expected_one = first + second
    expected_two = (np.outer(first, first)/2 + np.outer(first, second)
                    + np.outer(second, second)/2)
    family = PathSignatureFamily().fit(complete(points))
    before = pickle.dumps(family)
    result = family.transform(complete(points))[0]
    np.testing.assert_allclose(result[:2], expected_one, rtol=0, atol=1e-7)
    np.testing.assert_allclose(result[2:].reshape(2, 2), expected_two, rtol=0, atol=1e-7)
    assert result.size == 2 + 2**2
    assert before == pickle.dumps(family)
    # A repeated vertex changes sampling but not the geometric path.
    repeated = np.insert(points, 1, points[0], axis=0)
    np.testing.assert_allclose(family.transform(complete(repeated))[0], result, atol=1e-7)
    np.testing.assert_allclose(family.transform(complete(points*17))[0], result, atol=1e-7)


def test_short_window_and_unverified_coverage_rejected():
    points = np.array([[1., 0.], [0., 1.], [1., 1.]])
    with pytest.raises(ValueError, match='complete sequences'):
        PathSignatureFamily().fit(FeatureBatch(points[None], 2.))
    family = PathSignatureFamily().fit(complete(points))
    with pytest.raises(ValueError, match='complete sequences'):
        family.transform(FeatureBatch(points[None], 2.))
    with pytest.raises(ValueError, match='duration'):
        complete_short = CompleteSequenceBatch(points[None], 2.,
            durations_seconds=np.array([0.5]), full_coverage=True)
        family.transform(complete_short)
