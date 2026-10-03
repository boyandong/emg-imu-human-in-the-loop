"""Independent F7 affine-distance geometry, leakage and trial-mass oracles."""
import pickle

import numpy as np
import pytest

from emgimu.feature_bank.affine_spd_anchor import (
    AffineSpdPrototypeAnchor, affine_spd_distance, document_spd_matrices,
)


def test_affine_distance_known_diagonal_and_nondiagonal_congruence():
    calibration = np.stack([np.eye(2), np.diag([4., 1.])])
    query = np.diag([2., 1.])[None]
    anchor = AffineSpdPrototypeAnchor().fit(calibration, ['neutral', 'fist'], ['n', 'f'])
    ids, distances = anchor.transform(query, ['heldout'])
    assert ids.tolist() == ['heldout']
    np.testing.assert_allclose(distances[0], [np.log(2.), np.log(2.)], atol=1e-12)
    assert anchor.feature_names == ('F7.affine_spd.distance.fist',
                                    'F7.affine_spd.distance.neutral')
    # Affine invariance is stronger than rotation invariance: shear and
    # anisotropic scale must leave every matrix-log eigenvalue unchanged.
    change = np.array([[2., .7], [0., .5]])
    transform = lambda m: change @ m @ change.T
    changed = AffineSpdPrototypeAnchor().fit(
        np.stack([transform(m) for m in calibration]), ['neutral', 'fist'], ['n', 'f'])
    _, changed_distances = changed.transform(transform(query[0])[None], ['heldout'])
    np.testing.assert_allclose(changed_distances, distances, atol=1e-12)
    assert affine_spd_distance(np.eye(2), np.diag([1e-12, 1.])) == pytest.approx(12 * np.log(10.))


def test_equal_trial_mass_heldout_immutability_and_bad_inputs():
    # Trial a has four windows; trial b has one. Neutral prototype = (I+9I)/2.
    matrices = np.stack([np.eye(2)] * 4 + [9 * np.eye(2), 4 * np.eye(2)])
    ids = ['a'] * 4 + ['b', 'c']
    labels = ['neutral'] * 5 + ['fist']
    anchor = AffineSpdPrototypeAnchor().fit(matrices, labels, ids)
    np.testing.assert_allclose(anchor.prototypes_[1], 5 * np.eye(2))
    before = pickle.dumps(anchor)
    trial_ids, distance = anchor.transform(2 * np.eye(2)[None], ['test'])
    assert trial_ids.tolist() == ['test']
    np.testing.assert_allclose(distance[0, 1], np.sqrt(2) * np.log(5 / 2))
    assert pickle.dumps(anchor) == before
    with pytest.raises(ValueError, match='calibration trials'):
        anchor.transform(np.eye(2)[None], ['a'])
    with pytest.raises(ValueError, match='positive definite'):
        AffineSpdPrototypeAnchor().fit(np.stack([np.eye(2), np.zeros((2, 2))]),
                                       ['neutral', 'fist'], ['n', 'f'])
    with pytest.raises(ValueError, match='mixed calibration labels'):
        AffineSpdPrototypeAnchor().fit(np.stack([np.eye(2), np.eye(2), 2*np.eye(2)]),
                                       ['neutral', 'fist', 'fist'], ['a', 'a', 'b'])


def test_document_matrix_bridge_is_centered_and_strictly_spd():
    zero = np.zeros((1, 12, 8))
    cov = document_spd_matrices(zero)
    assert cov.shape == (1, 8, 8)
    assert np.linalg.eigvalsh(cov[0]).min() > 0
    np.testing.assert_allclose(cov, document_spd_matrices(zero + 123.))
    with pytest.raises(ValueError, match='shrinkage'):
        document_spd_matrices(zero, shrinkage=1.)
