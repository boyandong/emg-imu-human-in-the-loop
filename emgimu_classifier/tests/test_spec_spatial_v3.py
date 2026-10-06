"""Independent equation checks for the goal document's centered F2a/F2c/F3c."""
from __future__ import annotations

import pickle

import numpy as np
import pytest

from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.new_bank_v2 import DocumentTraceCovarianceV2
from emgimu.feature_bank.spec_spatial_v3 import (
    SpecRingRelativeCovarianceV3, SpecSpdTangentV3,
    SPD_RIDGE, SpecTraceCovarianceV3, centered_f2a_matrix, spec_spatial_v3_registry,
)


def direct_f2a(sample: np.ndarray, shrinkage: float) -> np.ndarray:
    centered = sample - sample.mean(axis=0)
    covariance = centered.T @ centered / (len(sample) - 1)
    regularized = (1 - shrinkage) * covariance + shrinkage * np.trace(covariance) / 8 * np.eye(8)
    return regularized / (np.trace(regularized) + 1e-12)


def upper(matrix: np.ndarray) -> np.ndarray:
    i, j = np.triu_indices(8)
    out = matrix[i, j].copy()
    out[i != j] *= np.sqrt(2)
    return out


def test_f2a_centered_formula_and_version_boundary() -> None:
    rng = np.random.default_rng(71)
    x = rng.normal(size=(3, 40, 8)) + np.arange(8)[None, None, :] * 3
    batch = FeatureBatch(x, 200)
    family = SpecTraceCovarianceV3().fit(batch)
    expected = np.stack([direct_f2a(sample, .05) for sample in x])
    np.testing.assert_allclose(centered_f2a_matrix(x, .05), expected, atol=1e-14)
    np.testing.assert_allclose(family.transform(batch), np.stack([upper(m) for m in expected]),
                               atol=3e-8)
    np.testing.assert_allclose(family.transform(FeatureBatch(x + 100, 200)),
                               family.transform(batch), atol=3e-7)
    old_uncentered = DocumentTraceCovarianceV2().fit(batch).transform(batch)
    assert np.max(np.abs(old_uncentered - family.transform(batch))) > .01
    assert len(family.feature_names) == 36
    np.testing.assert_array_equal(centered_f2a_matrix(np.zeros((1, 40, 8)), .05), 0)


def test_f2c_centered_source_reference_and_target_immutability() -> None:
    rng = np.random.default_rng(72)
    source = rng.normal(size=(4, 40, 8)) + np.arange(8)[None, None, :]
    target = rng.normal(size=(2, 40, 8)) + 20
    family = SpecSpdTangentV3().fit(FeatureBatch(source, 200))
    source_matrices = centered_f2a_matrix(source, .05) + SPD_RIDGE * np.eye(8)
    # Direct eigendecompositions recompute the log-Euclidean source reference.
    logs = []
    for matrix in source_matrices:
        values, vectors = np.linalg.eigh(matrix)
        logs.append((vectors * np.log(values)) @ vectors.T)
    values, vectors = np.linalg.eigh(np.mean(logs, axis=0))
    expected_ref = (vectors * np.exp(values)) @ vectors.T
    np.testing.assert_allclose(family.reference_, expected_ref, atol=1e-12)
    before = pickle.dumps(family)
    transformed = family.transform(FeatureBatch(target, 200))
    # Independent SciPy matrix functions check the entire nonzero query
    # tangent, rather than only its shape and the fitted reference matrix.
    from scipy.linalg import fractional_matrix_power, logm
    inverse_root=fractional_matrix_power(expected_ref,-.5)
    expected=[]
    for sample in target:
        matrix=direct_f2a(sample,.05)+SPD_RIDGE*np.eye(8)
        tangent=logm(inverse_root@matrix@inverse_root)
        assert np.max(np.abs(np.imag(tangent)))<1e-12
        expected.append(upper(np.real(tangent)))
    np.testing.assert_allclose(transformed,np.stack(expected),atol=3e-7,rtol=1e-6)
    assert transformed.shape == (2, 36) and np.isfinite(transformed).all()
    assert pickle.dumps(family) == before
    with pytest.raises(ValueError, match="sample rate"):
        family.transform(FeatureBatch(target, 250))


def test_f2c_zero_variance_source_maps_to_its_own_tangent_origin() -> None:
    for level in (0.0, 17.0):
        source = FeatureBatch(np.full((3, 40, 8), level), 200)
        family = SpecSpdTangentV3().fit(source)
        np.testing.assert_allclose(np.diag(family.reference_), SPD_RIDGE, rtol=1e-12)
        np.testing.assert_allclose(family.transform(source), 0, atol=1e-10)


def test_f3c_centered_ring_lags_and_long_window_delta() -> None:
    rng = np.random.default_rng(73)
    x = rng.normal(size=(2, 120, 8)) + np.arange(8)[None, None, :] * 2
    batch = FeatureBatch(x, 200)
    with pytest.raises(ValueError, match="circular"):
        SpecRingRelativeCovarianceV3().fit(batch)
    family = SpecRingRelativeCovarianceV3(ring_topology=True).fit(batch)
    actual = family.transform(batch)
    expected = []
    for sample in x:
        matrix = direct_f2a(sample, .05)
        early = direct_f2a(sample[:60], .05)
        late = direct_f2a(sample[60:], .05)
        row = []
        for lag in range(1, 5):
            values = np.array([matrix[i, (i + lag) % 8] for i in range(8)])
            row.extend([values.mean(), np.median(values), values.std(),
                        np.quantile(values, .25), np.quantile(values, .75)])
            row.append(abs(np.mean([late[i, (i + lag) % 8] for i in range(8)])
                           - np.mean([early[i, (i + lag) % 8] for i in range(8)])))
        expected.append(row)
    np.testing.assert_allclose(actual, expected, atol=3e-8)
    assert actual.shape == (2, 24) and len(family.feature_names) == 24
    rotated = np.roll(x, 3, axis=2)
    np.testing.assert_allclose(family.transform(FeatureBatch(rotated, 200)), actual, atol=3e-8)
    assert spec_spatial_v3_registry().create(family.family_id).family_id == family.family_id
