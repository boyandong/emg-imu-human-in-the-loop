import numpy as np
import pytest

from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.new_bank_v2 import (
    RestNoiseDetailV2, RingRelativeCovarianceV2, TraceCovarianceV2,
    new_bank_v2_registry,
)


def test_v2_registers_only_independent_v1_and_new_blocks():
    registry = new_bank_v2_registry()
    assert len(registry.family_ids) == 7
    assert isinstance(registry.create("new_v2_rest_noise_detail"), RestNoiseDetailV2)
    assert isinstance(registry.create("new_v2_trace_covariance"), TraceCovarianceV2)
    assert isinstance(registry.create("new_v2_ring_relative_covariance"), RingRelativeCovarianceV2)


def test_rest_noise_detail_requires_rest_and_uses_only_rest_thresholds():
    rng = np.random.default_rng(112)
    rest = rng.normal(scale=0.02, size=(4, 200, 8))
    active = rng.normal(scale=20, size=(4, 200, 8))
    source = FeatureBatch(np.concatenate((rest, active)), 1000.0)
    labels = np.array([0] * 4 + [1] * 4)
    family = RestNoiseDetailV2(rest_label=0)
    with pytest.raises(ValueError, match="Rest"):
        family.fit(source, np.ones(8))
    family.fit(source, labels)
    thresholds = family.thresholds_.copy()
    modified = FeatureBatch(np.concatenate((rest, active * 100)), 1000.0)
    second = RestNoiseDetailV2(rest_label=0).fit(modified, labels)
    np.testing.assert_allclose(thresholds, second.thresholds_, rtol=0, atol=0)
    assert family.rest_windows_ == 4
    assert family.transform(source).shape == (8, 48)
    with pytest.raises(ValueError, match="sample rate"):
        family.transform(FeatureBatch(source.emg, 999.0))


def test_trace_covariance_matches_formula_and_is_gain_invariant():
    rng = np.random.default_rng(113)
    x = rng.normal(size=(3, 250, 8)) * np.arange(1, 9)
    batch = FeatureBatch(x, 1000.0)
    family = TraceCovarianceV2(shrinkage=0.05).fit(batch)
    actual = family.transform(batch)
    centered = x[0] - x[0].mean(axis=0)
    covariance = centered.T @ centered / 249
    shrunk = 0.95 * covariance + 0.05 * np.trace(covariance) / 8 * np.eye(8)
    normalized = shrunk / np.trace(shrunk)
    i, j = np.triu_indices(8)
    expected = normalized[i, j].copy()
    expected[i != j] *= np.sqrt(2)
    np.testing.assert_allclose(actual[0], expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(actual, family.transform(FeatureBatch(x * 7, 1000.0)), atol=1e-7)
    assert actual.shape == (3, 36)


def test_ring_covariance_preserves_rotation_but_detects_nonring_swap():
    rng = np.random.default_rng(114)
    x = rng.normal(size=(5, 200, 8))
    x[:, :, 1] = 0.8 * x[:, :, 0] + 0.2 * rng.normal(size=(5, 200))
    source = FeatureBatch(x, 1000.0)
    family = RingRelativeCovarianceV2().fit(source)
    actual = family.transform(source)
    rotated = family.transform(FeatureBatch(np.roll(x, 2, axis=2), 1000.0))
    swapped = family.transform(FeatureBatch(x[:, :, [0, 2, 1, 3, 4, 5, 6, 7]], 1000.0))
    assert actual.shape == (5, 24)
    np.testing.assert_allclose(actual, rotated, atol=1e-6)
    assert np.max(np.abs(actual - swapped)) > 1e-4
    np.testing.assert_allclose(actual, family.transform(FeatureBatch(x * 5, 1000.0)), atol=1e-6)


def test_ring_covariance_short_window_omits_unstable_temporal_block():
    short = FeatureBatch(np.zeros((2, 50, 8)), 250.0)
    family = RingRelativeCovarianceV2().fit(short)
    assert not family.with_temporal_
    assert len(family.feature_names) == 20
    np.testing.assert_array_equal(family.transform(short), np.zeros((2, 20)))
    with pytest.raises(ValueError, match="window samples"):
        family.transform(FeatureBatch(np.zeros((2, 51, 8)), 250.0))
