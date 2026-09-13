import numpy as np
import pytest

from emgimu.datasets.hla_features import extract_hla_feature_tokens, hla_feature_names


def test_r0_and_r1_core_support_variable_channel_counts() -> None:
    rng = np.random.default_rng(42)
    for channels in (1, 4, 8, 16):
        window = rng.normal(size=(51, channels))
        assert extract_hla_feature_tokens(window, ("G0",)).shape == (channels, 6)
        assert extract_hla_feature_tokens(window, ("G0", "G5")).shape == (channels, 11)


def test_features_are_channel_permutation_equivariant() -> None:
    rng = np.random.default_rng(43)
    window = rng.normal(size=(63, 8))
    order = np.asarray([3, 7, 0, 5, 2, 6, 1, 4])
    original = extract_hla_feature_tokens(window)
    permuted = extract_hla_feature_tokens(window[:, order])
    np.testing.assert_allclose(permuted, original[order], rtol=1e-6, atol=1e-6)


def test_features_are_finite_for_zero_and_constant_windows() -> None:
    for window in (np.zeros((50, 8)), np.ones((50, 8))):
        assert np.isfinite(extract_hla_feature_tokens(window)).all()
    assert len(hla_feature_names()) == 11


def test_features_reject_nonfinite_input() -> None:
    window = np.ones((50, 4))
    window[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        extract_hla_feature_tokens(window)
