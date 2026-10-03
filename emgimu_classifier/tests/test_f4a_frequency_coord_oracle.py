"""Independent DFT F4a oracle and train/test frequency-grid invariant."""

import numpy as np
import pytest

from emgimu.feature_bank import FeatureBatch, SpectralStateFamily


def test_band_orientation_matches_direct_complex_dft_at_250_hz():
    sample_count, channels, rate = 50, 8, 250.
    t = np.arange(sample_count) / rate
    x = np.zeros((1, sample_count, channels))
    for channel, frequency in enumerate((25., 50., 75., 100.)):
        x[0, :, channel] = (channel + 1) * np.sin(2 * np.pi * frequency * t)
    x[0, :, 4] = .8 * np.sin(2 * np.pi * 25 * t) + .4 * np.sin(2 * np.pi * 75 * t)
    batch = FeatureBatch(x, rate)
    family = SpectralStateFamily().fit(batch)
    got = family.transform(batch)[0, :32].reshape(4, channels)

    # Construct the positive-frequency DFT directly, without calling FFT,
    # then partition the same frozen sub-Nyquist bands.
    centered = x[0] - x[0].mean(axis=0, keepdims=True)
    tapered = centered * np.hanning(sample_count)[:, None]
    frequency_grid = np.arange(sample_count // 2 + 1) * rate / sample_count
    kernel = np.exp(-2j * np.pi * np.outer(np.arange(sample_count // 2 + 1),
                                          np.arange(sample_count)) / sample_count)
    power = np.abs(kernel @ tapered) ** 2 / sample_count
    expected = []
    for index, (lower, upper) in enumerate(family.bands_):
        bins = (frequency_grid >= lower) & (frequency_grid <= upper if index == 3 else frequency_grid < upper)
        energy = power[bins].sum(axis=0)
        expected.append(energy / max(np.linalg.norm(energy), 1e-10))
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-7)
    assert family.bands_[-1][1] < rate / 2
    np.testing.assert_array_equal(got.argmax(axis=1), np.arange(4))


def test_fitted_spectral_bands_reject_another_sample_rate():
    x = np.zeros((1, 50, 8))
    family = SpectralStateFamily().fit(FeatureBatch(x, 250.))
    with pytest.raises(ValueError, match="sample rate differs"):
        family.transform(FeatureBatch(x, 2000.))
