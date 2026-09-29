"""Direct Fourier-sum oracle for the F4b and F4c spectral blocks."""

import numpy as np

from emgimu.feature_bank import FeatureBatch, SpectralStateFamily


def test_spectral_summary_and_cepstral_blocks_against_direct_fourier_sum():
    samples, channels, rate = 40, 3, 200.
    time = np.arange(samples) / rate
    raw = np.stack((
        np.sin(2*np.pi*15*time),
        0.4*np.sin(2*np.pi*35*time) + 0.2*np.cos(2*np.pi*10*time),
        np.zeros(samples),
    ), axis=1)
    batch = FeatureBatch(raw[None], rate)
    family = SpectralStateFamily(band_count=4, cepstral_coefficients=4).fit(batch)
    actual = family.transform(batch)[0]

    tapered = (raw - raw.mean(axis=0)) * np.hanning(samples)[:, None]
    frequencies = np.arange(samples//2 + 1) * rate/samples
    # Sum sin/cos terms directly, rather than calling the implementation FFT.
    power = np.empty((len(frequencies), channels))
    for index in range(len(frequencies)):
        angle = 2*np.pi*index*np.arange(samples)/samples
        real = np.sum(tapered*np.cos(angle)[:, None], axis=0)
        imag = -np.sum(tapered*np.sin(angle)[:, None], axis=0)
        power[index] = (real**2 + imag**2)/samples

    total = power.sum(axis=0)
    centroid = (frequencies[:, None]*power).sum(axis=0)/np.maximum(total, 1e-10)
    median = frequencies[np.argmax(np.cumsum(power, axis=0) >= total[None]*0.5, axis=0)]
    normalized = power/np.maximum(total[None], 1e-10)
    entropy = -(normalized*np.log(np.maximum(normalized, 1e-10))).sum(axis=0)/np.log(len(power))
    start = 4*channels
    np.testing.assert_allclose(actual[start:start+channels], total, atol=1e-5, rtol=1e-6)
    np.testing.assert_allclose(actual[start+channels:start+2*channels], centroid, atol=1e-5, rtol=1e-6)
    np.testing.assert_allclose(actual[start+2*channels:start+3*channels], median, atol=1e-6, rtol=0)
    np.testing.assert_allclose(actual[start+3*channels:start+4*channels], entropy, atol=1e-6, rtol=1e-6)
    assert total[-1] == 0 and centroid[-1] == 0 and median[-1] == 0 and entropy[-1] == 0

    log_power = np.log(np.maximum(power, 1e-10))
    expected_cepstral = []
    bins = len(frequencies)
    for order in range(1, 5):
        basis = np.cos(np.pi*(np.arange(bins)+0.5)*order/bins)
        coefficients = np.sum(log_power*basis[:, None], axis=0)
        expected_cepstral.extend((coefficients.mean(), coefficients.std()))
    np.testing.assert_allclose(actual[start+4*channels:], expected_cepstral, atol=1e-5, rtol=1e-6)
    assert actual.size == 4*channels + 4*channels + 8
