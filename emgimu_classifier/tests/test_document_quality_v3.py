import pickle

import numpy as np
import pytest

from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.document_quality_v3 import DocumentQualityObservationsV3


def test_strict_zero_flat_and_additive_mad_denominator():
    source = np.zeros((2, 8, 2))
    source[:, :, 1] = np.array([1, 2, 1, 2, 1, 2, 1, 2])
    family = DocumentQualityObservationsV3(zero_threshold=.125, flat_threshold=.125).fit(FeatureBatch(source, 200))
    query = source[:1].copy()
    query[0, :, 0] = [0, .125, .25, .25, .25, .375, .5, .625]
    before = pickle.dumps(family)
    values = family.transform(FeatureBatch(query, 200))[0]
    names = family.feature_names
    assert values[names.index("F9v3.zero_fraction.ch1")] == pytest.approx(1 / 8)
    assert values[names.index("F9v3.longest_flatline_ratio.ch1")] == pytest.approx(2 / 8)
    rms = np.sqrt(np.mean(query[0, :, 0] ** 2))
    assert values[names.index("F9v3.amplitude_z.ch1")] == pytest.approx(rms / 1e-10, rel=1e-6)
    assert values[names.index("F9v3.available.adc")] == 0
    assert values[names.index("F9v3.available.line")] == 0
    assert values[names.index("F9v3.available.low_frequency")] == 0
    assert values[names.index("F9v3.available.ring")] == 0
    assert before == pickle.dumps(family)


def test_line_and_low_band_match_direct_fourier_with_explicit_metadata():
    rate, samples = 1000, 1000
    t = np.arange(samples) / rate
    wave = np.sin(2 * np.pi * 50 * t) + .25 * np.sin(2 * np.pi * 45 * t) + .1 * np.sin(2 * np.pi * 5 * t)
    source = FeatureBatch(np.stack([wave, 2 * wave])[..., None], rate)
    family = DocumentQualityObservationsV3(line_frequency_hz=50, pre_highpass_available=True,
                                           adc_range=(-5, 5)).fit(source)
    before = pickle.dumps(family)
    values = family.transform(source)[0]
    frequency = np.fft.rfftfreq(samples, 1 / rate)
    power = np.abs(np.fft.rfft((wave - wave.mean()) * np.hanning(samples))) ** 2
    line = np.abs(frequency - 50) <= 1
    neighbor = ((frequency >= 44) & (frequency <= 47)) | ((frequency >= 53) & (frequency <= 56))
    expected_line = power[line].mean() / (power[neighbor].mean() + 1e-10)
    valid = (frequency >= 20) & (frequency <= 450)
    expected_low = power[frequency <= 10].sum() / (power[valid].sum() + 1e-10)
    names = family.feature_names
    assert values[names.index("F9v3.line_noise_ratio.ch1")] == pytest.approx(expected_line, rel=1e-6)
    assert values[names.index("F9v3.low_frequency_ratio.ch1")] == pytest.approx(expected_low, rel=1e-6)
    assert [values[names.index(f"F9v3.available.{key}")] for key in ("adc", "line", "low_frequency")] == [1, 1, 1]
    assert before == pickle.dumps(family)


def test_explicit_ring_order_and_missing_line_resolution():
    t = np.arange(16)
    x = np.stack([np.sin(t), np.cos(t), np.sin(t + .3)], axis=1)
    batch = FeatureBatch(np.stack([x, x * 1.1]), 200)
    family = DocumentQualityObservationsV3(ring_order=(2, 0, 1), line_frequency_hz=60).fit(batch)
    values = family.transform(batch)
    assert values.shape == (2, 8 * 3 + 5)
    assert np.isfinite(values).all()
    assert values[0, family.feature_names.index("F9v3.available.ring")] == 1
    assert values[0, family.feature_names.index("F9v3.available.line")] == 0
    with pytest.raises(ValueError, match="permutation"):
        DocumentQualityObservationsV3(ring_order=(0, 0, 1)).fit(batch)


def test_known_adc_covariance_and_ring_neighbor_observations():
    source = np.array([[0., 1., -1.], [.2, -.5, .3], [-.2, .4, .7],
                       [.1, -.1, -.6], [.4, .6, .1], [-.3, -.7, -.2],
                       [.3, .2, -.4], [-.1, .8, .5]])
    query = source.copy()
    query[0, 0] = 1.0
    family = DocumentQualityObservationsV3(adc_range=(-1, 1), ring_order=(2, 0, 1)).fit(
        FeatureBatch(np.stack([source, source]), 200)
    )
    observed = family.transform(FeatureBatch(query[None], 200))[0]
    names = family.feature_names
    assert observed[names.index("F9v3.clip_fraction.ch1")] == pytest.approx(1 / 8)
    direct_cov = np.linalg.norm(np.cov(query, rowvar=False) - np.cov(source, rowvar=False), ord="fro")
    assert observed[names.index("F9v3.covariance_distance")] == pytest.approx(direct_cov, rel=1e-6)
    source_corr = np.corrcoef(source, rowvar=False)
    query_corr = np.corrcoef(query, rowvar=False)
    expected_ch1 = ((query_corr[0, 2] + query_corr[0, 1])
                    - (source_corr[0, 2] + source_corr[0, 1])) / 2
    assert observed[names.index("F9v3.neighbor_correlation_shift.ch1")] == pytest.approx(expected_ch1, rel=1e-6)
