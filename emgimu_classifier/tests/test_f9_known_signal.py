"""Independent F9 spectral and robust-reference numerical oracles."""
import pickle

import numpy as np

from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.quality_observability import QualityObservabilityFamily


def direct_power(window: np.ndarray) -> np.ndarray:
    samples = len(window)
    centered = window - np.mean(window)
    tapered = centered * (0.5 - 0.5 * np.cos(2. * np.pi * np.arange(samples) / (samples - 1)))
    indices = np.arange(samples // 2 + 1)
    phase = np.exp(-2j * np.pi * np.outer(indices, np.arange(samples)) / samples)
    return np.abs(phase @ tapered) ** 2


def test_line_and_low_frequency_ratios_match_direct_fourier_sums():
    rate = 250.
    samples = 250
    t = np.arange(samples) / rate
    rng = np.random.default_rng(901)
    source = FeatureBatch(rng.normal(size=(8, samples, 2)), rate)
    family = QualityObservabilityFamily(line_frequency_hz=50., low_frequency_hz=10.,
                                        pre_highpass_available=True).fit(source)
    target = np.stack((np.sin(2*np.pi*50*t) + .3*np.sin(2*np.pi*45*t)
                       + .2*np.sin(2*np.pi*5*t) + .4*np.sin(2*np.pi*30*t),
                       np.sin(2*np.pi*35*t)), axis=1)[None]
    before = pickle.dumps(family)
    observed = family.transform(FeatureBatch(target, rate))[0]
    names = family.feature_names
    power = direct_power(target[0, :, 0])
    frequencies = np.arange(samples // 2 + 1)
    line = np.abs(frequencies - 50) <= 1
    neighbor = ((frequencies >= 44) & (frequencies <= 47)) | (
        (frequencies >= 53) & (frequencies <= 56))
    valid_emg = (frequencies >= 20) & (frequencies <= 118.75)
    expected_line = power[line].mean() / max(power[neighbor].mean(), 1e-10)
    expected_low = power[frequencies <= 10].sum() / max(power[valid_emg].sum(), 1e-10)
    np.testing.assert_allclose(observed[names.index("F9.line_noise_ratio.ch1")],
                               expected_line, rtol=1e-5)
    np.testing.assert_allclose(observed[names.index("F9v2.low_frequency_power_ratio.ch1")],
                               expected_low, rtol=1e-5)
    assert family.availability_ == {"adc_clipping": False, "line_noise": True,
                                    "low_frequency_pre_highpass": True}
    assert before == pickle.dumps(family)


def test_amplitude_and_neighbor_correlation_use_only_source_reference():
    rate = 250.
    samples = 250
    t = np.arange(samples) / rate
    rng = np.random.default_rng(902)
    source_values = rng.normal(size=(9, samples, 2))
    source = FeatureBatch(source_values, rate)
    family = QualityObservabilityFamily().fit(source)
    before = pickle.dumps(family)
    target = np.stack((2*np.sin(2*np.pi*30*t),
                       np.sin(2*np.pi*30*t) + .1*np.sin(2*np.pi*55*t)), axis=1)[None]
    observed = family.transform(FeatureBatch(target, rate))[0]
    names = family.feature_names
    source_rms = np.sqrt(np.mean(source_values[:, :, 0] ** 2, axis=1))
    median_rms = np.median(source_rms)
    mad_rms = np.median(np.abs(source_rms - median_rms))
    current_rms = np.sqrt(np.mean(target[0, :, 0] ** 2))
    expected_amplitude = (current_rms - median_rms) / (1.4826 * mad_rms)
    source_correlation = np.array([np.corrcoef(window.T)[0, 1]
                                   for window in source_values])
    current_correlation = np.corrcoef(target[0].T)[0, 1]
    correlation_center = np.median(source_correlation)
    correlation_scale = max(1.4826 * np.median(
        np.abs(source_correlation - correlation_center)), 1e-4)
    expected_correlation = abs(current_correlation - correlation_center) / correlation_scale
    np.testing.assert_allclose(observed[names.index("F9.amplitude_z.ch1")],
                               expected_amplitude, rtol=1e-6)
    np.testing.assert_allclose(observed[names.index("F9v2.correlation_anomaly.ch1")],
                               expected_correlation, rtol=1e-6)
    assert before == pickle.dumps(family)
