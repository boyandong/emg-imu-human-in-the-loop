"""Independent F1 X1-H replacement formula and sensor-contract checks."""

import numpy as np
import pytest

from emgimu.feature_bank import FeatureBatch, ScalePatternFamily


def test_eight_channel_rms_over_global_rms_and_no_scale_output():
    amplitudes = np.arange(1., 9.)
    x = np.tile(amplitudes, (1, 40, 1))
    batch = FeatureBatch(x, 250.)
    family = ScalePatternFamily().fit(batch)
    expected = amplitudes / np.sqrt(np.mean(amplitudes ** 2))
    assert len(family.feature_names) == 8
    np.testing.assert_allclose(family.transform(batch)[0], expected, rtol=1e-7)
    np.testing.assert_allclose(family.transform(FeatureBatch(x * 31., 250.))[0],
                               expected, rtol=1e-7)


def test_new_fit_rejects_channel_and_rate_mismatch():
    x = np.ones((1, 40, 8))
    family = ScalePatternFamily().fit(FeatureBatch(x, 250.))
    with pytest.raises(ValueError, match='channel count'):
        family.transform(FeatureBatch(x[:, :, :4], 250.))
    with pytest.raises(ValueError, match='sample rate'):
        family.transform(FeatureBatch(x, 2000.))


@pytest.mark.parametrize('rate', [float('nan'), float('inf'), -1., 0.])
def test_window_rejects_invalid_frequency_grid(rate):
    with pytest.raises(ValueError, match='finite and positive'):
        FeatureBatch(np.ones((1, 40, 8)), rate)
