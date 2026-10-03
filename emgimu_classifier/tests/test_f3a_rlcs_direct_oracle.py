"""Direct correlation oracle for F3a ring-lag mean/std."""

import numpy as np
import pytest

from emgimu.feature_bank import FeatureBatch
from emgimu.feature_bank.reconstructed_ring import ReconstructedRlcs


def test_rlcs_lag_profile_matches_direct_pearson_on_positive_envelopes():
    rng = np.random.default_rng(117)
    samples, channels, rate = 64, 8, 40.
    base = np.abs(rng.normal(size=(samples, channels))) + .3
    base[:, 1] = .7 * base[:, 0] + .3 * base[:, 1]
    base[:, 4] = .4 * base[:, 3] + .6 * base[:, 4]
    batch = FeatureBatch(base[None], rate)
    family = ReconstructedRlcs(envelope_ms=25.).fit(batch)
    got = family.transform(batch)[0]
    # At 40 Hz, 25 ms is exactly one sample, so the rectified envelope is
    # the positive input itself. Compute each Pearson coefficient directly.
    expected = []
    for lag in range(1, channels // 2 + 1):
        values = []
        for channel in range(channels):
            a = base[:, channel]
            b = base[:, (channel + lag) % channels]
            da, db = a - a.mean(), b - b.mean()
            values.append(float(np.dot(da, db) / np.sqrt(np.dot(da, da) * np.dot(db, db))))
        expected.extend((np.mean(values), np.std(values)))
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(family.transform(FeatureBatch(np.roll(base, 3, axis=1)[None], rate))[0],
                               got, rtol=1e-6, atol=1e-7)


def test_new_fit_rejects_rate_change_and_constant_channels_are_finite():
    zeros = np.zeros((1, 64, 8))
    family = ReconstructedRlcs().fit(FeatureBatch(zeros, 40.))
    assert np.isfinite(family.transform(FeatureBatch(zeros, 40.))).all()
    with pytest.raises(ValueError, match="sample rate differs"):
        family.transform(FeatureBatch(zeros, 200.))
