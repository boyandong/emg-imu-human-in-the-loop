"""Known correlation geometry for the new F3b formula implementation."""
import pickle

import numpy as np
import pytest

from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.document_ces_v3 import DocumentCesFamilyV3


def test_independent_three_plus_one_eigenvalue_oracle_and_channel_permutation():
    alternating = np.array([1., 2.] * 20)
    block = np.array([1.] * 10 + [2.] * 10 + [1.] * 10 + [2.] * 10)
    x = np.stack((alternating, alternating, 3. - alternating, block), axis=1)
    batch = FeatureBatch(np.stack((x, 2 * x)), 200.)
    family = DocumentCesFamilyV3(envelope_ms=5.).fit(batch)
    before = pickle.dumps(family)
    actual = family.transform(batch)
    # Three perfectly collinear channels and one independent channel have
    # correlation eigenvalues 3, 1, 0, 0, irrespective of amplitude or order.
    np.testing.assert_allclose(actual, [[.75, .25, 0., 0.]] * 2, atol=1e-7)
    permuted = FeatureBatch(batch.emg[:, :, [2, 0, 3, 1]], 200.)
    np.testing.assert_allclose(family.transform(permuted), actual, atol=1e-7)
    assert family.feature_names == tuple(f"F3b.ces.p{i}" for i in range(1, 5))
    assert pickle.dumps(family) == before


def test_zero_variance_channels_remain_finite_and_contract_is_fixed():
    x = np.ones((2, 40, 4))
    family = DocumentCesFamilyV3().fit(FeatureBatch(x, 200.))
    np.testing.assert_array_equal(family.transform(FeatureBatch(x, 200.)), 0.)
    with pytest.raises(ValueError, match="contract"):
        family.transform(FeatureBatch(x, 250.))


@pytest.mark.parametrize("rate", [200., 250., 440.])
def test_default_physical_envelope_matches_direct_local_mean_and_pearson(rate):
    rng = np.random.default_rng(736)
    x = rng.normal(size=(47, 8))
    width = round(rate * .025)
    left = (width - 1) // 2
    # Explicit clipped sample indices provide an oracle independent of the
    # production cumulative-sum implementation, including both window edges.
    envelope = np.array([np.abs(x[np.clip(np.arange(t-left, t-left+width), 0, 46)]).mean(axis=0)
                         for t in range(47)])
    eig = np.maximum(np.linalg.eigvalsh(np.corrcoef(envelope.T))[::-1], 0.)
    expected = eig / (eig.sum() + 1e-10)
    batch = FeatureBatch(x[None], rate)
    family = DocumentCesFamilyV3().fit(batch)
    family.envelope_ms = 999.  # Source-fitted physical width is immutable.
    np.testing.assert_allclose(family.transform(batch)[0], expected, atol=2e-7)
