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
