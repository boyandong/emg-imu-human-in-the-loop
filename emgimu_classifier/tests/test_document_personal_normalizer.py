"""Known-signal check for the specification's Q95 plus epsilon rule."""
import pickle

import numpy as np

from emgimu.feature_bank.calibration import (
    EPS, DocumentPersonalNormalizerV2, PersonalNormalizer,
)
from emgimu.feature_bank.core import FeatureBatch


def test_document_personal_normalizer_uses_raw_q95_plus_epsilon() -> None:
    calibration = np.zeros((4, 4, 2), dtype=np.float64)
    calibration[:, :, 0] = 2.0
    calibration[:, :, 1] = -3.0
    calibration[2:, :, 0] += EPS / 2
    labels = np.array([0, 0, 1, 1])
    batch = FeatureBatch(calibration, 200.0)
    exact = DocumentPersonalNormalizerV2(rest_label=0).fit(batch, labels)
    legacy = PersonalNormalizer(rest_label=0).fit(batch, labels)
    expected_small_scale = float(calibration[2, 0, 0] - 2.0)
    np.testing.assert_array_equal(exact.center_, [2.0, -3.0])
    np.testing.assert_allclose(exact.scale_, [expected_small_scale, 0.0], atol=0, rtol=0)
    np.testing.assert_allclose(legacy.scale_, [EPS, EPS], atol=0, rtol=0)

    target = np.array([[[2.0 + EPS / 2, -2.0]] * 4])
    target_batch = FeatureBatch(target, 200.0)
    frozen = pickle.dumps(exact)
    transformed = exact.transform(target_batch).emg[0, 0]
    np.testing.assert_allclose(transformed,
                               [expected_small_scale / (expected_small_scale + EPS), 1.0 / EPS],
                               rtol=1e-12, atol=1e-12)
    assert transformed[0] < legacy.transform(target_batch).emg[0, 0, 0]
    assert pickle.dumps(exact) == frozen
    np.testing.assert_array_equal(target_batch.emg, target)
