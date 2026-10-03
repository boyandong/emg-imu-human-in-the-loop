"""Independent near-epsilon F7 formula checks; legacy anchors remain unchanged."""
from __future__ import annotations

import pickle

import numpy as np

from emgimu.feature_bank.calibration import (
    DocumentPersonalAnchorV2, PersonalAnchor,
)


def test_document_standardized_distance_uses_raw_scale_plus_epsilon() -> None:
    calibration = np.array([[0., 0.], [0., 0.], [2e-10, 0.], [2e-10, 0.]])
    labels = np.array([0, 0, 1, 1])
    query = np.array([[1e-10, 0.]])
    exact = DocumentPersonalAnchorV2(metric="standardized_euclidean").fit(
        calibration, labels)
    legacy = PersonalAnchor(metric="standardized_euclidean").fit(calibration, labels)
    raw_mad = 1.4826e-10
    np.testing.assert_allclose(exact.scale_, [raw_mad, 0.], rtol=1e-14, atol=0)
    distances = np.array([1e-10, 1e-10]) / (raw_mad + 1e-10)
    before = pickle.dumps(exact)
    observed = exact.transform(query)[0]
    np.testing.assert_allclose(observed[:2], distances, rtol=1e-6)
    np.testing.assert_allclose(observed[2:4],
                               np.exp(-distances / exact.similarity_scale_), rtol=1e-6)
    np.testing.assert_allclose(observed[-2:], 0., atol=1e-7)
    assert pickle.dumps(exact) == before
    assert np.max(np.abs(observed[:2] - legacy.transform(query)[0, :2])) > .2


def test_document_cosine_uses_product_plus_epsilon() -> None:
    calibration = np.array([[1e-5, 0.], [1e-5, 0.],
                            [0., 1e-5], [0., 1e-5]])
    labels = np.array([0, 0, 1, 1])
    query = np.array([[1e-5, 0.]])
    exact = DocumentPersonalAnchorV2(metric="cosine").fit(calibration, labels)
    legacy = PersonalAnchor(metric="cosine").fit(calibration, labels)
    observed = exact.transform(query)[0]
    np.testing.assert_allclose(observed[:2], [.5, 1.], rtol=1e-6)
    np.testing.assert_allclose(observed[-2:], [.5, .5 / (.75 + 1e-10)], rtol=1e-6)
    assert legacy.transform(query)[0, 0] < 1e-6
