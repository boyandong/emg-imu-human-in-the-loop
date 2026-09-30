"""Independent two-dimensional personal-anchor distance and margin oracle."""
import pickle

import numpy as np

from emgimu.feature_bank.calibration import PersonalAnchor


def test_euclidean_standardized_and_cosine_anchor_coordinates():
    calibration = np.array([[0., 0.], [0., 2.], [4., 0.], [4., 2.]])
    labels = np.array(["neutral", "neutral", "fist", "fist"])
    query = np.array([[1., 1.]])
    prototypes = np.array([[4., 1.], [0., 1.]])  # sorted labels: fist, neutral
    median = np.array([2., 1.])
    scale = 1.4826 * np.median(np.abs(calibration - median), axis=0)
    expected = {
        "euclidean": np.array([3., 1.]),
        "standardized_euclidean": np.array([3. / scale[0], 1. / scale[0]]),
        "cosine": 1. - np.array([5. / (np.sqrt(2.) * np.sqrt(17.)),
                                 1. / np.sqrt(2.)]),
    }
    for metric, distances in expected.items():
        anchor = PersonalAnchor(metric=metric).fit(calibration, labels)
        before = pickle.dumps(anchor)
        observed = anchor.transform(query)[0]
        np.testing.assert_allclose(anchor.prototypes_, prototypes, rtol=0, atol=0)
        np.testing.assert_allclose(observed[:2], distances, rtol=1e-6, atol=1e-7)
        delta = calibration[:, None, :] - prototypes[None, :, :]
        if metric == "euclidean":
            source_distances = np.sqrt(np.sum(delta ** 2, axis=2))
        elif metric == "standardized_euclidean":
            source_distances = np.sqrt(np.sum((delta / scale) ** 2, axis=2))
        else:
            cosine = calibration @ prototypes.T / np.maximum(
                np.linalg.norm(calibration, axis=1)[:, None] *
                np.linalg.norm(prototypes, axis=1)[None, :], 1e-10)
            source_distances = 1. - cosine
        source_tau = max(float(np.median(source_distances)), 1e-10)
        np.testing.assert_allclose(observed[2:4], np.exp(-distances / source_tau),
                                   rtol=1e-6, atol=1e-7)
        margin = abs(float(distances[0] - distances[1]))
        np.testing.assert_allclose(observed[-2:],
                                   [margin, margin / float(np.mean(distances))],
                                   rtol=1e-6, atol=1e-7)
        assert before == pickle.dumps(anchor)
        np.testing.assert_array_equal(anchor.transform(np.vstack((query, [[1e9, -1e9]])))[0],
                                      observed)
