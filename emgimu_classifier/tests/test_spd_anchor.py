import pickle
import unittest

import numpy as np

from emgimu.feature_bank import FeatureBatch, SpdTangentFamily, SpdTangentPersonalAnchor


def matrix_log(matrix):
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
    return (vectors * np.log(values)) @ vectors.T


def tangent_oracle(window, reference, shrinkage):
    centered = window - window.mean(axis=0)
    covariance = centered.T @ centered / (len(window) - 1)
    covariance = (1 - shrinkage) * covariance + shrinkage * np.trace(covariance) / 2 * np.eye(2)
    covariance += 1e-10 * np.eye(2)
    covariance /= np.trace(covariance)
    values, vectors = np.linalg.eigh(reference)
    inverse_root = (vectors * values ** -0.5) @ vectors.T
    return matrix_log(inverse_root @ covariance @ inverse_root)


class SpdAnchorTests(unittest.TestCase):
    def test_frozen_source_tangent_anchor_matches_frobenius_matrix_oracle(self):
        rng = np.random.default_rng(20260922)
        source = FeatureBatch(rng.normal(size=(6, 96, 2)), 200)
        calibration = FeatureBatch(rng.normal(size=(4, 96, 2)), 200)
        evaluation = FeatureBatch(rng.normal(size=(2, 96, 2)), 200)
        labels = np.array([0, 0, 1, 1])
        family = SpdTangentFamily().fit(source)
        anchor = SpdTangentPersonalAnchor(family).fit(calibration, labels)
        before = pickle.dumps(anchor)
        observed = anchor.transform(evaluation)
        self.assertEqual(before, pickle.dumps(anchor))

        reference = family.reference_.copy()
        cal = [tangent_oracle(window, reference, family.shrinkage) for window in calibration.emg]
        means = [np.mean(cal[:2], axis=0), np.mean(cal[2:], axis=0)]
        expected = np.array([[np.linalg.norm(tangent_oracle(window, reference, family.shrinkage) - mean, 'fro')
                              for mean in means] for window in evaluation.emg])
        np.testing.assert_allclose(observed[:, :2], expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(anchor.transform(evaluation.take([0]))[0], observed[0], rtol=0, atol=0)
        family.reference_[:] = np.eye(2) * 10  # external source object cannot alter the frozen anchor
        np.testing.assert_array_equal(anchor.transform(evaluation), observed)

    def test_failed_calibration_preserves_previous_prototypes(self):
        rng = np.random.default_rng(41)
        batch = FeatureBatch(rng.normal(size=(5, 48, 2)), 200)
        family = SpdTangentFamily().fit(batch)
        anchor = SpdTangentPersonalAnchor(family).fit(batch.take([0, 1, 2, 3]), [0, 0, 1, 1])
        before = pickle.dumps(anchor)
        with self.assertRaises(ValueError):
            anchor.fit(batch.take([0, 1]), [0, 0])
        self.assertEqual(before, pickle.dumps(anchor))
