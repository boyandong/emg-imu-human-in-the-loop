import unittest
import numpy as np

from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.ring_covariance import RawRingCovarianceFamily


class RawRingCovarianceTests(unittest.TestCase):
    def test_document_raw_covariance_oracle(self):
        x = np.random.default_rng(12).normal(size=(3, 40, 8))
        family = RawRingCovarianceFamily(ring_topology=True, shrinkage=.05)
        result = family.fit_transform(FeatureBatch(x, 200))
        expected = []
        def cov(values):
            c = np.cov(values, rowvar=False)
            c = .95 * c + .05 * np.trace(c) / 8 * np.eye(8) + 1e-10 * np.eye(8)
            return c / np.trace(c)
        for sample in x:
            full, early, late = cov(sample), cov(sample[:20]), cov(sample[20:])
            row = []
            for lag in range(1, 5):
                indices = [(i, (i+lag) % 8) for i in range(8)]
                v = np.array([full[a,b] for a,b in indices])
                row.extend([v.mean(), np.median(v), v.std(), np.quantile(v,.25), np.quantile(v,.75),
                            abs(np.mean([late[a,b] for a,b in indices])-np.mean([early[a,b] for a,b in indices]))])
            expected.append(row)
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-8)
        self.assertEqual(len(family.feature_names), 24)

    def test_topology_and_rotation(self):
        x = np.random.default_rng(15).normal(size=(5, 60, 8))
        with self.assertRaises(ValueError):
            RawRingCovarianceFamily().fit(FeatureBatch(x, 200))
        family = RawRingCovarianceFamily(ring_topology=True).fit(FeatureBatch(x,200))
        original = family.transform(FeatureBatch(x,200))
        np.testing.assert_allclose(original, family.transform(FeatureBatch(np.roll(x,3,axis=2),200)), atol=1e-8)
        permuted = family.transform(FeatureBatch(x[:,:, [0,2,4,6,1,3,5,7]],200))
        self.assertGreater(np.max(np.abs(original-permuted)), .001)


if __name__ == '__main__':
    unittest.main()
