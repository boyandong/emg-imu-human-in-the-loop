import unittest

import numpy as np

from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.reconstructed_ring import ReconstructedCes, ReconstructedRlcs


class ReconstructedRingTests(unittest.TestCase):
    def test_rlcs_keeps_circular_topology_but_ces_discards_channel_order(self):
        rng = np.random.default_rng(42)
        base = rng.normal(size=(7, 256, 8))
        base[:, :, 1] = base[:, :, 0] * 0.7 + rng.normal(size=(7, 256)) * 0.3
        batch = FeatureBatch(base, 200.0)
        rotated = FeatureBatch(np.roll(base, 2, axis=2), 200.0)
        scrambled = FeatureBatch(base[:, :, [0, 2, 1, 3, 4, 5, 6, 7]], 200.0)
        rlcs = ReconstructedRlcs().fit(batch)
        ces = ReconstructedCes().fit(batch)
        self.assertEqual(len(rlcs.feature_names), 8)
        self.assertEqual(len(ces.feature_names), 8)
        np.testing.assert_allclose(rlcs.transform(batch), rlcs.transform(rotated), atol=1e-6)
        self.assertGreater(float(np.max(np.abs(rlcs.transform(batch) - rlcs.transform(scrambled)))), 1e-5)
        np.testing.assert_allclose(ces.transform(batch), ces.transform(scrambled), atol=1e-6)

    def test_constant_signal_is_finite_and_fit_checks_shape(self):
        batch = FeatureBatch(np.ones((2, 256, 8)), 200.0)
        for family in (ReconstructedRlcs(), ReconstructedCes()):
            with self.assertRaisesRegex(ValueError, "fit"):
                family.transform(batch)
            features = family.fit_transform(batch)
            self.assertEqual(features.shape, (2, 8))
            self.assertTrue(np.isfinite(features).all())
            with self.assertRaisesRegex(ValueError, "channel count"):
                family.transform(FeatureBatch(np.ones((2, 256, 7)), 200.0))


if __name__ == "__main__":
    unittest.main()
