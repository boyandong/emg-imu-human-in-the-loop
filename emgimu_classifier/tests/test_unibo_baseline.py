from __future__ import annotations

import unittest

import numpy as np

from emgimu.datasets.unibo_baseline import (
    UniBoWindows,
    extract_unibo_emg_features,
    hierarchical_segment_weights,
)
from emgimu.datasets.unibo_neural_baseline import UniBoFourChannelTCN, torch


class UniBoBaselineTests(unittest.TestCase):
    def test_feature_schema_is_four_channel_and_finite(self):
        rng = np.random.default_rng(7)
        features = extract_unibo_emg_features(rng.uniform(0.0, 10.0, size=(40, 4)))
        self.assertEqual(features.shape, (24,))
        self.assertTrue(np.isfinite(features).all())
        with self.assertRaises(ValueError):
            extract_unibo_emg_features(np.zeros((40, 8)))

    def test_hierarchical_weights_equalize_sessions_trials_and_segments(self):
        windows = UniBoWindows(
            features=np.zeros((9, 24), dtype=np.float32),
            labels=np.asarray([0, 0, 2, 2, 2, 0, 1, 1, 1], dtype=np.int16),
            trial_id=np.asarray(["a", "a", "a", "a", "a", "b", "c", "c", "c"]),
            subject_id=np.asarray(["u1"] * 6 + ["u2"] * 3),
            session_id=np.asarray(["d1"] * 6 + ["d2"] * 3),
            posture=np.ones(9, dtype=np.int16),
            timestamp_ms=np.arange(9, dtype=np.float64),
        )
        weights = hierarchical_segment_weights(windows)
        self.assertAlmostEqual(float(weights[:6].sum()), float(weights[6:].sum()))
        self.assertAlmostEqual(float(weights[:2].sum()), float(weights[2:5].sum()))
        self.assertTrue(np.all(weights > 0))

    @unittest.skipIf(torch is None, "PyTorch is optional")
    def test_four_channel_tcn_shape_and_contract(self):
        model = UniBoFourChannelTCN(8, (1, 2), 0.0)
        output = model(torch.zeros((3, 40, 4), dtype=torch.float32))
        self.assertEqual(tuple(output.shape), (3, 4))
        with self.assertRaises(ValueError):
            model(torch.zeros((3, 40, 8), dtype=torch.float32))


if __name__ == "__main__":
    unittest.main()
