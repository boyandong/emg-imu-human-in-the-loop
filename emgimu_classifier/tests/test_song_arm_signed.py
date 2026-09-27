"""Check the signed arm candidate's causal and orientation-sensitive contract."""
from __future__ import annotations

import unittest

import numpy as np

from benchmarks.new_bank_v2.song_arm_signed_study import signed_features


class SongArmSignedTests(unittest.TestCase):
    def test_opposite_device_axis_motion_is_distinguishable(self):
        positive = np.zeros((22, 6), dtype=np.float64)
        positive[:, 3] = 0.7
        positive[:5, 0] = -0.4
        positive[-5:, 0] = 0.4
        negative = -positive
        values = signed_features(np.stack((positive, negative)))
        np.testing.assert_allclose(values[0], -values[1])
        self.assertGreater(values[0, 0], 0)
        self.assertGreater(values[0, 3], 0)

    def test_each_window_uses_only_its_own_samples_and_rejects_invalid_input(self):
        windows = np.zeros((2, 22, 6), dtype=np.float64)
        first = signed_features(windows)[0].copy()
        windows[1, :, :] = 1000
        np.testing.assert_array_equal(signed_features(windows)[0], first)
        with self.assertRaisesRegex(ValueError, "finite"):
            signed_features(np.full((1, 22, 6), np.nan))
        with self.assertRaisesRegex(ValueError, "22"):
            signed_features(np.zeros((1, 21, 6)))


if __name__ == "__main__":
    unittest.main()
