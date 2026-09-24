import unittest

import numpy as np

from benchmarks.song_signal_calibration import correction, native_calibration_windows


class SongSignalCalibrationTest(unittest.TestCase):
    def test_common_gain_is_cancelled_by_calibration_profile(self):
        source = np.arange(1, 9, dtype=float)[None, None, :] * np.ones((12, 50, 1))
        np.testing.assert_allclose(correction(source, source * .5), np.full(8, 2))
        np.testing.assert_allclose(correction(source, source * 4), np.full(8, .25))
        with self.assertRaises(ValueError):
            correction(source, np.zeros_like(source))

    def test_budget_uses_balanced_preformal_blocks(self):
        hands = np.repeat(("neutral", "index_pinch", "fist", "open_hand"), 6)
        shots = np.tile(np.repeat((1, 2), 3), 4)
        data = {"calibration_shot": shots, "calibration_hand": hands,
                "calibration_batch": type("Batch", (), {"emg": np.ones((24, 50, 8))})()}
        self.assertEqual(native_calibration_windows(data, 1).shape, (12, 50, 8))
        self.assertEqual(native_calibration_windows(data, 2).shape, (24, 50, 8))
        with self.assertRaises(ValueError):
            native_calibration_windows(data, 5)


if __name__ == "__main__":
    unittest.main()
