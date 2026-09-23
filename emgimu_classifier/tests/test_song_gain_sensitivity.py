import unittest

import numpy as np

from benchmarks.song_gain_sensitivity import median_window_rms, summarize


class SongGainSensitivityTest(unittest.TestCase):
    def test_active_to_neutral_is_conditioned_on_active_truth(self):
        classes = np.asarray(("fist", "index_pinch", "neutral", "open_hand"))
        truth = np.asarray(("fist", "neutral", "open_hand", "index_pinch"))
        probability = np.asarray((
            (.8, .1, .05, .05),
            (.05, .05, .85, .05),
            (.05, .05, .85, .05),
            (.05, .8, .1, .05),
        ))
        result = summarize(truth, probability, classes)
        self.assertEqual(result["trials"], 4)
        self.assertAlmostEqual(result["predicted_neutral_fraction"], .5)
        self.assertAlmostEqual(result["active_to_neutral_error_fraction"], 1 / 3)

    def test_channel_rms_uses_samples_before_window_median(self):
        windows = np.ones((2, 50, 8))
        windows[1] *= 3
        np.testing.assert_allclose(median_window_rms(windows), np.full(8, 2.0))
        with self.assertRaises(ValueError):
            median_window_rms(np.ones((2, 49, 8)))


if __name__ == "__main__":
    unittest.main()
