import unittest

import numpy as np

from benchmarks.song_quality_observability import component_summary, quality_summary


class SongQualityObservabilityTest(unittest.TestCase):
    def test_summary_uses_per_hand_denominators(self):
        names = ("F9.mean_quality", "F9.min_quality", "F9.bad_channel_count")
        features = np.asarray(((1, 1, 0), (.2, .1, 1), (.8, .6, 0), (.9, .7, 0)))
        labels = np.asarray(("neutral", "fist", "fist", "index_pinch", "open_hand"))
        features = np.vstack((features, (.9, .7, 0)))
        result = quality_summary(features, names, labels)
        self.assertEqual(result["windows"], 5)
        self.assertEqual(result["by_hand"]["fist"]["windows"], 2)
        self.assertAlmostEqual(result["by_hand"]["fist"]["hypothetical_min_quality_below_0_5_fraction"], .5)
        self.assertAlmostEqual(result["any_bad_channel_fraction"], .2)
        with self.assertRaises(ValueError):
            quality_summary(features, names, labels[:-1])

    def test_component_attribution_distinguishes_activation_from_flatline(self):
        metrics = ("amplitude_z", "zero_fraction", "flatline_fraction", "clip_fraction")
        names = tuple(f"F9.{metric}.ch{c}" for metric in metrics for c in range(1, 9))
        names += tuple(f"F9v2.longest_flatline_ratio.ch{c}" for c in range(1, 9))
        names += ("F9.min_quality",)
        features = np.zeros((2, len(names)))
        features[0, names.index("F9.amplitude_z.ch2")] = 4
        features[0, names.index("F9.min_quality")] = .3
        features[1, names.index("F9.flatline_fraction.ch1")] = .98
        features[1, names.index("F9v2.longest_flatline_ratio.ch1")] = .98
        features[1, names.index("F9.min_quality")] = .02
        result = component_summary(features, names)
        self.assertEqual(result["amplitude_z_over_3_fraction"], .5)
        self.assertEqual(result["candidate_hardware_only_flag_fraction"], .5)
        self.assertEqual(result["legacy_flag_not_explained_by_components"], 0)


if __name__ == "__main__":
    unittest.main()
