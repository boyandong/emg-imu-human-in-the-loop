import unittest

import numpy as np

from benchmarks.song_real8_study import _trial_metrics, parse_label, window_starts


class SongReal8StudyTests(unittest.TestCase):
    def test_window_selection_stays_inside_stable_interval_without_overlap(self):
        for length in (20, 49, 50, 73, 99, 100, 130, 180, 305):
            starts = window_starts(1000, 1000 + length)
            self.assertLessEqual(len(starts), 3)
            self.assertTrue(all(1000 <= start and start + 50 <= 1000 + length for start in starts))
            self.assertTrue(all(b - a >= 50 for a, b in zip(starts, starts[1:])))
        self.assertEqual(len(window_starts(0, 49)), 0)

    def test_labels_reject_unexpected_states(self):
        self.assertEqual(parse_label("still_open_hand"), ("still", "open_hand"))
        self.assertEqual(parse_label("backward_index_pinch"), ("backward", "index_pinch"))
        with self.assertRaises(ValueError):
            parse_label("unknown_open_hand")

    def test_evaluation_counts_trials_not_windows(self):
        labels = np.array(["neutral", "neutral", "fist", "fist", "fist"])
        trials = np.array(["S04:1", "S04:1", "S04:2", "S04:2", "S04:2"])
        probabilities = np.array([[0.1, 0.9], [0.1, 0.9], [0.9, 0.1], [0.9, 0.1], [0.9, 0.1]])
        outcome = _trial_metrics(labels, probabilities, trials, np.array(["fist", "neutral"]))
        self.assertEqual(outcome["trials"], 2)
        self.assertEqual(outcome["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
