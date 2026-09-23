import unittest

import numpy as np

from benchmarks.song_calibration_study import paired_trial_audit, target_calibration
from emgimu.feature_bank.core import FeatureBatch


class _FirstSampleFamily:
    def transform(self, batch):
        return batch.emg[:, 0, :]


class SongCalibrationTests(unittest.TestCase):
    def test_budget_uses_only_complete_actual_calibration_blocks(self):
        hands = np.repeat(["neutral", "index_pinch", "fist", "open_hand"], 6)
        shots = np.tile(np.repeat([1, 2], 3), 4)
        data = {"calibration_batch": FeatureBatch(np.ones((24, 50, 8)), 250),
                "calibration_hand": hands, "calibration_shot": shots}
        self.assertEqual(len(target_calibration(data, _FirstSampleFamily(), 1)[1]), 12)
        self.assertEqual(len(target_calibration(data, _FirstSampleFamily(), 2)[1]), 24)
        data["calibration_shot"][0] = 2
        with self.assertRaisesRegex(ValueError, "incomplete or unbalanced"):
            target_calibration(data, _FirstSampleFamily(), 1)

    def test_pairing_reports_corrections_on_whole_trials(self):
        labels = np.array(["fist", "fist", "neutral", "neutral"])
        trials = np.array(["a", "a", "b", "b"])
        base = np.array([[0.1, 0.9], [0.1, 0.9], [0.9, 0.1], [0.9, 0.1]])
        improved = np.array([[0.9, 0.1], [0.9, 0.1], [0.9, 0.1], [0.9, 0.1]])
        result = paired_trial_audit(labels, trials, base, improved, np.array(["fist", "neutral"]))
        self.assertEqual(result["trial_count"], 2)
        self.assertEqual(result["corrected_trials"], 1)
        self.assertEqual(result["new_errors"], 0)


if __name__ == "__main__":
    unittest.main()
