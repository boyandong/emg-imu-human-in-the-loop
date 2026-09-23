import unittest

import numpy as np

from benchmarks.song_probability_calibration_audit import calibration_bins


class SongProbabilityCalibrationAuditTest(unittest.TestCase):
    def test_perfect_and_confidently_wrong_trials_have_expected_ece(self):
        labels=np.array(['a','b'])
        bins,ece=calibration_bins(np.array(['a','b']),np.array([[1.,0.],[0.,1.]]),labels)
        self.assertEqual(sum(row['trials'] for row in bins),2)
        self.assertEqual(ece,0)
        bins,ece=calibration_bins(np.array(['b','a']),np.array([[1.,0.],[0.,1.]]),labels)
        self.assertEqual(ece,1)

    def test_unknown_label_and_unnormalized_probability_rejected(self):
        labels=np.array(['a','b'])
        with self.assertRaises(ValueError):
            calibration_bins(['c'],[[.5,.5]],labels)
        with self.assertRaises(ValueError):
            calibration_bins(['a'],[[.8,.8]],labels)


if __name__ == '__main__':
    unittest.main()
