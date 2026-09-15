import tempfile
import unittest
from pathlib import Path
import numpy as np
from emgimu.feature_bank.wearing_session_study import calibration_indices
from benchmarks.consolidate_feature_bank import run_directory


class WearingSessionProtocolTests(unittest.TestCase):
    def test_one_shot_keeps_a_whole_native_trial_per_class(self):
        labels=np.repeat(np.arange(5),2)
        selected=calibration_indices(labels,1,18,'trial_1')
        self.assertEqual(len(selected),5)
        np.testing.assert_array_equal(np.sort(labels[selected]),np.arange(5))
        np.testing.assert_array_equal(np.sort(np.delete(labels,selected)),np.arange(5))
        np.testing.assert_array_equal(selected,calibration_indices(labels,1,18,'trial_1'))

    def test_unsupported_budgets_cannot_empty_evaluation(self):
        labels=np.repeat(np.arange(5),2)
        for shots in (2,5):
            with self.assertRaises(ValueError):calibration_indices(labels,shots,18,'trial_1')
        self.assertEqual(len(calibration_indices(labels,0,18,'trial_1')),0)

    def test_duplicate_source_roots_are_not_silently_selected(self):
        with tempfile.TemporaryDirectory() as folder:
            a=Path(folder)/'a';b=Path(folder)/'b'
            (a/'run').mkdir(parents=True);(b/'run').mkdir(parents=True)
            with self.assertRaises(ValueError):run_directory(a,'run',b)
            self.assertEqual(run_directory(a,'run'),a/'run')


if __name__=='__main__':unittest.main()
