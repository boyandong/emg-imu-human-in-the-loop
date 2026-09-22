import unittest

import numpy as np

from benchmarks.recover_prediction_disagreement import check_source_rates


class PredictionDisagreementRecoveryTests(unittest.TestCase):
    def test_joint_wrong_predictions_count_only_in_class_disagreement(self):
        truth = np.array([0, 0, 0, 0])
        a = np.array([0, 1, 1, 2])
        b = np.array([0, 0, 2, 1])
        row = {'disagreement': '0.25', 'a_correct_b_wrong': '0',
               'a_wrong_b_correct': '0.25'}
        self.assertEqual(check_source_rates(row, truth, a, b, np.ones(4), 'fixture'), .75)

    def test_wrong_source_pair_metric_rejects_replay(self):
        truth = np.array([0, 1])
        a = np.array([0, 1])
        b = np.array([1, 1])
        row = {'disagreement': '0.5', 'a_correct_b_wrong': '0',
               'a_wrong_b_correct': '0'}
        with self.assertRaisesRegex(ValueError, 'a_correct_b_wrong'):
            check_source_rates(row, truth, a, b, np.ones(2), 'fixture')
