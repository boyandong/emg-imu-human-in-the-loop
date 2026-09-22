import unittest

import numpy as np

from benchmarks.unibo_g5_temporal_interaction import SPECS, concatenate, interaction


class G5TemporalInteractionTests(unittest.TestCase):
    def test_four_arm_feature_order_and_second_difference(self):
        self.assertEqual(SPECS["B"], ("G0",))
        self.assertEqual(SPECS["B_plus_G5_plus_Temporal"], ("G0", "G5", "Temporal"))
        features = {"G0": np.array([[1., 2.]]), "G5": np.array([[3.]]),
                    "Temporal": np.array([[4., 5.]])}
        np.testing.assert_array_equal(concatenate(features, SPECS["B_plus_G5_plus_Temporal"]),
                                      [[1., 2., 3., 4., 5.]])
        scores = {
            "B": {"log_loss": 2., "macro_f1": .4, "brier": .3},
            "B_plus_G5": {"log_loss": 1.5, "macro_f1": .5, "brier": .2},
            "B_plus_Temporal": {"log_loss": 1.8, "macro_f1": .6, "brier": .25},
            "B_plus_G5_plus_Temporal": {"log_loss": 1., "macro_f1": .8, "brier": .1},
        }
        observed = interaction(scores)
        self.assertAlmostEqual(observed["S_negative_logloss"], .3)
        self.assertAlmostEqual(observed["S_macro_f1"], .1)
        self.assertAlmostEqual(observed["S_negative_brier"], .05)
