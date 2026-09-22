import unittest

import numpy as np

from benchmarks.quality_family_interactions import compositions, interaction


class QualityFamilyInteractionTests(unittest.TestCase):
    def test_fixed_four_arms_and_interaction_signs(self):
        probabilities = {
            "F0": np.array([[0.8, 0.2], [0.3, 0.7]]),
            "F1_X1H": np.array([[0.2, 0.8], [0.6, 0.4]]),
            "F9_Quality": np.array([[0.5, 0.5], [0.1, 0.9]]),
        }
        arms = compositions(probabilities, "F1_X1H")
        np.testing.assert_allclose(arms["B"], probabilities["F0"])
        np.testing.assert_allclose(arms["B_plus_family"],
                                   (probabilities["F0"] + probabilities["F1_X1H"]) / 2)
        np.testing.assert_allclose(arms["B_plus_quality"],
                                   (probabilities["F0"] + probabilities["F9_Quality"]) / 2)
        np.testing.assert_allclose(arms["B_plus_family_plus_quality"],
                                   sum(probabilities.values()) / 3)
        scored = {
            "B": {"log_loss": 2., "macro_f1": .4, "brier": .3},
            "B_plus_family": {"log_loss": 1.5, "macro_f1": .5, "brier": .2},
            "B_plus_quality": {"log_loss": 1.8, "macro_f1": .6, "brier": .25},
            "B_plus_family_plus_quality": {"log_loss": 1., "macro_f1": .8, "brier": .1},
        }
        effect = interaction(scored)
        self.assertAlmostEqual(effect["S_negative_logloss"], .3)
        self.assertAlmostEqual(effect["S_macro_f1"], .1)
        self.assertAlmostEqual(effect["S_negative_brier"], .05)
        with self.assertRaises(ValueError):
            compositions(probabilities, "unregistered_family")
