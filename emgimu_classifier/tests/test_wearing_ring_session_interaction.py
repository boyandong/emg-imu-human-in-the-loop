import unittest

import numpy as np

from benchmarks.wearing_ring_session_interaction import ARMS, compositions, interaction


class WearingRingSessionInteractionTests(unittest.TestCase):
    def test_identifiable_four_arms_and_second_difference(self):
        probabilities = {
            "F0": np.array([[.8, .2], [.4, .6]]),
            "F2b_CSP": np.array([[.6, .4], [.3, .7]]),
            "F3_Ring": np.array([[.2, .8], [.7, .3]]),
        }
        context = {"F0": .8, "F2b_CSP": .4, "F3_Ring": .2}
        arms = compositions(probabilities, context)
        self.assertEqual(tuple(arms), ARMS)
        np.testing.assert_allclose(arms["B"], (probabilities["F0"] + probabilities["F2b_CSP"]) / 2)
        np.testing.assert_allclose(arms["B_plus_Ring"], sum(probabilities.values()) / 3)
        np.testing.assert_allclose(arms["B_plus_Session"],
                                   (2 * probabilities["F0"] + probabilities["F2b_CSP"]) / 3)
        scores = {
            "B": {"log_loss": 2., "macro_f1": .4, "brier": .3},
            "B_plus_Ring": {"log_loss": 1.5, "macro_f1": .5, "brier": .2},
            "B_plus_Session": {"log_loss": 1.8, "macro_f1": .6, "brier": .25},
            "B_plus_Ring_plus_Session": {"log_loss": 1., "macro_f1": .8, "brier": .1},
        }
        observed = interaction(scores)
        self.assertAlmostEqual(observed["S_negative_logloss"], .3)
        self.assertAlmostEqual(observed["S_macro_f1"], .1)
        self.assertAlmostEqual(observed["S_negative_brier"], .05)

    def test_contract_rejects_missing_family_and_nonpositive_context(self):
        probabilities = {name: np.ones((2, 2)) / 2 for name in ("F0", "F2b_CSP", "F3_Ring")}
        with self.assertRaises(ValueError):
            compositions({"F0": probabilities["F0"]}, {name: 1. for name in probabilities})
        with self.assertRaises(ValueError):
            compositions(probabilities, {"F0": 1., "F2b_CSP": 0., "F3_Ring": 1.})


if __name__ == "__main__":
    unittest.main()
