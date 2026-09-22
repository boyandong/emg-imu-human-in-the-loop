import unittest

import numpy as np

from benchmarks.epn_anchor_spatial_interaction import ARMS, compositions, interaction


class EpnAnchorSpatialInteractionTests(unittest.TestCase):
    def test_fixed_four_arms_and_second_difference(self):
        raw = {
            "F0": np.array([[.8, .2], [.3, .7]]),
            "F2b_CSP": np.array([[.4, .6], [.5, .5]]),
        }
        personal = {
            "F0": np.array([[.9, .1], [.2, .8]]),
            "F2b_CSP": np.array([[.6, .4], [.4, .6]]),
        }
        arms = compositions(raw, personal)
        self.assertEqual(tuple(arms), ARMS)
        np.testing.assert_allclose(arms["B"], raw["F0"])
        np.testing.assert_allclose(arms["B_plus_CSP"], (raw["F0"] + raw["F2b_CSP"]) / 2)
        np.testing.assert_allclose(arms["B_plus_Anchor"], personal["F0"])
        np.testing.assert_allclose(arms["B_plus_CSP_plus_Anchor"],
                                   (personal["F0"] + personal["F2b_CSP"]) / 2)
        scores = {
            "B": {"log_loss": 2., "macro_f1": .4, "brier": .3},
            "B_plus_CSP": {"log_loss": 1.5, "macro_f1": .5, "brier": .2},
            "B_plus_Anchor": {"log_loss": 1.8, "macro_f1": .6, "brier": .25},
            "B_plus_CSP_plus_Anchor": {"log_loss": 1., "macro_f1": .8, "brier": .1},
        }
        observed = interaction(scores)
        self.assertAlmostEqual(observed["S_negative_logloss"], .3)
        self.assertAlmostEqual(observed["S_macro_f1"], .1)
        self.assertAlmostEqual(observed["S_negative_brier"], .05)

    def test_provider_contract_rejects_missing_or_misaligned_inputs(self):
        aligned = {"F0": np.ones((2, 2)) / 2, "F2b_CSP": np.ones((2, 2)) / 2}
        with self.assertRaises(ValueError):
            compositions({"F0": aligned["F0"]}, aligned)
        broken = {**aligned, "F2b_CSP": np.ones((3, 2)) / 2}
        with self.assertRaises(ValueError):
            compositions(aligned, broken)


if __name__ == "__main__":
    unittest.main()
