import unittest

import numpy as np

from benchmarks.epn_spd_anchor_core_increment import matched_arrays
from emgimu.datasets.epn612 import GESTURES


class SpdAnchorCoreIncrementTests(unittest.TestCase):
    def fixture(self):
        users, labels, trials, selected, anchors = [], [], [], [], []
        anchor_probability = [0.4, 0.12, 0.12, 0.12, 0.12, 0.12]
        for user in (19, 20, 21):
            for index in range(7):
                users.append(user)
                labels.append(index % 6)
                trial = f"user{user}:trial{index}"
                trials.append(trial)
                if index < 6:
                    selected.append({"phase": "final", "subject": str(user),
                                     "shots_per_class": "1", "trial_id": trial})
                else:
                    anchors.append({"phase": "final", "subject": str(user),
                                    "shots_per_class": "1", "trial_id": trial,
                                    "true_label": "0", **{f"p_{gesture}": str(value)
                                    for gesture, value in zip(GESTURES, anchor_probability)}})
        core = {"users": np.asarray(users), "labels": np.asarray(labels), "trials": np.asarray(trials),
                "baseline": np.full((len(trials), 6), 1 / 6),
                "full": np.tile([0.5, 0.1, 0.1, 0.1, 0.1, 0.1], (len(trials), 1))}
        return core, anchors, selected

    def test_exact_trial_join_and_fixed_mixture(self):
        core, anchors, selected = self.fixture()
        truth, users, arms = matched_arrays(core, anchors, selected, 1)
        self.assertEqual(truth.tolist(), [0, 0, 0])
        self.assertEqual(users.tolist(), [19, 20, 21])
        self.assertEqual(len(arms["Core"]), 3)
        np.testing.assert_allclose(arms["Core_plus_SPD_Anchor"],
                                   (arms["Core"] + arms["SPD_Anchor"]) / 2)

    def test_missing_or_mislabeled_anchor_trial_is_rejected(self):
        core, anchors, selected = self.fixture()
        with self.assertRaisesRegex(ValueError, "do not equal"):
            matched_arrays(core, anchors[:-1], selected, 1)
        anchors[0]["true_label"] = "2"
        with self.assertRaisesRegex(ValueError, "labels disagree"):
            matched_arrays(core, anchors, selected, 1)

    def test_validation_phase_uses_separate_users_and_labels(self):
        core, anchors, selected = self.fixture()
        core["users"] -= 3
        core["trials"] = np.asarray([trial.replace("user19", "user16")
                                     .replace("user20", "user17")
                                     .replace("user21", "user18") for trial in core["trials"]])
        for row in anchors + selected:
            row["subject"] = str(int(row["subject"]) - 3)
            row["trial_id"] = row["trial_id"].replace("user19", "user16")\
                .replace("user20", "user17").replace("user21", "user18")
            row["phase"] = "validation"
        truth, users, _ = matched_arrays(core, anchors, selected, 1, "validation")
        self.assertEqual(truth.tolist(), [0, 0, 0])
        self.assertEqual(users.tolist(), [16, 17, 18])
        with self.assertRaisesRegex(ValueError, "unexpected"):
            matched_arrays(core, anchors, selected, 1, "final")


if __name__ == "__main__":
    unittest.main()
