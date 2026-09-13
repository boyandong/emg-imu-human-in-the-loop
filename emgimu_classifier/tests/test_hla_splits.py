from __future__ import annotations

import unittest

from emgimu.datasets.hla_splits import TrialRef, build_personalization_split


def records() -> list[TrialRef]:
    result: list[TrialRef] = []
    for subject in ("s1", "s2", "s3"):
        for session in ("day1", "day2"):
            for label in (0, 1, 2):
                for repetition in range(5):
                    trial = f"{subject}-{session}-g{label}-r{repetition}"
                    for view in ("forearm", "wrist"):
                        result.append(TrialRef(
                            "dataset", subject, session, trial, label,
                            2.5 if label == 0 else 1.0, view,
                        ))
    return result


class HlaSplitTests(unittest.TestCase):
    def test_calibration_uses_whole_trials_and_keeps_sensor_views_together(self):
        split = build_personalization_split(
            records(), target_subject="s3", calibration_session="day1",
            active_task_labels=(1, 2), neutral_task_label=0,
            gesture_budget_per_class=3, rest_seconds=5.0, seed=42,
        )
        split.assert_disjoint()
        calibration_keys = {row.trial_key for row in split.calibration}
        for key in calibration_keys:
            views = {row.sensor_view for row in split.calibration if row.trial_key == key}
            self.assertEqual(views, {"forearm", "wrist"})
        active = {label: set() for label in (1, 2)}
        for row in split.calibration:
            if row.task_label in active:
                active[row.task_label].add(row.trial_key)
        self.assertEqual({label: len(keys) for label, keys in active.items()}, {1: 3, 2: 3})
        rest_keys = {row.trial_key for row in split.calibration if row.task_label == 0}
        self.assertEqual(len(rest_keys), 2)
        self.assertTrue(all(row.subject_id != "s3" for row in split.source_training))
        self.assertTrue(all(row.session_id == "day1" for row in split.same_session_evaluation))
        self.assertTrue(all(row.session_id == "day2" for row in split.cross_session_evaluation))

    def test_zero_shot_has_no_calibration_trials(self):
        split = build_personalization_split(
            records(), target_subject="s3", calibration_session=None,
            active_task_labels=(1, 2), neutral_task_label=0,
            gesture_budget_per_class=0, rest_seconds=0.0, seed=42,
        )
        self.assertFalse(split.calibration)
        self.assertEqual(
            len({row.trial_key for row in split.same_session_evaluation}), 15,
        )

    def test_seed_is_deterministic_and_changes_selected_trials(self):
        arguments = dict(
            target_subject="s3", calibration_session="day1",
            active_task_labels=(1, 2), neutral_task_label=0,
            gesture_budget_per_class=2, rest_seconds=2.5,
        )
        first = build_personalization_split(records(), seed=42, **arguments)
        repeated = build_personalization_split(records(), seed=42, **arguments)
        other = build_personalization_split(records(), seed=43, **arguments)
        self.assertEqual(first.calibration, repeated.calibration)
        self.assertNotEqual(
            {row.trial_key for row in first.calibration},
            {row.trial_key for row in other.calibration},
        )

    def test_insufficient_budget_fails_instead_of_reusing_trials(self):
        with self.assertRaisesRegex(ValueError, "needs 6"):
            build_personalization_split(
                records(), target_subject="s3", calibration_session="day1",
                active_task_labels=(1, 2), neutral_task_label=0,
                gesture_budget_per_class=6, rest_seconds=0.0, seed=42,
            )


if __name__ == "__main__":
    unittest.main()
