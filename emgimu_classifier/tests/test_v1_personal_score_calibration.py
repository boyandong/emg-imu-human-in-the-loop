"""Regression checks for labelled target anchor isolation and saved predictions."""
from __future__ import annotations

import csv
import json

import numpy as np

from benchmarks.new_bank_v2.v1_personal_score_calibration import (
    PROTOCOL, ROOT, anchor_probability, digest, read_study,
)


def test_target_calibration_never_uses_evaluation_repetitions() -> None:
    audit = json.loads((ROOT / "V1_PERSONAL_SCORE_CAL_AUDIT.json").read_text(encoding="utf-8"))
    assert len(audit["assignments"]) == 400
    for assignment in audit["assignments"]:
        study, arm, phase, subject, budget = (assignment[k] for k in
                                             ("study", "arm", "phase", "subject", "shots_per_class"))
        native = read_study(study)
        expected_cal = {native[(arm, phase, subject, label, rep)]["trial_id"]
                        for label in PROTOCOL["classes"] for rep in range(1, budget + 1)}
        expected_eval = {native[(arm, phase, subject, label, rep)]["trial_id"]
                         for label in PROTOCOL["classes"] for rep in (6, 7)}
        assert set(assignment["calibration_ids"]) == expected_cal
        assert set(assignment["evaluation_ids"]) == expected_eval
        assert not expected_cal & expected_eval


def test_zero_shot_exact_parent_and_anchor_formula() -> None:
    verification = json.loads((ROOT / "V1_PERSONAL_SCORE_CAL_VERIFICATION.json").read_text(encoding="utf-8"))
    predictions_path = ROOT / "V1_PERSONAL_SCORE_CAL_PREDICTIONS.csv"
    assert verification["prediction_sha256"] == digest(predictions_path)
    assert verification["verified_predictions"] == 3200
    assert verification["zero_shot_parent_replays"] == 800
    with predictions_path.open(newline="", encoding="utf-8") as stream:
        predictions = list(csv.DictReader(stream))
    rows = [row for row in predictions if row["study"] == "grab_user"
            and row["arm"] == "F0v2+ring_lag" and row["phase"] == "validation"
            and int(row["subject"]) == 5 and row["trial_id"].endswith("_gesture4_trial6")]
    native = read_study("grab_user")
    source = native[("F0v2+ring_lag", "validation", 5, 4, 6)]["p"]
    assert len(rows) == 4
    prototypes = np.stack([np.mean([native[("F0v2+ring_lag", "validation", 5, label, rep)]["p"]
                                       for rep in (1, 2)], axis=0)
                           for label in PROTOCOL["classes"]])
    for row in rows:
        budget = int(row["shots_per_class"])
        actual = np.array([float(row[f"p_{c}"]) for c in PROTOCOL["classes"]])
        if budget == 0:
            np.testing.assert_array_equal(actual, source)
        if budget == 2:
            np.testing.assert_allclose(actual, 0.5 * source + 0.5 * anchor_probability(source, prototypes), atol=1e-12)
