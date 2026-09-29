"""Native-split and frozen-source checks for new-v1 feature anchors."""
from __future__ import annotations

import csv
import json

import numpy as np

from benchmarks.new_bank_v2.roam_posture_run import sha256
from benchmarks.new_bank_v2.v1_feature_anchor import CLASSES, PROTOCOL, ROOT, frozen_probabilities
from benchmarks.new_bank_v2.v1_feature_anchor_verify import verify


def test_feature_anchor_target_assignments_are_disjoint_and_source_frozen() -> None:
    result = json.loads((ROOT / "V1_FEATURE_ANCHOR_RESULTS.json").read_text(encoding="utf-8"))
    assert result["parent_prediction_sha256"] == sha256(ROOT / "GRAB_DAY_V1_EXTENSION_PREDICTIONS.csv")
    assert len(result["assignments"]) == 320
    assert len(result["target_trial_ids"]) == 448
    assert set(result["source_replay_max_abs_error"].values()) == {0.0}
    for arm, digest in result["feature_sha256"].items():
        matrix_path = ROOT / f"V1_FEATURE_ANCHOR_{arm.replace('+', '_')}.npy"
        matrix = np.load(matrix_path, allow_pickle=False)
        assert matrix.shape == (448, result["feature_dimensions"][arm])
        assert sha256(matrix_path) == digest
    for assignment in result["assignments"]:
        arm, phase, subject, budget = (assignment[k] for k in
                                       ("arm", "phase", "subject", "shots_per_class"))
        day = PROTOCOL["target_days"][phase]
        assert set(assignment["calibration_ids"]) == {
            f"session{day}_participant{subject}_gesture{label}_trial{rep}"
            for label in CLASSES for rep in range(1, budget + 1)}
        assert set(assignment["evaluation_ids"]) == {
            f"session{day}_participant{subject}_gesture{label}_trial{rep}"
            for label in CLASSES for rep in (6, 7)}
        assert not set(assignment["calibration_ids"]) & set(assignment["evaluation_ids"])


def test_feature_anchor_readback_and_exact_zero_shot() -> None:
    audit = verify()
    assert audit["verified_predictions"] == 2560
    assert audit["verified_score_cells"] == 360
    assert audit["zero_shot_exact_parent_replays"] == 640
    assert audit["maximum_absolute_probability_error"] == 0.0
    frozen = frozen_probabilities()
    with (ROOT / "V1_FEATURE_ANCHOR_PREDICTIONS.csv").open(newline="", encoding="utf-8") as stream:
        zero = [row for row in csv.DictReader(stream) if row["shots_per_class"] == "0"]
    assert len(zero) == 640
    for row in zero:
        observed = np.asarray([float(row[f"p_{c}"]) for c in CLASSES])
        np.testing.assert_array_equal(observed, frozen[(row["arm"], row["trial_id"])])
