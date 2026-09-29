"""Independently replay saved target feature-space anchor predictions."""
from __future__ import annotations

import csv
import json
import re

import numpy as np

from benchmarks.new_bank_v2.grab_score_calibration import score
from benchmarks.new_bank_v2.roam_posture_run import sha256
from benchmarks.new_bank_v2.v1_feature_anchor import (
    ARMS, CLASSES, PREFIX, PROTOCOL, PROTOCOL_PATH, ROOT, anchor_probability,
    check_protocol, frozen_probabilities,
)

PATTERN = re.compile(r"session(?P<day>[23])_participant(?P<subject>\d+)_gesture(?P<label>\d+)_trial(?P<repetition>[1-7])\Z")


def check_score(saved: dict, rows: list[dict]) -> None:
    metrics = score(rows)
    if int(saved["trials"]) != len(rows):
        raise AssertionError("curve trial count changed")
    for name in ("macro_f1", "accuracy", "log_loss", "brier", "ece"):
        if abs(float(saved[name]) - metrics[name]) > 1e-12:
            raise AssertionError(f"curve {name} changed")
    for label, value in metrics["per_class_f1"].items():
        if abs(json.loads(saved["per_class_f1"])[label] - value) > 1e-12:
            raise AssertionError("curve class F1 changed")


def verify() -> dict:
    check_protocol()
    result_path = ROOT / "V1_FEATURE_ANCHOR_RESULTS.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    prediction_path = ROOT / "V1_FEATURE_ANCHOR_PREDICTIONS.csv"
    curve_path = ROOT / "V1_FEATURE_ANCHOR_CURVE.csv"
    if (result["status"] != "ok" or result["protocol_sha256"] != sha256(PROTOCOL_PATH)
            or result["parent_prediction_sha256"] != sha256(ROOT / f"{PREFIX}_PREDICTIONS.csv")
            or result["prediction_sha256"] != sha256(prediction_path)
            or result["curve_sha256"] != sha256(curve_path)):
        raise AssertionError("feature-anchor provenance changed")
    with prediction_path.open(newline="", encoding="utf-8") as stream:
        predictions = list(csv.DictReader(stream))
    with curve_path.open(newline="", encoding="utf-8") as stream:
        curve = list(csv.DictReader(stream))
    if len(predictions) != 2560 or len(curve) != 360 or len(result["assignments"]) != 320:
        raise AssertionError("feature-anchor coverage changed")
    frozen = frozen_probabilities()
    target_ids = result["target_trial_ids"]
    if len(target_ids) != 448 or len(set(target_ids)) != 448:
        raise AssertionError("native target order changed")
    positions = {trial_id: i for i, trial_id in enumerate(target_ids)}
    indexed = {}
    for row in predictions:
        key = row["arm"], row["phase"], int(row["subject"]), int(row["shots_per_class"]), row["trial_id"]
        if key in indexed:
            raise AssertionError("duplicate adapted prediction")
        indexed[key] = row
    cells = {}
    for row in curve:
        key = row["arm"], row["phase"], row["subject"], int(row["shots_per_class"])
        if key in cells:
            raise AssertionError("duplicate curve cell")
        cells[key] = row
    assignments = {(a["arm"], a["phase"], a["subject"], a["shots_per_class"]): a
                   for a in result["assignments"]}
    if len(assignments) != 320:
        raise AssertionError("duplicate assignment")
    checked = set()
    maximum_error = 0.0
    for arm in ARMS:
        matrix_path = ROOT / f"V1_FEATURE_ANCHOR_{arm.replace('+', '_')}.npy"
        if sha256(matrix_path) != result["feature_sha256"][arm]:
            raise AssertionError("source-standardized features changed")
        matrix = np.load(matrix_path, allow_pickle=False)
        if matrix.shape != (448, result["feature_dimensions"][arm]) or not np.isfinite(matrix).all():
            raise AssertionError("target feature matrix changed")
        for phase, day in PROTOCOL["target_days"].items():
            for budget in PROTOCOL["budgets"]:
                pooled = []
                for subject in PROTOCOL["subjects"]:
                    assignment = assignments[(arm, phase, subject, budget)]
                    cal = {f"session{day}_participant{subject}_gesture{label}_trial{rep}"
                           for label in CLASSES for rep in range(1, budget + 1)}
                    ev = {f"session{day}_participant{subject}_gesture{label}_trial{rep}"
                          for label in CLASSES for rep in (6, 7)}
                    if (set(assignment["calibration_ids"]) != cal
                            or set(assignment["evaluation_ids"]) != ev or cal & ev):
                        raise AssertionError("anchor trial assignment changed")
                    prototype = (np.stack([np.mean([matrix[positions[
                        f"session{day}_participant{subject}_gesture{label}_trial{rep}"]]
                        for rep in range(1, budget + 1)], axis=0) for label in CLASSES])
                        if budget else None)
                    subject_rows = []
                    for trial_id in sorted(ev):
                        match = PATTERN.fullmatch(trial_id)
                        if match is None:
                            raise AssertionError("invalid native target identity")
                        label, rep = int(match["label"]), int(match["repetition"])
                        key = arm, phase, subject, budget, trial_id
                        row = indexed[key]
                        if int(row["label"]) != label or int(row["repetition"]) != rep:
                            raise AssertionError("adapted trial label changed")
                        population = frozen[(arm, trial_id)]
                        expected = population if budget == 0 else (
                            0.5 * population + 0.5 * anchor_probability(matrix[positions[trial_id]], prototype))
                        observed = np.asarray([float(row[f"p_{c}"]) for c in CLASSES])
                        maximum_error = max(maximum_error, float(np.max(np.abs(expected - observed))))
                        checked.add(key)
                        subject_rows.append({"label": label, **{f"p_{c}": float(p)
                                                             for c, p in zip(CLASSES, observed)}})
                    check_score(cells[(arm, phase, str(subject), budget)], subject_rows)
                    pooled.extend(subject_rows)
                check_score(cells[(arm, phase, "ALL", budget)], pooled)
    if checked != set(indexed) or maximum_error > 1e-12:
        raise AssertionError("unverified or changed feature-anchor prediction")
    audit = {"status": "ok", "protocol_sha256": sha256(PROTOCOL_PATH),
             "result_sha256": sha256(result_path), "prediction_sha256": sha256(prediction_path),
             "curve_sha256": sha256(curve_path), "verified_predictions": len(checked),
             "verified_score_cells": len(cells), "verified_assignments": len(assignments),
             "zero_shot_exact_parent_replays": sum(int(row["shots_per_class"] == "0")
                                                     for row in predictions),
             "maximum_absolute_probability_error": maximum_error}
    (ROOT / "V1_FEATURE_ANCHOR_VERIFICATION.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit))
    return audit


if __name__ == "__main__":
    verify()
