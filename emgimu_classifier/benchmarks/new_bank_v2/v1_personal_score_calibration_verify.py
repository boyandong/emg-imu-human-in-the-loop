"""Read-back verification of frozen native calibration/evaluation assignments."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from benchmarks.new_bank_v2.grab_score_calibration import score
from benchmarks.new_bank_v2.v1_personal_score_calibration import (
    ARMS, CLASSES, PROTOCOL, ROOT, anchor_probability, check_parents, digest, read_study,
)


def verify() -> dict:
    check_parents()
    audit_path = ROOT / "V1_PERSONAL_SCORE_CAL_AUDIT.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    predictions_path = ROOT / "V1_PERSONAL_SCORE_CAL_PREDICTIONS.csv"
    curve_path = ROOT / "V1_PERSONAL_SCORE_CAL_CURVE.csv"
    if (audit["protocol_sha256"] != digest(ROOT / "V1_PERSONAL_SCORE_CAL_PROTOCOL.json")
            or audit["predictions_sha256"] != digest(predictions_path)
            or audit["curve_sha256"] != digest(curve_path)):
        raise AssertionError("calibration provenance changed")
    with predictions_path.open(newline="", encoding="utf-8") as stream:
        predictions = list(csv.DictReader(stream))
    with curve_path.open(newline="", encoding="utf-8") as stream:
        curve = list(csv.DictReader(stream))
    if len(predictions) != 3200 or len(curve) != 480 or len(audit["assignments"]) != 400:
        raise AssertionError("calibration coverage changed")
    indexed_predictions = {}
    for row in predictions:
        key = row["study"], row["arm"], row["phase"], int(row["subject"]), int(row["shots_per_class"]), row["trial_id"]
        if key in indexed_predictions:
            raise AssertionError("duplicate calibration prediction")
        indexed_predictions[key] = row
    indexed_curve = {}
    for row in curve:
        key = row["study"], row["arm"], row["phase"], row["subject"], int(row["shots_per_class"])
        if key in indexed_curve:
            raise AssertionError("duplicate calibration score cell")
        indexed_curve[key] = row
    max_probability_error = 0.0
    checked = set()
    for study, spec in PROTOCOL["studies"].items():
        native = read_study(study)
        for arm in ARMS:
            for phase in ("validation", "final"):
                for budget in PROTOCOL["budgets"]:
                    pooled = []
                    for subject in spec["subjects"][phase]:
                        assignment = next((a for a in audit["assignments"]
                                           if (a["study"], a["arm"], a["phase"], a["subject"], a["shots_per_class"])
                                           == (study, arm, phase, subject, budget)), None)
                        if assignment is None:
                            raise AssertionError("missing native split assignment")
                        cal_expected = {native[(arm, phase, subject, label, rep)]["trial_id"]
                                        for label in CLASSES for rep in range(1, budget + 1)}
                        eval_expected = {native[(arm, phase, subject, label, rep)]["trial_id"]
                                         for label in CLASSES for rep in (6, 7)}
                        if (set(assignment["calibration_ids"]) != cal_expected
                                or set(assignment["evaluation_ids"]) != eval_expected
                                or cal_expected & eval_expected):
                            raise AssertionError("native calibration/evaluation split changed")
                        prototype = (np.stack([np.mean([native[(arm, phase, subject, label, rep)]["p"]
                                                        for rep in range(1, budget + 1)], axis=0)
                                               for label in CLASSES]) if budget else None)
                        subject_rows = []
                        for label in CLASSES:
                            for rep in (6, 7):
                                source = native[(arm, phase, subject, label, rep)]
                                key = study, arm, phase, subject, budget, source["trial_id"]
                                row = indexed_predictions[key]
                                if int(row["label"]) != label or int(row["repetition"]) != rep:
                                    raise AssertionError("prediction native label changed")
                                expected = source["p"] if budget == 0 else (
                                    0.5 * source["p"] + 0.5 * anchor_probability(source["p"], prototype))
                                actual = np.array([float(row[f"p_{c}"]) for c in CLASSES])
                                max_probability_error = max(max_probability_error,
                                                            float(np.max(np.abs(actual - expected))))
                                checked.add(key)
                                subject_rows.append({"label": label, **{f"p_{c}": float(p)
                                                                            for c, p in zip(CLASSES, actual)}})
                        pooled.extend(subject_rows)
                        _check_curve(indexed_curve[(study, arm, phase, str(subject), budget)], subject_rows)
                    _check_curve(indexed_curve[(study, arm, phase, "ALL", budget)], pooled)
    if checked != set(indexed_predictions) or max_probability_error > 1e-12:
        raise AssertionError("incomplete or changed calibrated predictions")
    result = {"status": "ok", "protocol_sha256": audit["protocol_sha256"],
              "prediction_sha256": digest(predictions_path), "curve_sha256": digest(curve_path),
              "audit_sha256": digest(audit_path), "verified_predictions": len(checked),
              "verified_score_cells": len(curve), "verified_assignments": len(audit["assignments"]),
              "maximum_absolute_probability_error": max_probability_error,
              "zero_shot_parent_replays": sum(int(row["shots_per_class"] == "0") for row in predictions)}
    (ROOT / "V1_PERSONAL_SCORE_CAL_VERIFICATION.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
    return result


def _check_curve(record: dict, rows: list[dict]) -> None:
    observed = score(rows)
    if int(record["trials"]) != len(rows):
        raise AssertionError("score cell trial count changed")
    for metric in ("macro_f1", "accuracy", "log_loss", "brier", "ece"):
        if abs(float(record[metric]) - observed[metric]) > 1e-12:
            raise AssertionError(f"score cell {metric} changed")
    for label, value in observed["per_class_f1"].items():
        if abs(json.loads(record["per_class_f1"])[label] - value) > 1e-12:
            raise AssertionError("per-class score changed")


if __name__ == "__main__":
    verify()
