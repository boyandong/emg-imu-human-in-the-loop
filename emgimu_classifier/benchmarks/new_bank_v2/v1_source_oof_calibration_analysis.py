"""Verify source OOF temperatures and read back held-out calibration cells."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from benchmarks.new_bank_v2.family_screen import _score
from benchmarks.new_bank_v2.roam_posture_run import sha256

ROOT = Path(__file__).resolve().parent
STUDIES = {"roam_posture": ("ROAM_V1_EXTENSION", [0, 1, 2], 162, 180),
           "grab_user": ("GRAB_V1_EXTENSION", [4, 15, 16, 17], 112, 56),
           "grab_day": ("GRAB_DAY_V1_EXTENSION", [4, 15, 16, 17], 224, 224)}
FIELDS = ["study", "phase", "scope", "subject", "condition", "arm", "method",
          "temperature", "evaluation_unit", "evaluation_trials", "macro_f1", "accuracy",
          "log_loss", "brier", "ece", "per_class_f1_json"]


def calibrate(p: np.ndarray, temperature: float) -> np.ndarray:
    if temperature == 1.0:
        return p.copy()
    power = np.clip(p, 1e-12, 1.0) ** (1.0 / temperature)
    return power / power.sum(axis=1, keepdims=True)


def score_rows(rows: list[dict], classes: list[int], method: str) -> dict:
    labels = [str(c) for c in classes]
    prefix = "raw" if method == "uncalibrated" else "cal"
    source = [{"label": r["label"], **{f"p_{c}": r[f"{prefix}_p_{c}"] for c in classes}}
              for r in rows]
    return _score(source, labels)


def analyze() -> dict:
    result_path = ROOT / "V1_SOURCE_OOF_CAL_RESULTS.json"
    prediction_path = ROOT / "V1_SOURCE_OOF_CAL_PREDICTIONS.csv"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (result["protocol_sha256"] != sha256(ROOT / "V1_SOURCE_OOF_CAL_PROTOCOL.json")
            or result["prediction_sha256"] != sha256(prediction_path)
            or result["prediction_rows"] != 7090):
        raise ValueError("source OOF calibration frozen artifact changed")
    with prediction_path.open(newline="", encoding="utf-8") as stream:
        predictions = list(csv.DictReader(stream))
    keys = [(r["study"], r["partition"], r["arm"], r["trial_id"]) for r in predictions]
    if len(keys) != len(set(keys)):
        raise AssertionError("duplicate OOF or target native ID")
    output = []
    parent_hashes = {}
    for study, (prefix, classes, source_count, target_count) in STUDIES.items():
        parent_result = json.loads((ROOT / f"{prefix}_RESULTS.json").read_text(encoding="utf-8"))
        parent_path = ROOT / f"{prefix}_PREDICTIONS.csv"
        if sha256(parent_path) != parent_result["prediction_sha256"]:
            raise AssertionError(f"frozen target source changed: {study}")
        parent_hashes[study] = sha256(parent_path)
        with parent_path.open(newline="", encoding="utf-8") as stream:
            parent = {(r["arm"], r["phase"], r["trial_id"]): r for r in csv.DictReader(stream)}
        source_ids = set(parent_result["source_trial_ids"])
        target_ids = set(parent_result["validation_trial_ids"]) | set(parent_result["final_trial_ids"])
        if source_ids & target_ids:
            raise AssertionError("parent source/target native IDs overlap")
        folds = result["studies"][study]["source_folds"]
        seen_held = []
        for fold in folds:
            if fold["held_subject"] in fold["train_subjects"]:
                raise AssertionError("held OOF subject fitted within fold")
            if fold["train_trials"] + fold["held_trials"] != source_count:
                raise AssertionError("OOF fold source count changed")
            seen_held.extend(fold["held_trial_ids"])
        if len(seen_held) != source_count or set(seen_held) != source_ids:
            raise AssertionError("OOF held native IDs fail exact one-fold coverage")
        arm_names = parent_result["protocol"]["arms"]
        for arm in arm_names:
            arm_result = result["studies"][study]["arms"][arm]
            temperature = float(arm_result["temperature"])
            source = [r for r in predictions if r["study"] == study and
                      r["partition"] == "source_oof" and r["arm"] == arm]
            if len(source) != source_count or {r["trial_id"] for r in source} != source_ids:
                raise AssertionError("source OOF native trial count or identity changed")
            source_y = np.asarray([int(r["label"]) for r in source])
            raw = np.asarray([[float(r[f"raw_p_{c}"]) for c in classes] for r in source])
            scored = [(float(-np.mean(np.log(np.clip(
                calibrate(raw, float(t))[np.arange(len(source_y)),
                                      np.searchsorted(classes, source_y)], 1e-12, 1.0)))),
                       abs(float(np.log(t))), float(t)) for t in result["temperature_candidates"]]
            chosen = min(scored)[2]
            if chosen != temperature:
                raise AssertionError("source OOF temperature not grid-selected")
            for method, field in (("uncalibrated", "source_oof_raw"),
                                  ("source_oof_temperature", "source_oof_calibrated")):
                replay = score_rows(source, classes, method)
                for metric in ("macro_f1", "accuracy", "log_loss", "brier", "ece"):
                    if abs(replay[metric] - arm_result[field][metric]) > 1e-12:
                        raise AssertionError(f"source score read-back failed: {study}/{arm}/{metric}")
            for phase in ("validation", "final"):
                target = [r for r in predictions if r["study"] == study and
                          r["partition"] == phase and r["arm"] == arm]
                expected_ids = set(parent_result[f"{phase}_trial_ids"])
                if len(target) != target_count or {r["trial_id"] for r in target} != expected_ids:
                    raise AssertionError("held-out target count or identity changed")
                for row in target:
                    prior = parent[(arm, phase, row["trial_id"])]
                    label_field = "label" if study == "roam_posture" else "gesture"
                    if int(row["label"]) != int(prior[label_field]):
                        raise AssertionError("held-out target label changed")
                    p = np.asarray([float(row[f"raw_p_{c}"]) for c in classes])
                    q = np.asarray([float(row[f"cal_p_{c}"]) for c in classes])
                    np.testing.assert_allclose(p, [float(prior[f"p_{c}"]) for c in classes],
                                               atol=1e-12, rtol=0)
                    np.testing.assert_allclose(q, calibrate(p[None, :], temperature)[0],
                                               atol=1e-12, rtol=0)
                groups = [("pooled", "ALL", "ALL", target)]
                for subject in sorted({int(r["subject"]) for r in target}):
                    groups.append(("subject", str(subject), "ALL",
                                   [r for r in target if int(r["subject"]) == subject]))
                if study == "roam_posture":
                    for posture in parent_result["protocol"]["target_postures"]:
                        groups.append(("posture", "ALL", posture,
                                       [r for r in target if r["condition"] == posture]))
                for scope, subject, condition, group in groups:
                    for method in ("uncalibrated", "source_oof_temperature"):
                        values = score_rows(group, classes, method)
                        if scope == "pooled":
                            expected = arm_result["target"][phase][
                                "raw" if method == "uncalibrated" else "calibrated"]
                            for metric in ("macro_f1", "accuracy", "log_loss", "brier", "ece"):
                                if abs(values[metric] - expected[metric]) > 1e-12:
                                    raise AssertionError("held-out target score replay failed")
                        output.append({"study": study, "phase": phase, "scope": scope,
                                       "subject": subject, "condition": condition,
                                       "arm": arm, "method": method, "temperature": temperature,
                                       "evaluation_unit": "native_label_bout" if study == "roam_posture"
                                                          else "native_recording",
                                       "evaluation_trials": values["trials"],
                                       **{key: values[key] for key in ("macro_f1", "accuracy",
                                                                         "log_loss", "brier", "ece",
                                                                         "per_class_f1_json")}})
    if len(output) != 440:
        raise AssertionError("source-calibrated target cell count changed")
    table = ROOT / "V1_SOURCE_OOF_CAL_CELLS.csv"
    with table.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output)
    audit = {"status": "ok", "prediction_rows": len(predictions),
             "target_score_cells": len(output), "source_result_sha256": sha256(result_path),
             "source_prediction_sha256": sha256(prediction_path),
             "parent_prediction_sha256": parent_hashes, "cell_sha256": sha256(table),
             "boundary": "Subject-held source OOF chooses zero-target-shot temperature. Target user/day labels never tune it; public shifts remain and personal shots are not modeled."}
    (ROOT / "V1_SOURCE_OOF_CAL_VERIFICATION.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"source OOF calibration read-back: {len(output)} target cells", flush=True)
    return audit


if __name__ == "__main__":
    analyze()
