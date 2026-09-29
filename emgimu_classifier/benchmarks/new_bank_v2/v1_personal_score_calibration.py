"""Frozen 0/1/2/5-shot score-space anchors for independent family arms."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from benchmarks.new_bank_v2.grab_score_calibration import csv_text, score

ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "V1_PERSONAL_SCORE_CAL_PROTOCOL.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
CLASSES = tuple(PROTOCOL["classes"])
ARMS = tuple(PROTOCOL["arms"])
PATTERN = re.compile(
    r"session(?P<day>[123])_participant(?P<subject>\d+)_gesture(?P<gesture>\d+)_trial(?P<repetition>[1-7])\Z"
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_parents() -> None:
    if len(ARMS) != 5 or len(set(ARMS)) != 5 or CLASSES != (4, 15, 16, 17):
        raise AssertionError("frozen arms or classes changed")
    if PROTOCOL["budgets"] != [0, 1, 2, 5] or PROTOCOL["evaluation_repetitions"] != [6, 7]:
        raise AssertionError("frozen calibration schedule changed")
    for parent, expected in PROTOCOL["parent_sha256"].items():
        for suffix, sha256 in expected.items():
            path = ROOT / f"{parent}_{suffix.upper()}.{ 'csv' if suffix == 'predictions' else 'json'}"
            if digest(path) != sha256:
                raise AssertionError(f"frozen parent changed: {path.name}")


def read_study(study: str) -> dict:
    spec = PROTOCOL["studies"][study]
    with (ROOT / f"{spec['parent']}_PREDICTIONS.csv").open(newline="", encoding="utf-8") as stream:
        source = list(csv.DictReader(stream))
    indexed = {}
    for row in source:
        arm, phase = row["arm"], row["phase"]
        if arm not in ARMS or phase not in ("validation", "final"):
            raise AssertionError("unexpected parent arm or phase")
        match = PATTERN.fullmatch(row["trial_id"])
        if match is None:
            raise AssertionError("invalid native trial ID")
        day, subject, label, repetition = (int(match[k]) for k in
                                           ("day", "subject", "gesture", "repetition"))
        if (day != spec["days"][phase] or subject not in spec["subjects"][phase]
                or subject != int(row["subject"]) or label != int(row["gesture"])
                or label not in CLASSES):
            raise AssertionError("parent prediction conflicts with native trial identity")
        p = np.array([float(row[f"p_{c}"]) for c in CLASSES], dtype=float)
        if not np.isfinite(p).all() or (p < 0).any() or not np.isclose(p.sum(), 1, atol=1e-12):
            raise AssertionError("invalid source probability")
        key = arm, phase, subject, label, repetition
        if key in indexed:
            raise AssertionError("duplicate native trial")
        indexed[key] = {"trial_id": row["trial_id"], "p": p}
    expected = len(ARMS) * sum(len(spec["subjects"][phase]) for phase in ("validation", "final")) * len(CLASSES) * 7
    if len(indexed) != expected:
        raise AssertionError(f"{study}: expected {expected} source predictions, got {len(indexed)}")
    return indexed


def anchor_probability(query: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    logits = -np.sum((prototypes - query[None, :]) ** 2, axis=1) / PROTOCOL["score_anchor_temperature"]
    weights = np.exp(logits - logits.max())
    return weights / weights.sum()


def evaluate() -> dict:
    check_parents()
    rows, curve, assignments = [], [], []
    for study, spec in PROTOCOL["studies"].items():
        indexed = read_study(study)
        for arm in ARMS:
            for phase in ("validation", "final"):
                for budget in PROTOCOL["budgets"]:
                    phase_rows = []
                    for subject in spec["subjects"][phase]:
                        calibration_ids = []
                        if budget:
                            prototypes = np.stack([np.mean([
                                indexed[(arm, phase, subject, label, rep)]["p"]
                                for rep in range(1, budget + 1)], axis=0)
                                for label in CLASSES])
                            calibration_ids = [indexed[(arm, phase, subject, label, rep)]["trial_id"]
                                               for label in CLASSES for rep in range(1, budget + 1)]
                        evaluation_ids = []
                        for label in CLASSES:
                            for rep in PROTOCOL["evaluation_repetitions"]:
                                source = indexed[(arm, phase, subject, label, rep)]
                                evaluation_ids.append(source["trial_id"])
                                p0 = source["p"]
                                p = p0 if budget == 0 else (
                                    PROTOCOL["population_weight"] * p0
                                    + (1 - PROTOCOL["population_weight"]) * anchor_probability(p0, prototypes))
                                if not np.isfinite(p).all() or (p < 0).any() or not np.isclose(p.sum(), 1, atol=1e-12):
                                    raise AssertionError("invalid adapted probability")
                                phase_rows.append({"study": study, "arm": arm, "phase": phase,
                                                   "subject": subject, "shots_per_class": budget,
                                                   "trial_id": source["trial_id"], "label": label,
                                                   "repetition": rep,
                                                   **{f"p_{c}": float(value) for c, value in zip(CLASSES, p)}})
                        if set(calibration_ids) & set(evaluation_ids):
                            raise AssertionError("calibration/evaluation leakage")
                        assignments.append({"study": study, "arm": arm, "phase": phase,
                                            "subject": subject, "shots_per_class": budget,
                                            "calibration_ids": calibration_ids,
                                            "evaluation_ids": evaluation_ids})
                    rows.extend(phase_rows)
                    for group_subject in ("ALL", *spec["subjects"][phase]):
                        subset = phase_rows if group_subject == "ALL" else [
                            row for row in phase_rows if row["subject"] == group_subject]
                        metrics = score(subset)
                        curve.append({"study": study, "arm": arm, "phase": phase,
                                      "subject": group_subject, "shots_per_class": budget,
                                      "signal_seconds": budget * len(CLASSES) * PROTOCOL["recording_seconds"],
                                      **{k: (json.dumps(v, sort_keys=True) if k == "per_class_f1" else v)
                                         for k, v in metrics.items()}})
                print(f"{study}/{arm}/{phase}: calibrated four budgets", flush=True)
    if len(rows) != 3200 or len(curve) != 480 or len(assignments) != 400:
        raise AssertionError("calibration output coverage changed")
    predictions_path = ROOT / "V1_PERSONAL_SCORE_CAL_PREDICTIONS.csv"
    curve_path = ROOT / "V1_PERSONAL_SCORE_CAL_CURVE.csv"
    predictions_path.write_text(csv_text(rows), encoding="utf-8", newline="")
    curve_path.write_text(csv_text(curve), encoding="utf-8", newline="")
    audit = {"status": "ok", "protocol_sha256": digest(PROTOCOL_PATH),
             "parent_sha256": PROTOCOL["parent_sha256"],
             "predictions_sha256": digest(predictions_path), "curve_sha256": digest(curve_path),
             "prediction_rows": len(rows), "curve_rows": len(curve),
             "assignments": assignments,
             "scope": PROTOCOL["scope"]}
    (ROOT / "V1_PERSONAL_SCORE_CAL_AUDIT.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} predictions and {len(curve)} score cells", flush=True)
    return audit


if __name__ == "__main__":
    evaluate()
