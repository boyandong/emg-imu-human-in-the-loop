"""Independently verify frozen force calibration trial allocation and scores."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, recall_score

from emgimu.datasets.libemg_force import FILE_RE

ROOT = Path(__file__).resolve().parent
CLASSES = np.arange(7)
ARMS = ("F0v2", "F0v2+F2a")
METHODS = {"source_only": 0, "one_shot": 1, "two_shot": 2}


def _score(rows: list[dict]) -> dict:
    y = np.asarray([int(r["label"]) for r in rows])
    p = np.asarray([[float(r[f"p_{label}"]) for label in CLASSES] for r in rows])
    pred = p.argmax(axis=1)
    result = {"trials": len(rows), "accuracy": float(accuracy_score(y, pred)),
              "macro_f1": float(f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0)),
              "log_loss": float(log_loss(y, p, labels=CLASSES)),
              "brier": float(np.mean(np.sum((p - np.eye(7)[y]) ** 2, axis=1))),
              "per_class_recall": {str(c): float(v) for c, v in zip(
                  CLASSES, recall_score(y, pred, labels=CLASSES, average=None, zero_division=0))}}
    return result


def _compare(recorded: dict, recomputed: dict, context: str) -> None:
    if set(recorded) != set(recomputed):
        raise AssertionError(f"score keys changed: {context}")
    for key, value in recomputed.items():
        if isinstance(value, dict):
            _compare(recorded[key], value, context + "/" + key)
        elif not np.isclose(recorded[key], value, rtol=0, atol=1e-12):
            raise AssertionError(f"score mismatch: {context}/{key}")


def verify() -> dict:
    protocol = ROOT / "FORCE_CAL_PROTOCOL.json"
    result = json.loads((ROOT / "FORCE_CAL_RESULTS.json").read_text(encoding="utf-8"))
    if result["protocol_sha256"] != hashlib.sha256(protocol.read_bytes()).hexdigest():
        raise AssertionError("calibration protocol changed")
    with (ROOT / "FORCE_CAL_PREDICTIONS.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    with (ROOT / "FORCE_TRIAL_PREDICTIONS.csv").open(newline="", encoding="utf-8") as stream:
        parent_rows = list(csv.DictReader(stream))
    if len(rows) != 3360 or len(parent_rows) != 5880 or len(result["assignments"]) != 40:
        raise AssertionError("frozen prediction or assignment count changed")
    if result["parent_predictions_sha256"] != hashlib.sha256(
            (ROOT / "FORCE_TRIAL_PREDICTIONS.csv").read_bytes()).hexdigest():
        raise AssertionError("parent predictions changed")
    parent = {(r["phase"], r["arm"], r["trial_id"]): r for r in parent_rows}
    lookup: dict[tuple[str, str, str, str], dict] = {}
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    assignments = result["assignments"]
    for row in rows:
        key = (row["phase"], row["arm"], row["method"], row["trial_id"])
        if key in lookup or row["arm"] not in ARMS or row["method"] not in METHODS:
            raise AssertionError("duplicate or unexpected prediction key")
        lookup[key] = row
        match = FILE_RE.fullmatch(row["trial_id"] + ".csv")
        assignment = f"{row['phase']}_{row['subject']}_{row['condition']}"
        if (match is None or assignment not in assignments or
                int(match["subject"]) != int(row["subject"]) or
                match["condition"] != row["condition"] or
                int(match["label"]) - 1 != int(row["label"]) or
                int(match["repetition"]) not in (3, 4) or
                row["trial_id"] not in assignments[assignment]["evaluation"]):
            raise AssertionError("evaluation trial identity invalid")
        p = np.asarray([float(row[f"p_{c}"]) for c in CLASSES])
        if not np.all(np.isfinite(p)) or np.any(p < 0) or not np.isclose(p.sum(), 1, atol=1e-12):
            raise AssertionError("invalid probability vector")
        if row["method"] == "source_only":
            frozen = parent[(row["phase"], row["arm"], row["trial_id"])]
            if not np.allclose(p, [float(frozen[f"p_{c}"]) for c in CLASSES], atol=1e-12, rtol=0):
                raise AssertionError("source-only probabilities differ from parent")
        grouped[(row["phase"], row["arm"], row["method"])].append(row)
    for name, assignment in assignments.items():
        phase, subject, condition = name.split("_", 2)
        source = set(assignment["source"])
        test = set(assignment["evaluation"])
        cal = assignment["calibration"]
        if (len(test) != 14 or len(source) == 0 or set(cal) != {"0", "1", "2"}
                or cal["0"] != [] or len(cal["1"]) != 7 or len(cal["2"]) != 14
                or not set(cal["1"]) <= set(cal["2"])):
            raise AssertionError("incomplete nested assignment")
        if source & test or source & set(cal["2"]) or test & set(cal["2"]):
            raise AssertionError("source/calibration/evaluation trial leakage")
        for shot, trials in cal.items():
            labels = []
            for trial in trials:
                match = FILE_RE.fullmatch(trial + ".csv")
                if (match is None or int(match["subject"]) != int(subject) or
                        match["condition"] != condition or
                        int(match["repetition"]) not in ({1} if shot == "1" else {1, 2})):
                    raise AssertionError("calibration trial identity invalid")
                labels.append(int(match["label"]) - 1)
            if shot != "0" and Counter(labels) != Counter({int(c): int(shot) for c in CLASSES}):
                raise AssertionError("calibration class count invalid")
        expected = {(phase, arm, method, trial) for arm in ARMS for method in METHODS
                    for trial in test}
        if not expected <= set(lookup):
            raise AssertionError("missing matched evaluation prediction")
    paired = {}
    for phase in ("validation", "final"):
        for arm in ARMS:
            for method in METHODS:
                subset = grouped[(phase, arm, method)]
                if len(subset) != 280:
                    raise AssertionError("pooled evaluation trial count changed")
                record = result["scores"][phase][arm][method]
                _compare(record["pooled"], _score(subset), f"{phase}/{arm}/{method}/pooled")
                for axis, key in (("by_subject", "subject"), ("by_condition", "condition")):
                    for cell, saved in record[axis].items():
                        part = [r for r in subset if r[key] == cell]
                        _compare(saved, _score(part), f"{phase}/{arm}/{method}/{axis}/{cell}")
                if not np.isclose(record["minimum_subject_macro_f1"],
                                  min(v["macro_f1"] for v in record["by_subject"].values()), atol=1e-12):
                    raise AssertionError("minimum subject score changed")
                if not np.isclose(record["worst_condition_macro_f1"],
                                  min(v["macro_f1"] for v in record["by_condition"].values()), atol=1e-12):
                    raise AssertionError("worst condition score changed")
            source = {r["trial_id"]: r for r in grouped[(phase, arm, "source_only")]}
            for method in ("one_shot", "two_shot"):
                calibrated = {r["trial_id"]: r for r in grouped[(phase, arm, method)]}
                corrected = new = 0
                for trial, base in source.items():
                    later = calibrated[trial]
                    was_correct = np.argmax([float(base[f"p_{c}"]) for c in CLASSES]) == int(base["label"])
                    now_correct = np.argmax([float(later[f"p_{c}"]) for c in CLASSES]) == int(later["label"])
                    corrected += int(not was_correct and now_correct)
                    new += int(was_correct and not now_correct)
                paired[f"{phase}_{arm}_{method}"] = {"corrected": corrected, "new_errors": new}
    audit = {"status": "ok", "prediction_rows": len(rows), "assignments": len(assignments),
             "score_groups_recomputed": len(grouped), "parent_source_only_replay": "exact within 1e-12",
             "paired_vs_source_only": paired,
             "boundary": "matched native trials, public target calibration; no device-live claim"}
    (ROOT / "FORCE_CAL_VERIFICATION.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "prediction_rows": len(rows), "assignments": len(assignments)}))
    return audit


if __name__ == "__main__":
    verify()
