"""Independently rescore saved native-trial probabilities and source boundaries."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, recall_score

from emgimu.datasets.electrode_shift import PATH_RE

ROOT = Path(__file__).resolve().parent
CLASSES = np.arange(5)


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _score(rows: list[dict]) -> dict:
    y = np.asarray([int(r["label"]) for r in rows])
    p = np.asarray([[float(r[f"p_{c}"]) for c in CLASSES] for r in rows])
    if not len(y) or not np.isfinite(p).all() or np.any(p < 0):
        raise AssertionError("invalid saved probabilities")
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-12)
    pred = p.argmax(axis=1)
    return {"trials": len(y), "accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0)),
            "log_loss": float(log_loss(y, p, labels=CLASSES)),
            "brier": float(np.mean(np.sum((p - np.eye(5)[y]) ** 2, axis=1))),
            "per_class_recall": {str(c): float(v) for c, v in zip(
                CLASSES, recall_score(y, pred, labels=CLASSES, average=None, zero_division=0))}}


def _assert_score(saved: dict, actual: dict) -> None:
    if saved.keys() != actual.keys():
        raise AssertionError("score schema mismatch")
    for key, value in actual.items():
        if isinstance(value, dict):
            _assert_score(saved[key], value)
        elif isinstance(value, (int, float)):
            np.testing.assert_allclose(saved[key], value, rtol=0, atol=1e-12)


def verify() -> dict:
    result = json.loads((ROOT / "WEARING_RESULTS.json").read_text(encoding="utf-8"))
    protocol_path = ROOT / "WEARING_PROTOCOL.json"
    if hashlib.sha256(protocol_path.read_bytes()).hexdigest() != result["protocol_sha256"]:
        raise AssertionError("protocol changed after fitting")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if result["protocol"] != protocol:
        raise AssertionError("embedded protocol differs")
    rows = _read(ROOT / "WEARING_TRIAL_PREDICTIONS.csv")
    arms = protocol["arms"]
    if len(rows) != 1200:
        raise AssertionError("expected five arms on 240 native target trials")
    index = {}
    for row in rows:
        key = (row["phase"], int(row["subject"]), row["arm"], row["trial_id"])
        if key in index:
            raise AssertionError("duplicate arm/trial prediction")
        index[key] = row
        match = PATH_RE.fullmatch(row["trial_id"])
        if (match is None or int(match["subject"]) != key[1]
                or match["domain"] != row["domain"]
                or int(match["label"]) != int(row["label"])):
            raise AssertionError("native trial metadata mismatch")
    for phase in ("validation", "final"):
        for subject in protocol[f"{phase}_subjects"]:
            split = result["split_trial_ids"][f"{phase}_{subject}"]
            source, target = set(split["source"]), set(split["target"])
            if len(source) != 25 or len(target) != 40 or source & target:
                raise AssertionError("native source/target split invalid")
            for arm in arms:
                found = {key[3] for key in index if key[:3] == (phase, subject, arm)}
                if found != target:
                    raise AssertionError("target trial coverage differs by arm")
    for phase in ("validation", "final"):
        for arm in arms:
            part = [row for row in rows if row["phase"] == phase and row["arm"] == arm]
            saved = result["scores"][phase][arm]
            _assert_score(saved["pooled"], _score(part))
            for subject, score in saved["by_subject"].items():
                _assert_score(score, _score([r for r in part if r["subject"] == subject]))
            for domain, score in saved["by_domain"].items():
                _assert_score(score, _score([r for r in part if r["domain"] == domain]))
            np.testing.assert_allclose(saved["minimum_subject_macro_f1"],
                                       min(x["macro_f1"] for x in saved["by_subject"].values()), atol=1e-12)
            np.testing.assert_allclose(saved["worst_domain_macro_f1"],
                                       min(x["macro_f1"] for x in saved["by_domain"].values()), atol=1e-12)
    selected = min(arms, key=lambda arm: (
        -result["scores"]["validation"][arm]["pooled"]["macro_f1"],
        result["scores"]["validation"][arm]["pooled"]["log_loss"]))
    if selected != result["validation_selected_arm"]:
        raise AssertionError("validation selection changed")
    previous = {(
        r["phase"], int(r["subject"]), r["trial_id"]): r for r in
        _read(ROOT.parent / "new_bank_v1" / "TRIAL_PREDICTIONS.csv")
        if r["dataset"] == "wearing" and r["arm"] == "F0"}
    baseline = {key[:2] + key[3:]: row for key, row in index.items() if key[2] == "F0"}
    if previous.keys() != baseline.keys():
        raise AssertionError("earlier F0 trial identities differ")
    baseline_error = max(abs(float(baseline[key][f"p_{c}"]) - float(row[f"p_{c}"]))
                         for key, row in previous.items() for c in CLASSES)
    if baseline_error > 1e-12:
        raise AssertionError("earlier F0 baseline not exactly replayed")
    corrections = {}
    for phase in ("validation", "final"):
        base = {(key[1], key[3]): row for key, row in index.items()
                if key[0] == phase and key[2] == "F0"}
        chosen = {(key[1], key[3]): row for key, row in index.items()
                  if key[0] == phase and key[2] == selected}
        corrected = new_errors = 0
        for key, left in base.items():
            right = chosen[key]
            truth = int(left["label"])
            left_pred = int(np.argmax([float(left[f"p_{c}"]) for c in CLASSES]))
            right_pred = int(np.argmax([float(right[f"p_{c}"]) for c in CLASSES]))
            corrected += left_pred != truth and right_pred == truth
            new_errors += left_pred == truth and right_pred != truth
        corrections[phase] = {"corrected_F0_errors": int(corrected),
                              "new_errors_vs_F0": int(new_errors)}
    output = {"status": "ok", "prediction_rows": len(rows),
              "score_groups_recomputed": 10, "baseline_max_probability_error": baseline_error,
              "validation_selected_arm": selected, "paired_vs_F0": corrections}
    (ROOT / "WEARING_VERIFICATION.json").write_text(json.dumps(output, indent=2) + "\n",
                                                    encoding="utf-8")
    print(json.dumps(output))
    return output


if __name__ == "__main__":
    verify()
