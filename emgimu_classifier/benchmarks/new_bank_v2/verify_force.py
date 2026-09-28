"""Independent native-trial score and baseline replay audit for new v2 force study."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, recall_score

from emgimu.datasets.libemg_force import FILE_RE

ROOT = Path(__file__).resolve().parent
CLASSES = np.arange(7)


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _score(rows: list[dict]) -> dict:
    y = np.asarray([int(r["label"]) for r in rows])
    p = np.asarray([[float(r[f"p_{c}"]) for c in CLASSES] for r in rows])
    if not len(y) or not np.isfinite(p).all() or np.any(p < 0):
        raise AssertionError("invalid probabilities")
    np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-12)
    pred = p.argmax(axis=1)
    return {"trials": len(y), "accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0)),
            "log_loss": float(log_loss(y, p, labels=CLASSES)),
            "brier": float(np.mean(np.sum((p - np.eye(7)[y]) ** 2, axis=1))),
            "per_class_recall": {str(c): float(value) for c, value in zip(
                CLASSES, recall_score(y, pred, labels=CLASSES, average=None, zero_division=0))}}


def _assert_score(saved: dict, actual: dict) -> None:
    if saved.keys() != actual.keys():
        raise AssertionError("score fields differ")
    for name, value in actual.items():
        if isinstance(value, dict):
            _assert_score(saved[name], value)
        else:
            np.testing.assert_allclose(saved[name], value, rtol=0, atol=1e-12)


def verify() -> dict:
    result = json.loads((ROOT / "FORCE_RESULTS.json").read_text(encoding="utf-8"))
    protocol_path = ROOT / "FORCE_PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (result["protocol"] != protocol or
            hashlib.sha256(protocol_path.read_bytes()).hexdigest() != result["protocol_sha256"] or
            result["source_rest_windows"] != 192):
        raise AssertionError("frozen protocol/source rest count mismatch")
    previous_result = json.loads((ROOT.parent / "new_bank_v1" / "RESULTS.json").read_text(encoding="utf-8"))
    if result["archive_sha256"] != previous_result["archive_sha256"]["force"]:
        raise AssertionError("source archive changed")
    rows = _read(ROOT / "FORCE_TRIAL_PREDICTIONS.csv")
    arms = protocol["arms"]
    if len(rows) != 5880:
        raise AssertionError("five arms must cover 1,176 held-out native trials")
    index = {}
    for row in rows:
        key = (row["phase"], row["arm"], row["trial_id"])
        if key in index:
            raise AssertionError("duplicate arm/trial prediction")
        index[key] = row
        native = FILE_RE.fullmatch(row["trial_id"] + ".csv")
        if (native is None or int(native["subject"]) != int(row["subject"])
                or native["condition"].lower() != row["condition"].lower()
                or int(native["label"]) - 1 != int(row["label"])):
            raise AssertionError("native trial metadata mismatch")
    for phase in ("validation", "final"):
        split = result["split_trial_ids"][phase]
        source, target = set(split["source_trials"]), set(split["target_trials"])
        if (len(source) != 168 or len(target) != 588 or source & target
                or set(split["source_subjects"]) != set(protocol["source_subjects"])
                or set(split["target_subjects"]) != set(protocol[f"{phase}_subjects"])):
            raise AssertionError("native subject/trial split mismatch")
        for arm in arms:
            part = [row for row in rows if row["phase"] == phase and row["arm"] == arm]
            if {r["trial_id"] for r in part} != target:
                raise AssertionError("trial coverage changed across arms")
            saved = result["scores"][phase][arm]
            _assert_score(saved["pooled"], _score(part))
            for subject, score in saved["by_subject"].items():
                _assert_score(score, _score([r for r in part if r["subject"] == subject]))
            for condition, score in saved["by_condition"].items():
                _assert_score(score, _score([r for r in part if r["condition"] == condition]))
            np.testing.assert_allclose(saved["minimum_subject_macro_f1"],
                                       min(v["macro_f1"] for v in saved["by_subject"].values()), atol=1e-12)
            np.testing.assert_allclose(saved["worst_condition_macro_f1"],
                                       min(v["macro_f1"] for v in saved["by_condition"].values()), atol=1e-12)
    selected = min(arms, key=lambda arm: (
        -result["scores"]["validation"][arm]["pooled"]["macro_f1"],
        result["scores"]["validation"][arm]["pooled"]["log_loss"]))
    if selected != result["validation_selected_arm"]:
        raise AssertionError("validation arm selection changed")
    previous = {(r["phase"], r["trial_id"]): r for r in
                _read(ROOT.parent / "new_bank_v1" / "TRIAL_PREDICTIONS.csv")
                if r["dataset"] == "force" and r["arm"] == "F0"}
    baseline = {(key[0], key[2]): row for key, row in index.items() if key[1] == "F0"}
    if previous.keys() != baseline.keys():
        raise AssertionError("previous F0 target trials differ")
    baseline_error = max(abs(float(row[f"p_{c}"]) - float(baseline[key][f"p_{c}"]))
                         for key, row in previous.items() for c in CLASSES)
    if baseline_error > 1e-6:
        raise AssertionError("previous F0 probabilities differ beyond numerical tolerance")
    for phase in ("validation", "final"):
        earlier = previous_result["results"]["force"][phase]["arms"]["F0"]["pooled"]
        current = result["scores"][phase]["F0"]["pooled"]
        np.testing.assert_allclose(current["macro_f1"], earlier["macro_f1"], rtol=0, atol=0)
        for metric in ("log_loss", "brier"):
            np.testing.assert_allclose(current[metric], earlier[metric], rtol=0, atol=1e-6)
    paired = {}
    for phase in ("validation", "final"):
        base = {key[2]: row for key, row in index.items() if key[:2] == (phase, "F0")}
        chosen = {key[2]: row for key, row in index.items() if key[:2] == (phase, selected)}
        corrected = harmed = 0
        for trial, left in base.items():
            right = chosen[trial]
            y = int(left["label"])
            before = int(np.argmax([float(left[f"p_{c}"]) for c in CLASSES]))
            after = int(np.argmax([float(right[f"p_{c}"]) for c in CLASSES]))
            corrected += before != y and after == y
            harmed += before == y and after != y
        paired[phase] = {"corrected_F0_errors": int(corrected),
                         "new_errors_vs_F0": int(harmed)}
    audit = {"status": "ok", "prediction_rows": len(rows),
             "score_groups_recomputed": 10,
             "baseline_max_probability_error": baseline_error,
             "baseline_probability_tolerance": 1e-6,
             "validation_selected_arm": selected, "paired_vs_F0": paired}
    (ROOT / "FORCE_VERIFICATION.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit))
    return audit


if __name__ == "__main__":
    verify()
