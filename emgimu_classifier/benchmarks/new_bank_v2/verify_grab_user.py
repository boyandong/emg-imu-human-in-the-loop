"""Independent native-record and score audit of GRABMyo same-day cross-user study."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, recall_score

from benchmarks.grabmyo_crossday import run as grab

ROOT = Path(__file__).resolve().parent


def metric(rows: list[dict], classes: np.ndarray) -> dict:
    y = np.asarray([int(row["gesture"]) for row in rows])
    subjects = np.asarray([int(row["subject"]) for row in rows])
    p = np.asarray([[float(row[f"p_{c}"]) for c in classes] for row in rows])
    if not len(y) or not np.isfinite(p).all() or np.any(p < 0):
        raise AssertionError("empty split or invalid probabilities")
    np.testing.assert_allclose(p.sum(axis=1), 1, rtol=0, atol=1e-12)
    pred = classes[p.argmax(axis=1)]
    user_f1 = {str(subject): float(f1_score(y[subjects == subject], pred[subjects == subject],
                                            labels=classes, average="macro", zero_division=0))
               for subject in sorted(set(subjects))}
    return {"accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, labels=classes, average="macro", zero_division=0)),
            "log_loss": float(log_loss(y, p, labels=classes)),
            "brier": float(np.mean(np.sum((p - (y[:, None] == classes[None, :])) ** 2, axis=1))),
            "per_class_recall": {str(c): float(value) for c, value in zip(
                classes, recall_score(y, pred, labels=classes, average=None, zero_division=0))},
            "per_subject_macro_f1": user_f1,
            "minimum_subject_macro_f1": min(user_f1.values())}


def same(saved: dict, actual: dict) -> None:
    if saved.keys() != actual.keys():
        raise AssertionError("saved and rescored metric fields differ")
    for key, value in actual.items():
        if isinstance(value, dict):
            same(saved[key], value)
        else:
            np.testing.assert_allclose(saved[key], value, rtol=0, atol=1e-12)


def verify() -> dict:
    protocol_path = ROOT / "GRAB_USER_PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    result_path = ROOT / "GRAB_USER_RESULTS.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    prediction_path = ROOT / "GRAB_USER_PREDICTIONS.csv"
    if result["protocol"] != protocol or result["protocol_sha256"] != hashlib.sha256(protocol_path.read_bytes()).hexdigest():
        raise AssertionError("frozen protocol changed")
    data_root = Path("D:/emg-imu-benchmarks/data/raw/grabmyo_crossday_subset_v1")
    if result["official_sha256_manifest_sha256"] != hashlib.sha256((data_root / "SHA256SUMS.txt").read_bytes()).hexdigest():
        raise AssertionError("official checksum manifest changed")
    if result["prediction_sha256"] != hashlib.sha256(prediction_path.read_bytes()).hexdigest():
        raise AssertionError("prediction table changed")
    if result["verified_day1_files"] != 448 or result["source_rest_windows"] != 560:
        raise AssertionError("source count changed")
    native = {row["stem"]: row for row in grab.records() if row["session"] == 1}
    if len(native) != 224:
        raise AssertionError("native Day1 manifest changed")
    expected = {phase: {name for name, row in native.items()
                        if row["subject"] in protocol[f"{phase}_subjects"]}
                for phase in ("validation", "final")}
    source = {name for name, row in native.items() if row["subject"] in protocol["source_subjects"]}
    if (source != set(result["source_trial_ids"]) or len(source) != 112
            or any(len(ids) != 56 or ids & source for ids in expected.values())
            or expected["validation"] & expected["final"]):
        raise AssertionError("subject-disjoint split changed")
    for phase in ("validation", "final"):
        if expected[phase] != set(result[f"{phase}_trial_ids"]):
            raise AssertionError("saved target trial IDs changed")
    with prediction_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    arms = protocol["arms"]
    classes = np.asarray(protocol["gesture_codes"])
    if len(rows) != 2 * 56 * len(arms):
        raise AssertionError("expected 448 arm–record probabilities")
    index = {}
    for row in rows:
        phase, arm, trial = row["phase"], row["arm"], row["trial_id"]
        if phase not in expected or arm not in arms or trial not in expected[phase]:
            raise AssertionError("wrong phase, arm or target ID")
        if int(row["subject"]) != native[trial]["subject"] or int(row["gesture"]) != native[trial]["gesture"]:
            raise AssertionError("native label or subject mismatch")
        key = (phase, arm, trial)
        if key in index:
            raise AssertionError("duplicate arm–record row")
        index[key] = row
    checked = 0
    for arm in arms:
        if result["scores"][arm]["train_trials"] != 112:
            raise AssertionError("source trial count changed")
        for phase in ("validation", "final"):
            subset = [row for row in rows if row["arm"] == arm and row["phase"] == phase]
            if {row["trial_id"] for row in subset} != expected[phase]:
                raise AssertionError("candidate arms are not native-record matched")
            saved = result["scores"][arm][phase]
            if saved["trials"] != len(subset):
                raise AssertionError("saved trial count changed")
            same({key: value for key, value in saved.items() if key != "trials"}, metric(subset, classes))
            checked += 1
    selected = min(arms, key=lambda arm: (
        -result["scores"][arm]["validation"]["macro_f1"],
        result["scores"][arm]["validation"]["log_loss"]))
    if selected != result["validation_selected_arm"]:
        raise AssertionError("validation selection changed")
    paired = {}
    for phase in ("validation", "final"):
        for arm in arms[1:]:
            correct0, correct1 = [], []
            for trial in sorted(expected[phase]):
                truth = native[trial]["gesture"]
                for target, name in ((correct0, arms[0]), (correct1, arm)):
                    row = index[phase, name, trial]
                    probabilities = [float(row[f"p_{c}"]) for c in classes]
                    target.append(int(classes[np.argmax(probabilities)]) == truth)
            paired[f"{phase}:{arm}"] = {
                "base_correct_added_wrong": int(np.sum(np.asarray(correct0) & ~np.asarray(correct1))),
                "base_wrong_added_correct": int(np.sum(~np.asarray(correct0) & np.asarray(correct1))),
            }
    audit = {"status": "ok", "prediction_rows": len(rows), "native_target_records": 112,
             "score_groups_checked": checked, "validation_selected_arm": selected,
             "paired_correctness": paired,
             "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
             "prediction_sha256": result["prediction_sha256"],
             "boundary": protocol["scope"]}
    (ROOT / "GRAB_USER_VERIFICATION.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "rows": len(rows), "groups": checked}))
    return audit


if __name__ == "__main__":
    verify()
