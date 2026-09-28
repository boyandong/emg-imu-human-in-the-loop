"""Independent read-back verifier for the frozen reduced-bank MANUS study."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, recall_score

from emgimu.datasets.semg_manus import PATH_RE

ROOT = Path(__file__).resolve().parent


def score(rows: list[dict], classes: np.ndarray) -> dict:
    y = np.asarray([int(row["label"]) for row in rows])
    p = np.asarray([[float(row[f"p_{c}"]) for c in classes] for row in rows])
    if not len(y) or not np.isfinite(p).all() or np.any(p < 0):
        raise AssertionError("empty group or invalid probabilities")
    np.testing.assert_allclose(p.sum(axis=1), 1, rtol=0, atol=1e-12)
    predicted = classes[np.argmax(p, axis=1)]
    return {"trials": len(y), "accuracy": float(accuracy_score(y, predicted)),
            "macro_f1": float(f1_score(y, predicted, labels=classes,
                                       average="macro", zero_division=0)),
            "log_loss": float(log_loss(y, p, labels=classes)),
            "brier": float(np.mean(np.sum((p - np.eye(len(classes))[y]) ** 2, axis=1))),
            "per_class_recall": {str(c): float(value) for c, value in zip(
                classes, recall_score(y, predicted, labels=classes,
                                           average=None, zero_division=0))}}


def same(saved: dict, actual: dict) -> None:
    if saved.keys() != actual.keys():
        raise AssertionError("saved score keys changed")
    for key, value in actual.items():
        if isinstance(value, dict):
            same(saved[key], value)
        else:
            np.testing.assert_allclose(saved[key], value, rtol=0, atol=1e-12)


def verify() -> dict:
    protocol_path = ROOT / "MANUS_SPATIAL_PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    results = json.loads((ROOT / "MANUS_SPATIAL_RESULTS.json").read_text(encoding="utf-8"))
    prediction_path = ROOT / "MANUS_SPATIAL_PREDICTIONS.csv"
    if results["protocol"] != protocol or results["protocol_sha256"] != hashlib.sha256(protocol_path.read_bytes()).hexdigest():
        raise AssertionError("protocol changed")
    if results["archive_sha256"].lower() != protocol["archive_sha256"].lower():
        raise AssertionError("source archive digest changed")
    if results["prediction_sha256"] != hashlib.sha256(prediction_path.read_bytes()).hexdigest():
        raise AssertionError("prediction table changed")
    with prediction_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    arms = protocol["arms"]
    users, speeds = set(protocol["users"]), set(protocol["conditions"])
    classes = np.arange(len(protocol["gestures"]))
    if len(rows) != 2 * 108 * len(arms):
        raise AssertionError("expected 864 arm–trial predictions")
    index = {}
    per_cell = Counter()
    for row in rows:
        phase, arm, trial = row["phase"], row["arm"], row["trial_id"]
        if phase not in ("validation", "final") or arm not in arms:
            raise AssertionError("unexpected phase or arm")
        native = PATH_RE.fullmatch(trial)
        session = protocol[f"{phase}_session"]
        label, user, speed = int(row["label"]), int(row["subject"]), row["condition"]
        if (native is None or int(native["session"]) != session
                or int(native["user"]) != user or native["speed"] != speed
                or native["gesture"] != protocol["gestures"][label]
                or user not in users or speed not in speeds):
            raise AssertionError("native trial identity mismatch")
        key = (phase, arm, trial)
        if key in index:
            raise AssertionError("duplicate arm–trial identity")
        index[key] = row
        per_cell[phase, arm, user, speed] += 1
    if set(per_cell.values()) != {6} or len(per_cell) != 2 * len(arms) * 6 * 3:
        raise AssertionError("incomplete user–speed cells")
    checked_groups = 0
    for phase in ("validation", "final"):
        split = results["split_trial_ids"][phase]
        source_trials, target_trials = set(split["source_trials"]), set(split["target_trials"])
        if (len(source_trials) != 108 or len(target_trials) != 108
                or source_trials & target_trials or split["source_users"] != protocol["users"]
                or split["target_users"] != protocol["users"]):
            raise AssertionError("source/target split contract failed")
        if target_trials != {trial for ph, arm, trial in index if ph == phase and arm == arms[0]}:
            raise AssertionError("saved target IDs differ from predictions")
        for arm in arms:
            arm_rows = [row for row in rows if row["phase"] == phase and row["arm"] == arm]
            if {row["trial_id"] for row in arm_rows} != target_trials:
                raise AssertionError("candidate arms are not native-trial matched")
            reference = results["scores"][phase][arm]
            same(reference["pooled"], score(arm_rows, classes))
            checked_groups += 1
            for user in protocol["users"]:
                same(reference["by_user"][str(user)], score(
                    [row for row in arm_rows if int(row["subject"]) == user], classes))
                checked_groups += 1
            for speed in protocol["conditions"]:
                same(reference["by_speed"][speed], score(
                    [row for row in arm_rows if row["condition"] == speed], classes))
                checked_groups += 1
            np.testing.assert_allclose(reference["minimum_user_macro_f1"],
                                       min(item["macro_f1"] for item in reference["by_user"].values()), rtol=0, atol=1e-12)
            np.testing.assert_allclose(reference["minimum_speed_macro_f1"],
                                       min(item["macro_f1"] for item in reference["by_speed"].values()), rtol=0, atol=1e-12)
    selected = min(arms, key=lambda arm: (
        -results["scores"]["validation"][arm]["pooled"]["macro_f1"],
        results["scores"]["validation"][arm]["pooled"]["log_loss"]))
    if selected != results["validation_selected_arm"]:
        raise AssertionError("validation selection changed")
    audit = {"status": "ok", "prediction_rows": len(rows),
             "native_target_trials": 216, "checked_score_groups": checked_groups,
             "validation_selected_arm": selected,
             "results_sha256": hashlib.sha256((ROOT / "MANUS_SPATIAL_RESULTS.json").read_bytes()).hexdigest(),
             "prediction_sha256": results["prediction_sha256"],
             "boundary": protocol["scope"]}
    (ROOT / "MANUS_SPATIAL_VERIFICATION.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "rows": len(rows), "score_groups": checked_groups}))
    return audit


if __name__ == "__main__":
    verify()
