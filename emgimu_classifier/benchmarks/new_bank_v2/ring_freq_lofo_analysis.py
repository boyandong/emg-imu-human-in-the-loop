"""Independent read-back of native-trial four-arm ring/spectral LOFO cells."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss

from benchmarks.new_bank_v2.roam_posture_run import sha256

ROOT = Path(__file__).resolve().parent
FULL = "F0v2+ring_lag+frequency_direction"
ARMS = (FULL, "ring_lag+frequency_direction", "F0v2+frequency_direction",
        "F0v2+ring_lag")
REMOVED = {FULL: "NONE", ARMS[1]: "F0v2", ARMS[2]: "ring_lag",
           ARMS[3]: "frequency_direction"}
FIELDS = ["study", "phase", "group_type", "group", "arm", "removed_family",
          "n_trials", "macro_f1", "accuracy", "log_loss", "brier",
          "per_class_f1_json", "full_minus_removed_macro_f1",
          "removed_minus_full_log_loss", "removed_minus_full_brier",
          "full_wrong_removed_right", "full_right_removed_wrong"]


def score_group(rows: list[dict], classes: np.ndarray) -> dict:
    y = np.asarray([int(r["label"]) for r in rows])
    probability = np.asarray([[float(r[f"p_{c}"]) for c in classes] for r in rows])
    decision = classes[np.argmax(probability, axis=1)]
    one_hot = (y[:, None] == classes[None, :]).astype(float)
    return {"n_trials": len(y),
            "macro_f1": float(f1_score(y, decision, labels=classes, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y, decision)),
            "log_loss": float(log_loss(y, probability, labels=classes)),
            "brier": float(np.mean(np.sum((probability - one_hot) ** 2, axis=1))),
            "per_class_f1_json": json.dumps({str(c): float(value) for c, value in zip(
                classes, f1_score(y, decision, labels=classes,
                                  average=None, zero_division=0))}, sort_keys=True),
            "decision": decision, "truth": y}


def analyze() -> dict:
    result_path = ROOT / "RING_FREQ_LOFO_RESULTS.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    path = ROOT / "RING_FREQ_LOFO_PREDICTIONS.csv"
    parent_path = ROOT / "RING_FREQ_INTERACTION_PREDICTIONS.csv"
    if (result["protocol_sha256"] != sha256(ROOT / "RING_FREQ_LOFO_PROTOCOL.json")
            or result["prediction_sha256"] != sha256(path)
            or result["parent_prediction_sha256"] != sha256(parent_path)
            or result["prediction_rows"] != 3680):
        raise ValueError("LOFO frozen source hashes changed")
    with path.open(newline="", encoding="utf-8") as stream:
        predictions = list(csv.DictReader(stream))
    with parent_path.open(newline="", encoding="utf-8") as stream:
        parent = {(r["study"], r["arm"], r["phase"], r["trial_id"]): r
                  for r in csv.DictReader(stream) if r["arm"] in (FULL, ARMS[2], ARMS[3])}
    copied = {(r["study"], r["arm"], r["phase"], r["trial_id"]): r for r in predictions
              if r["arm"] in (FULL, ARMS[2], ARMS[3])}
    if parent.keys() != copied.keys() or any(parent[k] != copied[k] for k in parent):
        raise AssertionError("three copied LOFO arms differ from frozen parent")
    output = []
    for study in result["scores"]:
        classes = np.asarray([0, 1, 2] if study == "roam_posture" else [4, 15, 16, 17])
        for phase in ("validation", "final"):
            subset = [r for r in predictions if r["study"] == study and r["phase"] == phase]
            by_arm = {arm: {r["trial_id"]: r for r in subset if r["arm"] == arm} for arm in ARMS}
            identities = list(by_arm[FULL])
            if any(set(rows) != set(identities) for rows in by_arm.values()):
                raise AssertionError("LOFO four-arm native trial pairing failed")
            groups = [("pooled", "all", identities)]
            for subject in sorted({int(r["subject"]) for r in by_arm[FULL].values()}):
                groups.append(("subject", str(subject),
                               [key for key in identities if int(by_arm[FULL][key]["subject"]) == subject]))
            if study == "roam_posture":
                for posture in ("resting", "hanging", "unsupported", "reaching"):
                    groups.append(("posture", posture,
                                   [key for key in identities if by_arm[FULL][key]["condition"] == posture]))
            for group_type, group, keys in groups:
                scored = {}
                for arm in ARMS:
                    cell_rows = [by_arm[arm][key] for key in keys]
                    if any(int(a["label"]) != int(b["label"]) for a, b in zip(
                            cell_rows, [by_arm[FULL][key] for key in keys])):
                        raise AssertionError("LOFO labels differ between arms")
                    scored[arm] = score_group(cell_rows, classes)
                    if group_type == "pooled":
                        saved = result["scores"][study][phase][arm]
                        for metric in ("macro_f1", "accuracy", "log_loss", "brier"):
                            if abs(scored[arm][metric] - saved[metric]) > 1e-12:
                                raise AssertionError(f"saved pooled LOFO score mismatch: {study}/{phase}/{arm}")
                full = scored[FULL]
                for arm in ARMS:
                    item = scored[arm]
                    d0, d1, truth = full["decision"], item["decision"], full["truth"]
                    output.append({"study": study, "phase": phase,
                                   "group_type": group_type, "group": group,
                                   "arm": arm, "removed_family": REMOVED[arm],
                                   **{k: item[k] for k in ("n_trials", "macro_f1", "accuracy",
                                                         "log_loss", "brier", "per_class_f1_json")},
                                   "full_minus_removed_macro_f1": full["macro_f1"] - item["macro_f1"],
                                   "removed_minus_full_log_loss": item["log_loss"] - full["log_loss"],
                                   "removed_minus_full_brier": item["brier"] - full["brier"],
                                   "full_wrong_removed_right": int(np.sum((d0 != truth) & (d1 == truth))),
                                   "full_right_removed_wrong": int(np.sum((d0 == truth) & (d1 != truth)))})
    if len(output) != 176:
        raise AssertionError("LOFO group inventory changed")
    table = ROOT / "RING_FREQ_LOFO_CELLS.csv"
    with table.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output)
    audit = {"status": "ok", "prediction_rows": len(predictions),
             "parent_copied_rows": len(copied), "score_cells": len(output),
             "protocol_sha256": sha256(ROOT / "RING_FREQ_LOFO_PROTOCOL.json"),
             "result_sha256": sha256(result_path), "prediction_sha256": sha256(path),
             "parent_prediction_sha256": sha256(parent_path), "cell_sha256": sha256(table),
             "boundary": "Fixed rejected candidate full bank; final results descriptive; repeated-subject and shared-recording correlations remain."}
    (ROOT / "RING_FREQ_LOFO_VERIFICATION.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"ring/frequency LOFO read-back: {len(output)} matched cells", flush=True)
    return audit


if __name__ == "__main__":
    analyze()
