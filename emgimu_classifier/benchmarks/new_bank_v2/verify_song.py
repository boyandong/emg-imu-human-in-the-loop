"""Read back Song v2 trial probabilities and independently check summaries."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmarks.new_bank_v1.song_run import metrics


ROOT = Path(__file__).resolve().parent
OLD_ROWS = ROOT.parent / "new_bank_v1" / "SONG_TRIAL_PREDICTIONS.csv"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def verify() -> dict:
    results = json.loads((ROOT / "SONG_RESULTS.json").read_text(encoding="utf-8"))
    protocol_path = ROOT / "SONG_PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if results["protocol"] != protocol or results["protocol_sha256"] != hashlib.sha256(protocol_path.read_bytes()).hexdigest():
        raise ValueError("protocol hash or content mismatch")
    rows = _rows(ROOT / "SONG_TRIAL_PREDICTIONS.csv")
    classes = np.asarray(protocol["classes"])
    arms = (protocol["reference_arm"], *protocol["arms"])
    expected_count = len(arms) * sum(len(results["sessions"][p]["trial_ids"])
                                     for p in ("validation", "final"))
    if len(rows) != expected_count or len(rows) != results["prediction_rows"]:
        raise ValueError("prediction row count mismatch")
    reference = {(r["phase"], r["trial_id"]): r for r in _rows(OLD_ROWS) if r["arm"] == "F0"}
    replay_count = 0
    max_replay_error = 0.0
    metric_groups = 0
    paired = {}
    for phase, sid in (("validation", "S03"), ("final", "S04")):
        expected_ids = results["sessions"][phase]["trial_ids"]
        grouped = {}
        for arm in arms:
            subset = [r for r in rows if r["phase"] == phase and r["arm"] == arm]
            if ([r["trial_id"] for r in subset] != expected_ids
                    or any(r["session"] != sid for r in subset)):
                raise ValueError(f"trial coverage/order mismatch: {phase}/{arm}")
            truth = np.asarray([r["label"] for r in subset])
            probability = np.asarray([[float(r[f"p_{c}"]) for c in classes] for r in subset])
            if (not np.all(np.isfinite(probability)) or np.any(probability < 0)
                    or not np.allclose(probability.sum(axis=1), 1, atol=1e-12)):
                raise ValueError(f"invalid probabilities: {phase}/{arm}")
            actual = metrics(truth, probability, classes)
            stated = results["sessions"][phase]["arms"][arm]
            for key in ("trials", "accuracy", "macro_f1", "log_loss", "brier"):
                if not np.isclose(actual[key], stated[key], atol=1e-12):
                    raise ValueError(f"metric mismatch: {phase}/{arm}/{key}")
            for label in classes:
                if not np.isclose(actual["recall"][label], stated["recall"][label], atol=1e-12):
                    raise ValueError(f"recall mismatch: {phase}/{arm}/{label}")
            grouped[arm] = (truth, classes[probability.argmax(axis=1)])
            metric_groups += 1
            if arm == protocol["reference_arm"]:
                for row in subset:
                    prior = reference.get((phase, row["trial_id"]))
                    if prior is None or prior["label"] != row["label"]:
                        raise ValueError(f"old F0 trial missing or relabelled: {row['trial_id']}")
                    replay_count += 1
                    max_replay_error = max(max_replay_error, *(abs(float(row[f"p_{c}"]) - float(prior[f"p_{c}"])) for c in classes))
        base_truth, base_pred = grouped["F0v2"]
        if any(not np.array_equal(base_truth, item[0]) for item in grouped.values()):
            raise ValueError("truth differs among arms")
        paired[phase] = {}
        for arm in protocol["arms"][1:]:
            pred = grouped[arm][1]
            paired[phase][arm] = {
                "added_correct_F0v2_wrong": int(np.sum((pred == base_truth) & (base_pred != base_truth))),
                "added_wrong_F0v2_correct": int(np.sum((pred != base_truth) & (base_pred == base_truth))),
                "both_correct": int(np.sum((pred == base_truth) & (base_pred == base_truth))),
                "both_wrong": int(np.sum((pred != base_truth) & (base_pred != base_truth))),
            }
    if replay_count != sum(len(results["sessions"][p]["trial_ids"]) for p in ("validation", "final")):
        raise ValueError("old F0 replay coverage incomplete")
    if max_replay_error > 1e-10:
        raise ValueError(f"old F0 probability replay failed: {max_replay_error}")
    scores = results["sessions"]["validation"]["arms"]
    selected = min(protocol["arms"], key=lambda arm: (
        -scores[arm]["macro_f1"], scores[arm]["log_loss"], protocol["arms"].index(arm)))
    if selected != results["validation_selected_arm"]:
        raise ValueError("validation selection mismatch")
    verification = {"status": "verified", "prediction_rows": len(rows),
                    "heldout_native_trials": replay_count,
                    "old_F0_probability_max_abs_error": max_replay_error,
                    "metric_groups_read_back": metric_groups,
                    "validation_selected_arm": selected,
                    "paired_vs_F0v2": paired}
    (ROOT / "SONG_VERIFICATION.json").write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    return verification


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
