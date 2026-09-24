"""Read-back audit for the frozen new-bank GRABMyo cross-day predictions."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from benchmarks.grabmyo_crossday import run as grab
from benchmarks.new_bank_v1.grabmyo_run import ARMS, CLASSES, ROOT, verify_reproduced_baseline


def verify() -> dict:
    results = json.loads((ROOT / "GRABMYO_RESULTS.json").read_text(encoding="utf-8"))
    with (ROOT / "GRABMYO_TRIAL_PREDICTIONS.csv").open(encoding="utf-8", newline="") as stream:
        saved = list(csv.DictReader(stream))
    expected = {(record["stem"], split, arm)
                for record in grab.records()
                for split in ("validation", "final")
                if record["session"] == grab.PROTOCOL["sessions"][split]
                for arm in ARMS}
    observed = {(row["trial_id"], row["split"], row["arm"]) for row in saved}
    if len(saved) != 1792 or len(observed) != len(saved) or observed != expected:
        raise ValueError("prediction trial/arm coverage differs from frozen protocol")
    native = {record["stem"]: record for record in grab.records()}
    rows = []
    for row in saved:
        record = native[row["trial_id"]]
        subject, label = int(row["subject"]), int(row["gesture"])
        if subject != record["subject"] or label != record["gesture"]:
            raise ValueError("saved native label or subject differs")
        probability = np.array([float(row[f"p_{category}"]) for category in CLASSES])
        if not np.all(np.isfinite(probability)) or np.any(probability < 0) or abs(probability.sum() - 1) > 1e-9:
            raise ValueError("invalid prediction probability")
        rows.append({**row, "subject": subject, "gesture": label,
                     **{f"p_{category}": float(value) for category, value in zip(CLASSES, probability)}})
    verify_reproduced_baseline(rows)
    for arm in ARMS:
        for split in ("validation", "final"):
            part = [row for row in rows if row["arm"] == arm and row["split"] == split]
            y = np.asarray([row["gesture"] for row in part])
            subjects = np.asarray([row["subject"] for row in part])
            probability = np.asarray([[row[f"p_{category}"] for category in CLASSES] for row in part])
            recomputed = grab.score(y, probability, CLASSES, subjects)
            reported = results["arms"][arm][split]
            if reported["trials"] != len(part):
                raise ValueError("reported trial count differs")
            for name in ("accuracy", "macro_f1", "log_loss", "brier", "minimum_subject_macro_f1"):
                if abs(recomputed[name] - reported[name]) > 1e-10:
                    raise ValueError(f"reported {arm} {split} {name} differs")
            for name in ("per_class_recall", "per_subject_macro_f1"):
                if recomputed[name].keys() != reported[name].keys() or any(
                    abs(recomputed[name][key] - reported[name][key]) > 1e-10
                    for key in recomputed[name]):
                    raise ValueError(f"reported {arm} {split} {name} differs")
    selected = min(ARMS, key=lambda name:
        (-results["arms"][name]["validation"]["macro_f1"],
         results["arms"][name]["validation"]["log_loss"]))
    if selected != results["validation_selected_arm"]:
        raise ValueError("selection rule differs from saved validation result")
    lookup = {(row["split"], row["trial_id"], row["arm"]): row for row in rows}
    paired = {}
    for split in ("validation", "final"):
        changed = {"selected_correct_baseline_wrong": 0, "selected_wrong_baseline_correct": 0,
                   "both_correct": 0, "both_wrong": 0}
        for record in grab.records():
            if record["session"] != grab.PROTOCOL["sessions"][split]:
                continue
            base = lookup[(split, record["stem"], "F0")]
            added = lookup[(split, record["stem"], selected)]
            base_correct = CLASSES[np.argmax([base[f"p_{category}"] for category in CLASSES])] == record["gesture"]
            added_correct = CLASSES[np.argmax([added[f"p_{category}"] for category in CLASSES])] == record["gesture"]
            key = ("selected_correct_baseline_wrong" if added_correct and not base_correct else
                   "selected_wrong_baseline_correct" if base_correct and not added_correct else
                   "both_correct" if base_correct else "both_wrong")
            changed[key] += 1
        if sum(changed.values()) != 224:
            raise ValueError("paired trial coverage differs")
        paired[split] = changed
    artifact = {"status": "verified", "prediction_rows": len(rows),
                "distinct_trial_ids": len({row["trial_id"] for row in rows}),
                "prior_grabmyo_F0_probability_replay": "exact within 1e-10 for all 448 held-out trials",
                "metric_groups_read_back": len(ARMS) * 2,
                "validation_selected_arm": selected,
                "paired_vs_F0": paired}
    (ROOT / "GRABMYO_VERIFICATION.json").write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return artifact


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
