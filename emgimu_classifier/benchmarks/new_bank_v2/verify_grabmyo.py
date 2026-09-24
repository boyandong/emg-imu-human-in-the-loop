"""Read-back validation of new v2 GRABMyo source, scores and paired trials."""
from __future__ import annotations

import csv
import json

import numpy as np

from benchmarks.grabmyo_crossday import run as grab
from benchmarks.new_bank_v1.grabmyo_run import verify_reproduced_baseline
from benchmarks.new_bank_v2.grabmyo_run import ARMS, CLASSES, ROOT


def verify() -> dict:
    results = json.loads((ROOT / "GRABMYO_RESULTS.json").read_text(encoding="utf-8"))
    with (ROOT / "GRABMYO_TRIAL_PREDICTIONS.csv").open(encoding="utf-8", newline="") as stream:
        saved = list(csv.DictReader(stream))
    native = {row["stem"]: row for row in grab.records()}
    expected = {(row["stem"], split, arm) for row in native.values()
                for split, day in (("validation", 2), ("final", 3))
                if row["session"] == day for arm in ARMS}
    seen = {(row["trial_id"], row["split"], row["arm"]) for row in saved}
    if len(saved) != 1792 or len(seen) != len(saved) or seen != expected:
        raise ValueError("frozen arm/day/trial coverage differs")
    rows = []
    for row in saved:
        record = native[row["trial_id"]]
        subject, gesture = int(row["subject"]), int(row["gesture"])
        if subject != record["subject"] or gesture != record["gesture"]:
            raise ValueError("native subject or gesture mismatch")
        p = np.asarray([float(row[f"p_{label}"]) for label in CLASSES])
        if not np.isfinite(p).all() or np.any(p < 0) or abs(p.sum() - 1) > 1e-9:
            raise ValueError("invalid model probabilities")
        rows.append({**row, "subject": subject, "gesture": gesture,
                     **{f"p_{label}": float(value) for label, value in zip(CLASSES, p)}})
    verify_reproduced_baseline(rows)
    lookup = {(row["split"], row["trial_id"], row["arm"]): row for row in rows}
    paired = {}
    for arm in ARMS:
        for split in ("validation", "final"):
            part = [row for row in rows if row["arm"] == arm and row["split"] == split]
            y = np.asarray([row["gesture"] for row in part])
            subjects = np.asarray([row["subject"] for row in part])
            p = np.asarray([[row[f"p_{label}"] for label in CLASSES] for row in part])
            measured = grab.score(y, p, CLASSES, subjects)
            reported = results["arms"][arm][split]
            if reported["trials"] != len(part):
                raise ValueError("trial count differs")
            for field in ("accuracy", "macro_f1", "log_loss", "brier", "minimum_subject_macro_f1"):
                if abs(reported[field] - measured[field]) > 1e-10:
                    raise ValueError(f"reported {arm} {split} {field} differs")
            for field in ("per_class_recall", "per_subject_macro_f1"):
                if reported[field].keys() != measured[field].keys() or any(
                    abs(reported[field][key] - measured[field][key]) > 1e-10 for key in measured[field]):
                    raise ValueError(f"reported {arm} {split} {field} differs")
        if arm == "F0":
            continue
        counts = {"added_correct_F0_wrong": 0, "added_wrong_F0_correct": 0,
                  "both_correct": 0, "both_wrong": 0}
        for record in native.values():
            if record["session"] != 3:
                continue
            base = lookup[("final", record["stem"], "F0")]
            added = lookup[("final", record["stem"], arm)]
            base_ok = CLASSES[np.argmax([base[f"p_{label}"] for label in CLASSES])] == record["gesture"]
            added_ok = CLASSES[np.argmax([added[f"p_{label}"] for label in CLASSES])] == record["gesture"]
            key = ("added_correct_F0_wrong" if added_ok and not base_ok else
                   "added_wrong_F0_correct" if base_ok and not added_ok else
                   "both_correct" if base_ok else "both_wrong")
            counts[key] += 1
        if sum(counts.values()) != 224:
            raise ValueError("final paired coverage differs")
        paired[arm] = counts
    chosen = min(ARMS, key=lambda arm: (-results["arms"][arm]["validation"]["macro_f1"],
                                         results["arms"][arm]["validation"]["log_loss"]))
    if chosen != results["validation_selected_arm"]:
        raise ValueError("selection rule differs")
    artifact = {"status": "verified", "prediction_rows": len(rows),
                "heldout_native_trials": len(seen) // len(ARMS),
                "baseline_probability_replay": "448 old F0 rows matched within 1e-10",
                "metric_groups_read_back": 8, "validation_selected_arm": chosen,
                "final_paired_vs_F0": paired}
    (ROOT / "GRABMYO_VERIFICATION.json").write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return artifact


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
