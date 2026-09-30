"""Retrospectively apply the existing default-bank guard to frozen F2 screens."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "F2_CROSS_AXIS_REVIEW_PROTOCOL.json"
CELLS = ROOT / "F2_CROSS_AXIS_REVIEW_CELLS.csv"
AUDIT = ROOT / "F2_CROSS_AXIS_REVIEW_AUDIT.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def review() -> dict:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    sources = {}
    for name, expected in protocol["source_results_sha256"].items():
        path = ROOT / name
        if digest(path) != expected:
            raise AssertionError(f"Frozen F2 source changed: {name}")
        sources[name] = json.loads(path.read_text(encoding="utf-8"))
    wear_ac, wear_b, manus_new, manus_old, grab_new, grab_old = (
        sources[name] for name in protocol["source_results_sha256"])
    if wear_ac["split_trial_ids"] != wear_b["split_trial_ids"]:
        raise AssertionError("Wearing trial partitions differ")
    if grab_new["source_trial_ids"] != grab_old["source_trial_ids"]:
        raise AssertionError("GRAB source trials differ")
    for phase in ("validation", "final"):
        if (manus_new["split_trial_ids"][phase]["source_trials"] !=
                manus_old["split_trial_ids"][phase]["source_trials"] or
                manus_new["split_trial_ids"][phase]["target_trials"] !=
                manus_old["split_trial_ids"][phase]["target_trials"] or
                grab_new[f"{phase}_trial_ids"] != grab_old[f"{phase}_trial_ids"]):
            raise AssertionError(f"Matched {phase} target trials differ")
    rows = []
    for phase in ("validation", "final"):
        axes = {
            "wearing_shift": {
                "F0v2": wear_b["scores"][phase]["F0v2"],
                "F0v2+F2a": wear_ac["scores"][phase]["F0v2+F2a_trace"],
                "F0v2+F2b": wear_b["scores"][phase]["F0v2+F2b_document"],
                "F0v2+F2c": wear_ac["scores"][phase]["F0v2+F2c_spd"],
            },
            "manus_session": {
                "F0v2": manus_new["scores"][phase]["F0v2"],
                "F0v2+F2a": manus_old["scores"][phase]["F0v2+F2a"],
                "F0v2+F2b": manus_new["scores"][phase]["F0v2+F2b_document"],
                "F0v2+F2c": manus_new["scores"][phase]["F0v2+F2c_spd"],
            },
            "grab_unseen_user": {
                "F0v2": grab_new["scores"]["F0v2"][phase],
                "F0v2+F2a": grab_old["scores"]["F0v2+F2a"][phase],
                "F0v2+F2b": grab_new["scores"]["F0v2+F2b_document"][phase],
                "F0v2+F2c": grab_new["scores"]["F0v2+F2c_spd"][phase],
            },
        }
        for axis in protocol["axes"]:
            baseline = axes[axis]["F0v2"]["pooled"] if axis != "grab_unseen_user" else axes[axis]["F0v2"]
            for arm in protocol["arms"]:
                measures = axes[axis][arm]["pooled"] if axis != "grab_unseen_user" else axes[axis][arm]
                if measures["trials"] != baseline["trials"]:
                    raise AssertionError(f"Unmatched {axis} trial count")
                rows.append({"phase": phase, "axis": axis, "arm": arm,
                             "trials": measures["trials"],
                             "macro_f1": measures["macro_f1"],
                             "log_loss": measures["log_loss"],
                             "delta_macro_f1": measures["macro_f1"]-baseline["macro_f1"],
                             "delta_log_loss": measures["log_loss"]-baseline["log_loss"]})
    with CELLS.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    eligible = []
    violations = {}
    for arm in protocol["arms"][1:]:
        selected = [row for row in rows if row["phase"] == "validation" and row["arm"] == arm]
        bad = [f"{row['axis']}:{metric}" for row in selected
               for metric, failed in (("macro_f1", row["delta_macro_f1"] < -1e-12),
                                      ("log_loss", row["delta_log_loss"] > 1e-12)) if failed]
        violations[arm] = bad
        if not bad and any(row["delta_macro_f1"] > 1e-12 or row["delta_log_loss"] < -1e-12
                           for row in selected):
            eligible.append(arm)
    if len(eligible) > 1:
        raise AssertionError("Guard does not define a tie-break among multiple eligible additions")
    result = {"status": "retrospective_frozen_f2_review",
              "protocol_sha256": digest(PROTOCOL),
              "source_results_sha256": protocol["source_results_sha256"],
              "cells_sha256": digest(CELLS), "cells": len(rows),
              "validation_violations": violations,
              "eligible_additions": eligible,
              "three_axis_default": eligible[0] if eligible else "F0v2",
              "scope": protocol["boundary"]}
    AUDIT.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"cells": len(rows), "three_axis_default": result["three_axis_default"]}))
    return result


if __name__ == "__main__":
    review()
