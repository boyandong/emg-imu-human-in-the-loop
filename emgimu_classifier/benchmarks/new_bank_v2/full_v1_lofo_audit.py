"""Read back seven frozen full-bank experiments and report matched LOFO effects."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from benchmarks.grabmyo_crossday.run import score
from benchmarks.new_bank_v2.full_v1_lofo_run import ARMS, PROTOCOL, PROTOCOL_PATH
from benchmarks.new_bank_v2.roam_posture_run import sha256
from benchmarks.new_bank_v2.roam_v1_quality_run import summarize as quality_summary

ROOT = Path(__file__).resolve().parent
FIELDS = ("axis", "phase", "arm", "n_predictions", "macro_f1", "log_loss", "brier",
          "minimum_subject_macro_f1", "delta_f1_vs_F0v2", "delta_log_loss_vs_F0v2",
          "full_minus_removal_f1", "removal_minus_full_log_loss")


def scored(part: list[dict], classes: np.ndarray) -> dict:
    y = np.asarray([int(r["label"]) for r in part])
    p = np.asarray([[float(r[f"p_{c}"]) for c in classes] for r in part])
    users = np.asarray([int(r["subject"]) for r in part])
    if not np.isfinite(p).all() or np.any(p < 0) or np.max(np.abs(p.sum(axis=1) - 1)) > 1e-8:
        raise AssertionError("non-finite or unnormalized full-bank probabilities")
    return score(y, p, classes, users)


def audit() -> dict:
    cells, axis_audits = [], {}
    for axis in PROTOCOL["axes"]:
        path = ROOT / f"FULL_V1_LOFO_{axis}_PREDICTIONS.csv"
        result_path = ROOT / f"FULL_V1_LOFO_{axis}_RESULTS.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if (result["prediction_sha256"] != sha256(path)
                or result["prediction_rows"] != len(rows)
                or result["protocol_sha256"] != sha256(PROTOCOL_PATH)
                or result["f0v2_parent_replay_max_abs_error"] > 1e-8):
            raise AssertionError(f"full-bank output provenance changed: {axis}")
        parent_name = PROTOCOL["axes"][axis]
        for kind in ("result", "prediction"):
            expected = PROTOCOL[f"parent_{kind}s_sha256" if kind == "result" else
                                "parent_predictions_sha256"][parent_name]
            parent_path = ROOT / f"{parent_name}_{'RESULTS.json' if kind == 'result' else 'PREDICTIONS.csv'}"
            if sha256(parent_path) != expected:
                raise AssertionError(f"full-bank frozen parent changed: {axis}/{kind}")
        if set(r["arm"] for r in rows) != set(ARMS):
            raise AssertionError(f"missing full-bank arm: {axis}")
        classes = np.asarray(sorted(int(key[2:]) for key in rows[0] if key.startswith("p_")))
        unique = {(r["phase"], r["arm"], r["condition"], r["trial_id"]) for r in rows}
        if len(unique) != len(rows):
            raise AssertionError(f"duplicate matched full-bank row: {axis}")
        if axis != "synthetic_quality":
            trial_sets = {phase: {r["trial_id"] for r in rows if r["phase"] == phase}
                          for phase in ("validation", "final")}
            if trial_sets["validation"] & trial_sets["final"]:
                raise AssertionError(f"target split overlap: {axis}")
        counts = {}
        metrics = {}
        for phase in ("validation", "final"):
            keys = {arm: {(r["condition"], r["trial_id"]) for r in rows
                          if r["phase"] == phase and r["arm"] == arm} for arm in ARMS}
            if any(keys[arm] != keys["F0v2"] for arm in ARMS):
                raise AssertionError(f"unmatched LOFO target trials: {axis}/{phase}")
            counts[phase] = len(keys["F0v2"])
            metrics[phase] = {}
            for arm in ARMS:
                part = [r for r in rows if r["phase"] == phase and r["arm"] == arm]
                if axis == "synthetic_quality":
                    by_fault = {name: scored([r for r in part if r["condition"] == name], classes)
                                for name in result["scores"][arm][phase]}
                    for name, item in by_fault.items():
                        if abs(item["macro_f1"] - result["scores"][arm][phase][name]["macro_f1"]) > 1e-10:
                            raise AssertionError("quality score readback mismatch")
                    summary = quality_summary(by_fault)
                    stored = result["summary"][arm][phase]
                    for key in ("synthetic_quality_mean_macro_f1", "synthetic_quality_min_macro_f1",
                                "mean_named_fault_log_loss"):
                        if abs(summary[key] - stored[key]) > 1e-10:
                            raise AssertionError(f"quality summary changed: {key}")
                    item = {"macro_f1": summary["synthetic_quality_mean_macro_f1"],
                            "log_loss": summary["mean_named_fault_log_loss"],
                            "brier": float(np.mean([by_fault[name]["brier"]
                                                    for name in result["scores"][arm][phase] if name != "clean"])),
                            "minimum_subject_macro_f1": summary["minimum_subject_fault_macro_f1"]}
                else:
                    item = scored(part, classes)
                    stored = result["scores"][arm][phase]
                    if axis == "wearing_shift":
                        stored = stored["pooled"]
                    for key in ("macro_f1", "log_loss", "brier"):
                        if abs(item[key] - stored[key]) > 1e-10:
                            raise AssertionError(f"score readback changed: {axis}/{phase}/{arm}/{key}")
                metrics[phase][arm] = item
            for arm in ARMS:
                item, baseline, full = metrics[phase][arm], metrics[phase]["F0v2"], metrics[phase]["full"]
                cells.append({"axis": axis, "phase": phase, "arm": arm,
                              "n_predictions": counts[phase],
                              "macro_f1": item["macro_f1"], "log_loss": item["log_loss"],
                              "brier": item["brier"],
                              "minimum_subject_macro_f1": item["minimum_subject_macro_f1"],
                              "delta_f1_vs_F0v2": item["macro_f1"] - baseline["macro_f1"],
                              "delta_log_loss_vs_F0v2": item["log_loss"] - baseline["log_loss"],
                              "full_minus_removal_f1": full["macro_f1"] - item["macro_f1"]
                              if arm.startswith("minus_") else "",
                              "removal_minus_full_log_loss": item["log_loss"] - full["log_loss"]
                              if arm.startswith("minus_") else ""})
        axis_audits[axis] = {"prediction_sha256": sha256(path),
                             "result_sha256": sha256(result_path),
                             "prediction_rows": len(rows), "native_targets_per_phase": counts,
                             "f0v2_parent_replay_max_abs_error": result["f0v2_parent_replay_max_abs_error"]}
    if len(cells) != 98 or sum(item["prediction_rows"] for item in axis_audits.values()) != 26684:
        raise AssertionError("full-bank seven-axis inventory changed")
    path = ROOT / "FULL_V1_LOFO_CELLS.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(cells)
    validation = [r for r in cells if r["phase"] == "validation" and r["arm"] == "full"]
    final = [r for r in cells if r["phase"] == "final" and r["arm"] == "full"]
    audit_result = {"protocol_sha256": sha256(PROTOCOL_PATH),
                    "cells_sha256": sha256(path), "cells": len(cells),
                    "axis_audits": axis_audits,
                    "validation_full_minus_F0v2": {r["axis"]: r["delta_f1_vs_F0v2"] for r in validation},
                    "final_full_minus_F0v2": {r["axis"]: r["delta_f1_vs_F0v2"] for r in final},
                    "validation_full_positive_axes": sum(r["delta_f1_vs_F0v2"] > 0 for r in validation),
                    "scope": PROTOCOL["boundary"]}
    (ROOT / "FULL_V1_LOFO_AUDIT.json").write_text(json.dumps(audit_result, indent=2) + "\n",
                                                    encoding="utf-8")
    return audit_result


if __name__ == "__main__":
    result = audit()
    print(f"seven-axis full-bank LOFO: {result['cells']} cells, "
          f"{result['validation_full_positive_axes']}/7 positive validation axes")
