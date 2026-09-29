"""Conservative validation-only decision across seven saved new-v1 screens."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from benchmarks.new_bank_v2.roam_posture_run import sha256

ROOT = Path(__file__).resolve().parent
PREFIX = "V1_CROSS_AXIS_DECISION"
PROTOCOL_PATH = ROOT / f"{PREFIX}_PROTOCOL.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
SOURCE_AXIS = dict(zip(PROTOCOL["axes"], PROTOCOL["source_results_sha256"]))


def measures(source: str, result: dict, phase: str, arm: str) -> dict:
    if source == "ROAM_V1_QUALITY":
        summary = result["summary"][phase][arm]
        return {"macro_f1": summary["synthetic_quality_mean_macro_f1"],
                "log_loss": summary["mean_named_fault_log_loss"],
                "clean_f1": summary["clean_macro_f1"],
                "minimum_fault_f1": summary["minimum_named_fault_macro_f1"]}
    if source in ("ROAM_V1_EXTENSION", "GRAB_V1_EXTENSION", "GRAB_DAY_V1_EXTENSION"):
        score = result["scores"][arm][phase]
    else:
        score = result["scores"][phase][arm]["pooled"]
    return {"macro_f1": score["macro_f1"], "log_loss": score["log_loss"],
            "clean_f1": None, "minimum_fault_f1": None}


def build() -> dict:
    if len(SOURCE_AXIS) != 7 or len(PROTOCOL["arms"]) != 5:
        raise AssertionError("seven-axis decision scope changed")
    source_results = {}
    for axis, source in SOURCE_AXIS.items():
        path = ROOT / f"{source}_RESULTS.json"
        if sha256(path) != PROTOCOL["source_results_sha256"][source]:
            raise AssertionError(f"frozen result changed: {source}")
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["protocol"]["arms"] != PROTOCOL["arms"]:
            raise AssertionError(f"arm identity changed: {source}")
        prediction = ROOT / f"{source}_PREDICTIONS.csv"
        if not prediction.is_file() or result["prediction_sha256"] != sha256(prediction):
            raise AssertionError(f"saved predictions changed: {source}")
        source_results[axis] = (source, result)
    rows = []
    tolerance = PROTOCOL["tolerance"]
    violations = {arm: [] for arm in PROTOCOL["arms"][1:]}
    for axis, (source, result) in source_results.items():
        for phase in ("validation", "final"):
            baseline = measures(source, result, phase, "F0v2")
            for arm in PROTOCOL["arms"]:
                current = measures(source, result, phase, arm)
                delta_f1 = current["macro_f1"] - baseline["macro_f1"]
                delta_loss = current["log_loss"] - baseline["log_loss"]
                row = {"axis": axis, "source": source, "phase": phase, "arm": arm,
                       "macro_f1": current["macro_f1"], "log_loss": current["log_loss"],
                       "delta_macro_f1": delta_f1, "delta_log_loss": delta_loss,
                       "clean_f1": current["clean_f1"],
                       "minimum_fault_f1": current["minimum_fault_f1"],
                       "delta_clean_f1": (None if current["clean_f1"] is None else
                                          current["clean_f1"] - baseline["clean_f1"]),
                       "delta_minimum_fault_f1": (None if current["minimum_fault_f1"] is None
                                                   else current["minimum_fault_f1"]
                                                   - baseline["minimum_fault_f1"])}
                rows.append(row)
                if phase == "validation" and arm != "F0v2":
                    failures = []
                    if delta_f1 < -tolerance:
                        failures.append("macro_f1")
                    if delta_loss > tolerance:
                        failures.append("log_loss")
                    if row["delta_clean_f1"] is not None and row["delta_clean_f1"] < -tolerance:
                        failures.append("quality_clean_f1")
                    if (row["delta_minimum_fault_f1"] is not None
                            and row["delta_minimum_fault_f1"] < -tolerance):
                        failures.append("quality_minimum_fault_f1")
                    for metric in failures:
                        violations[arm].append({"axis": axis, "metric": metric})
    if len(rows) != 70:
        raise AssertionError("seven-axis arm-phase coverage changed")
    eligible = []
    for arm in PROTOCOL["arms"][1:]:
        improved = any(row["phase"] == "validation" and row["arm"] == arm
                       and (row["delta_macro_f1"] > tolerance
                            or row["delta_log_loss"] < -tolerance) for row in rows)
        if not violations[arm] and improved:
            eligible.append(arm)
    if len(eligible) > 1:
        raise AssertionError("policy needs an explicit tie break before promoting additions")
    decision = eligible[0] if eligible else "F0v2"
    path = ROOT / f"{PREFIX}_CELLS.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    audit = {"status": "ok", "protocol_sha256": sha256(PROTOCOL_PATH),
             "source_results_sha256": PROTOCOL["source_results_sha256"],
             "saved_cells_sha256": sha256(path), "cells": len(rows),
             "validation_axes": PROTOCOL["axes"],
             "validation_violations": violations, "eligible_additions": eligible,
             "deployment_default": decision,
             "selection_uses_final": False,
             "boundary": PROTOCOL["interpretation"]}
    (ROOT / f"{PREFIX}_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n",
                                                   encoding="utf-8")
    print(json.dumps({"cells": len(rows), "deployment_default": decision,
                      "violations": {arm: len(value) for arm, value in violations.items()}}), flush=True)
    return audit


if __name__ == "__main__":
    build()
