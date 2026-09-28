"""Versioned six-axis descriptive envelope with synthetic quality explicitly tagged."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from benchmarks.new_bank_v1.paired_analysis import csv_text

ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "ENVELOPE_QUALITY_PROTOCOL.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def build(verify: bool = False) -> dict:
    parent_path = ROOT / "ROBUSTNESS_ENVELOPE_5AXIS.csv"
    result_path = ROOT / "ROAM_QUALITY_RESULTS.json"
    screen_path = ROOT / "ROAM_QUALITY_SCREEN.csv"
    if (sha(parent_path) != PROTOCOL["parent_five_axis_envelope_sha256"]
            or sha(result_path) != PROTOCOL["quality_result_sha256"]
            or sha(screen_path) != PROTOCOL["quality_screen_sha256"]):
        raise AssertionError("frozen five-axis or quality source changed")
    parent = read(parent_path)
    quality = json.loads(result_path.read_text(encoding="utf-8"))
    screen = read(screen_path)
    if len(parent) != 8 or len(screen) != 672:
        raise AssertionError("source envelope or quality rows unavailable")
    output = []
    for phase in ("validation", "final"):
        for arm in PROTOCOL["arms"]:
            previous = [row for row in parent if row["phase"] == phase and row["feature_bank"] == arm]
            if len(previous) != 1:
                raise AssertionError("missing previous axis cell")
            previous = previous[0]
            cells = [row for row in screen if row["phase"] == phase
                     and row["feature_family"] == arm and row["scope"] == "subject"
                     and row["condition"] != "clean"]
            if len(cells) != 65:
                raise AssertionError("missing subject-by-fault cells")
            worst_fault_cell = min(cells, key=lambda row: float(row["macro_f1"]))
            summary = quality["summary"][arm][phase]
            observed = {axis: float(previous[f"R_{axis}_macro_f1"])
                        for axis in PROTOCOL["observed_axes"][:-1]}
            observed["quality_synthetic"] = summary["synthetic_quality_mean_macro_f1"]
            minimum_axis = min(observed, key=observed.get)
            prior_worst = float(previous["minimum_observed_cell_macro_f1"])
            if float(worst_fault_cell["macro_f1"]) < prior_worst:
                worst = {"macro_f1": float(worst_fault_cell["macro_f1"]),
                         "axis": "quality_synthetic", "scope": "subject_fault",
                         "subject": worst_fault_cell["subject"],
                         "condition": worst_fault_cell["condition"]}
            else:
                worst = {"macro_f1": prior_worst,
                         "axis": previous["minimum_observed_cell_axis"],
                         "scope": previous["minimum_observed_cell_scope"],
                         "subject": previous["minimum_observed_cell_subject"],
                         "condition": previous["minimum_observed_cell_condition"]}
            output.append({"phase": phase, "feature_bank": arm,
                           **{f"R_{axis}_macro_f1": observed[axis] for axis in PROTOCOL["observed_axes"]},
                           **{f"R_{axis}_macro_f1": "N/A" for axis in PROTOCOL["unobserved_axes"]},
                           "R_quality_synthetic_clean_macro_f1": summary["clean_macro_f1"],
                           "R_quality_synthetic_min_family_macro_f1": summary["synthetic_quality_min_macro_f1"],
                           "R_quality_synthetic_min_family": summary["synthetic_quality_min_family"],
                           "R_quality_synthetic_min_named_fault_macro_f1": summary["minimum_named_fault_macro_f1"],
                           "mean_R_available": sum(observed.values()) / len(observed),
                           "R_min": observed[minimum_axis], "R_min_axis": minimum_axis,
                           "minimum_observed_cell_macro_f1": worst["macro_f1"],
                           "minimum_observed_cell_axis": worst["axis"],
                           "minimum_observed_cell_scope": worst["scope"],
                           "minimum_observed_cell_subject": worst["subject"],
                           "minimum_observed_cell_condition": worst["condition"],
                           "observed_axes": 6, "unobserved_axes": 2,
                           "aggregation": "equal_axis_unweighted_descriptive_correlated_quality_posture_day_user"})
    if len(output) != 8:
        raise AssertionError("expected four arms by two phases")
    path = ROOT / "ROBUSTNESS_ENVELOPE_6AXIS_SYNTHETIC.csv"
    payload = csv_text(output).encode("utf-8")
    if verify:
        if path.read_bytes() != payload:
            raise AssertionError("six-axis synthetic envelope changed")
    else:
        path.write_bytes(payload)
    audit = {"status": "ok", "protocol_sha256": sha(PROTOCOL_PATH),
             "parent_five_axis_envelope_sha256": sha(parent_path),
             "quality_result_sha256": sha(result_path),
             "quality_screen_sha256": sha(screen_path), "rows": len(output),
             "observed_axes": PROTOCOL["observed_axes"],
             "unobserved_axes": PROTOCOL["unobserved_axes"], "boundary": PROTOCOL["boundary"]}
    payload = json.dumps(audit, indent=2) + "\n"
    audit_path = ROOT / "ENVELOPE_QUALITY_AUDIT.json"
    if verify:
        if audit_path.read_text(encoding="utf-8") != payload:
            raise AssertionError("six-axis audit changed")
    else:
        audit_path.write_text(payload, encoding="utf-8")
    print(json.dumps({"status": "ok", "verify": verify, "rows": len(output)}))
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    build(parser.parse_args().verify)
