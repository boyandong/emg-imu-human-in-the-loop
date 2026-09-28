"""Versioned five-axis envelope with independent public 8-channel posture data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from benchmarks.new_bank_v1.paired_analysis import csv_text

ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "ENVELOPE_POSTURE_PROTOCOL.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def build(verify: bool = False) -> dict:
    old_path = ROOT / "GRAB_USER_SCREEN.csv"
    posture_path = ROOT / "ROAM_POSTURE_SCREEN.csv"
    if (sha(old_path) != PROTOCOL["parent_four_axis_screen_sha256"]
            or sha(ROOT / "ROBUSTNESS_ENVELOPE_4AXIS.csv") != PROTOCOL["parent_four_axis_envelope_sha256"]
            or sha(posture_path) != PROTOCOL["new_posture_screen_sha256"]):
        raise AssertionError("frozen new-v2 envelope sources changed")
    # The four-axis table is the frozen parent for axis scores and worst cells.
    old_envelope = read(ROOT / "ROBUSTNESS_ENVELOPE_4AXIS.csv")
    posture = read(posture_path)
    if len(old_envelope) != 8 or len(posture) != 240:
        raise AssertionError("four-axis or ROAM source row count changed")
    output = []
    for phase in ("validation", "final"):
        for arm in PROTOCOL["arms"]:
            parent = [row for row in old_envelope if row["phase"] == phase
                      and row["feature_bank"] == arm]
            pooled = [row for row in posture if row["phase"] == phase
                      and row["feature_family"] == arm and row["scope"] == "pooled"]
            posture_cells = [row for row in posture if row["phase"] == phase
                             and row["feature_family"] == arm and row["scope"] == "condition"]
            individual_cells = [row for row in posture if row["phase"] == phase
                                and row["feature_family"] == arm
                                and row["scope"] in ("condition", "subject", "subject_condition")]
            if len(parent) != 1 or len(pooled) != 1 or len(posture_cells) != 4 or len(individual_cells) != 29:
                raise AssertionError(f"missing paired posture cells: {phase}/{arm}")
            parent = parent[0]
            observed = {axis: float(parent[f"R_{axis}_macro_f1"])
                        for axis in PROTOCOL["observed_axes"][:-1]}
            observed["posture"] = float(pooled[0]["macro_f1"])
            worst_posture = min(posture_cells, key=lambda row: float(row["macro_f1"]))
            worst_roam_cell = min(individual_cells, key=lambda row: float(row["macro_f1"]))
            prior_worst = float(parent["minimum_observed_cell_macro_f1"])
            if float(worst_roam_cell["macro_f1"]) < prior_worst:
                worst = {"macro_f1": worst_roam_cell["macro_f1"], "axis": "posture",
                         "scope": worst_roam_cell["scope"], "subject": worst_roam_cell["subject"],
                         "condition": worst_roam_cell["condition"]}
            else:
                worst = {"macro_f1": prior_worst,
                         "axis": parent["minimum_observed_cell_axis"],
                         "scope": parent["minimum_observed_cell_scope"],
                         "subject": parent["minimum_observed_cell_subject"],
                         "condition": parent["minimum_observed_cell_condition"]}
            minimum_axis = min(observed, key=observed.get)
            output.append({"phase": phase, "feature_bank": arm,
                           **{f"R_{axis}_macro_f1": observed[axis] for axis in PROTOCOL["observed_axes"]},
                           **{f"R_{axis}_macro_f1": "N/A" for axis in PROTOCOL["unobserved_axes"]},
                           "R_posture_min_macro_f1": float(worst_posture["macro_f1"]),
                           "R_posture_min_condition": worst_posture["condition"],
                           "mean_R_available": sum(observed.values()) / len(observed),
                           "R_min": observed[minimum_axis], "R_min_axis": minimum_axis,
                           "minimum_observed_cell_macro_f1": float(worst["macro_f1"]),
                           "minimum_observed_cell_axis": worst["axis"],
                           "minimum_observed_cell_scope": worst["scope"],
                           "minimum_observed_cell_subject": worst["subject"],
                           "minimum_observed_cell_condition": worst["condition"],
                           "observed_axes": 5, "unobserved_axes": 2,
                           "aggregation": "equal_axis_unweighted_descriptive_correlated_day_user"})
    if len(output) != 8:
        raise AssertionError("expected four arms by two phases")
    target = ROOT / "ROBUSTNESS_ENVELOPE_5AXIS.csv"
    payload = csv_text(output).encode("utf-8")
    if verify:
        if target.read_bytes() != payload:
            raise AssertionError("five-axis envelope changed")
    else:
        target.write_bytes(payload)
    audit = {"status": "ok", "protocol_sha256": sha(PROTOCOL_PATH),
             "parent_four_axis_table_sha256": sha(ROOT / "ROBUSTNESS_ENVELOPE_4AXIS.csv"),
             "new_posture_screen_sha256": sha(posture_path), "rows": len(output),
             "observed_axes": PROTOCOL["observed_axes"],
             "unobserved_axes": PROTOCOL["unobserved_axes"], "boundary": PROTOCOL["boundary"]}
    payload = json.dumps(audit, indent=2) + "\n"
    path = ROOT / "ENVELOPE_POSTURE_AUDIT.json"
    if verify:
        if path.read_text(encoding="utf-8") != payload:
            raise AssertionError("five-axis audit changed")
    else:
        path.write_text(payload, encoding="utf-8")
    print(json.dumps({"status": "ok", "verify": verify, "rows": len(output)}))
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    build(parser.parse_args().verify)
