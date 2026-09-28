"""Qualified seven-axis envelope with public MANUS and external-source Rest."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from benchmarks.new_bank_v1.paired_analysis import csv_text

ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "ENVELOPE_SPEED_PROTOCOL.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def build(verify: bool = False) -> dict:
    parent_path = ROOT / "ROBUSTNESS_ENVELOPE_6AXIS_SYNTHETIC.csv"
    result_path = ROOT / "MANUS_REST_TRANSFER_RESULTS.json"
    screen_path = ROOT / "MANUS_REST_TRANSFER_SCREEN.csv"
    if (sha(parent_path) != PROTOCOL["parent_six_axis_envelope_sha256"]
            or sha(result_path) != PROTOCOL["speed_result_sha256"]
            or sha(screen_path) != PROTOCOL["speed_screen_sha256"]):
        raise AssertionError("frozen six-axis or MANUS source changed")
    parent = read(parent_path)
    speed = json.loads(result_path.read_text(encoding="utf-8"))
    screen = read(screen_path)
    if len(parent) != 8 or len(screen) != 224:
        raise AssertionError("source envelope or speed rows unavailable")
    output = []
    for phase in ("validation", "final"):
        for arm in PROTOCOL["arms"]:
            prior = [row for row in parent if row["phase"] == phase and row["feature_bank"] == arm]
            if len(prior) != 1:
                raise AssertionError("missing prior phase/arm cell")
            prior = prior[0]
            current = speed["scores"][phase][arm]
            observed = {axis: float(prior[f"R_{axis}_macro_f1"])
                        for axis in PROTOCOL["observed_axes"][:-1]}
            observed["speed_external_rest"] = current["pooled"]["macro_f1"]
            minimum_axis = min(observed, key=observed.get)
            speed_cells = [row for row in screen if row["phase"] == phase
                           and row["feature_family"] == arm and row["scope"] == "subject_condition"]
            if len(speed_cells) != 18:
                raise AssertionError("missing subject-by-speed cells")
            worst_speed_cell = min(speed_cells, key=lambda row: float(row["macro_f1"]))
            prior_worst = float(prior["minimum_observed_cell_macro_f1"])
            if float(worst_speed_cell["macro_f1"]) < prior_worst:
                worst = {"macro_f1": float(worst_speed_cell["macro_f1"]),
                         "axis": "speed_external_rest", "scope": "subject_speed",
                         "subject": worst_speed_cell["subject"],
                         "condition": worst_speed_cell["condition"]}
            else:
                worst = {"macro_f1": prior_worst,
                         "axis": prior["minimum_observed_cell_axis"],
                         "scope": prior["minimum_observed_cell_scope"],
                         "subject": prior["minimum_observed_cell_subject"],
                         "condition": prior["minimum_observed_cell_condition"]}
            by_speed = current["by_speed"]
            slowest = min(by_speed, key=lambda name: by_speed[name]["macro_f1"])
            output.append({"phase": phase, "feature_bank": arm,
                           **{f"R_{axis}_macro_f1": observed[axis] for axis in PROTOCOL["observed_axes"]},
                           "R_quality_real_macro_f1": "N/A",
                           "R_speed_min_macro_f1": by_speed[slowest]["macro_f1"],
                           "R_speed_min_condition": slowest,
                           "mean_R_available": sum(observed.values()) / len(observed),
                           "R_min": observed[minimum_axis], "R_min_axis": minimum_axis,
                           "minimum_observed_cell_macro_f1": worst["macro_f1"],
                           "minimum_observed_cell_axis": worst["axis"],
                           "minimum_observed_cell_scope": worst["scope"],
                           "minimum_observed_cell_subject": worst["subject"],
                           "minimum_observed_cell_condition": worst["condition"],
                           "observed_axes": 7, "unobserved_axes": 1,
                           "aggregation": "qualified_equal_axis_unweighted_descriptive_correlated_public_axes"})
    if len(output) != 8:
        raise AssertionError("expected four arms by two phases")
    path = ROOT / "ROBUSTNESS_ENVELOPE_7AXIS_QUALIFIED.csv"
    payload = csv_text(output).encode("utf-8")
    if verify:
        if path.read_bytes() != payload:
            raise AssertionError("seven-axis qualified envelope changed")
    else:
        path.write_bytes(payload)
    audit = {"status": "ok", "protocol_sha256": sha(PROTOCOL_PATH),
             "parent_six_axis_envelope_sha256": sha(parent_path),
             "speed_result_sha256": sha(result_path),
             "speed_screen_sha256": sha(screen_path), "rows": len(output),
             "observed_axes": PROTOCOL["observed_axes"],
             "unobserved_axes": PROTOCOL["unobserved_axes"], "boundary": PROTOCOL["boundary"]}
    payload = json.dumps(audit, indent=2) + "\n"
    audit_path = ROOT / "ENVELOPE_SPEED_AUDIT.json"
    if verify:
        if audit_path.read_text(encoding="utf-8") != payload:
            raise AssertionError("seven-axis qualified audit changed")
    else:
        audit_path.write_text(payload, encoding="utf-8")
    print(json.dumps({"status": "ok", "verify": verify, "rows": len(output)}))
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    build(parser.parse_args().verify)
