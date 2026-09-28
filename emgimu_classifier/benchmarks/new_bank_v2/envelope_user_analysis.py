"""Versioned four-axis envelope incorporating subject-disjoint GRABMyo user test."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from benchmarks.new_bank_v1.paired_analysis import csv_text

ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "ENVELOPE_USER_PROTOCOL.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def build(verify: bool = False) -> dict:
    old = ROOT / "FAMILY_SCREEN.csv"
    new = ROOT / "GRAB_USER_SCREEN.csv"
    if (sha(old) != PROTOCOL["parent_three_axis_screen_sha256"]
            or sha(new) != PROTOCOL["new_user_axis_screen_sha256"]):
        raise AssertionError("frozen new-v2 family screen source changed")
    parent_audit = json.loads((ROOT / "FAMILY_SCREEN_AUDIT.json").read_text(encoding="utf-8"))
    user_audit = json.loads((ROOT / "GRAB_USER_PAIRED_AUDIT.json").read_text(encoding="utf-8"))
    if (parent_audit["rows"]["FAMILY_SCREEN.csv"] != 256
            or user_audit["rows"]["GRAB_USER_SCREEN.csv"] != 24):
        raise AssertionError("screen audits unavailable")
    previous = read(old)
    user = read(new)
    if len(previous) != 256 or len(user) != 24:
        raise AssertionError("screen row counts changed")
    output = []
    for phase in ("validation", "final"):
        for arm in PROTOCOL["arms"]:
            observed = {}
            for axis in PROTOCOL["observed_axes"]:
                source = user if axis == "user" else previous
                source_axis = "grab_user" if axis == "user" else axis
                cell = [row for row in source if row["axis"] == source_axis and
                        row["phase"] == phase and row["feature_family"] == arm and
                        row["scope"] == "pooled"]
                if len(cell) != 1:
                    raise AssertionError(f"missing pooled {axis}/{phase}/{arm}")
                observed[axis] = float(cell[0]["macro_f1"])
            cells = [row for row in previous if row["axis"] in ("force", "wearing", "day") and
                     row["phase"] == phase and row["feature_family"] == arm and
                     row["scope"] in ("subject", "condition")]
            cells += [row for row in user if row["axis"] == "grab_user" and
                      row["phase"] == phase and row["feature_family"] == arm and
                      row["scope"] == "subject"]
            if len(cells) != 30:
                raise AssertionError("expected 28 prior and two subject-disjoint user cells")
            worst = min(cells, key=lambda row: float(row["macro_f1"]))
            minimum_axis = min(observed, key=observed.get)
            output.append({"phase": phase, "feature_bank": arm,
                           **{f"R_{axis}_macro_f1": observed[axis] for axis in PROTOCOL["observed_axes"]},
                           **{f"R_{axis}_macro_f1": "N/A" for axis in PROTOCOL["unobserved_axes"]},
                           "mean_R_available": sum(observed.values()) / len(observed),
                           "R_min": min(observed.values()), "R_min_axis": minimum_axis,
                           "minimum_observed_cell_macro_f1": float(worst["macro_f1"]),
                           "minimum_observed_cell_axis": "user" if worst["axis"] == "grab_user" else worst["axis"],
                           "minimum_observed_cell_scope": worst["scope"],
                           "minimum_observed_cell_subject": worst["subject"],
                           "minimum_observed_cell_condition": worst["condition"],
                           "observed_axes": 4, "unobserved_axes": 3,
                           "aggregation": "equal_axis_unweighted_descriptive_correlated_day_user"})
    if len(output) != 8:
        raise AssertionError("expected four arms by two phases")
    target = ROOT / "ROBUSTNESS_ENVELOPE_4AXIS.csv"
    payload = csv_text(output).encode("utf-8")
    if verify:
        if target.read_bytes() != payload:
            raise AssertionError("four-axis robustness table changed")
    else:
        target.write_bytes(payload)
    audit = {"status": "ok", "protocol_sha256": sha(PROTOCOL_PATH),
             "old_family_screen_sha256": sha(old), "new_user_screen_sha256": sha(new),
             "rows": len(output), "observed_axes": PROTOCOL["observed_axes"],
             "unobserved_axes": PROTOCOL["unobserved_axes"],
             "boundary": PROTOCOL["boundary"]}
    payload = json.dumps(audit, indent=2) + "\n"
    audit_path = ROOT / "ENVELOPE_USER_AUDIT.json"
    if verify:
        if audit_path.read_text(encoding="utf-8") != payload:
            raise AssertionError("four-axis audit changed")
    else:
        audit_path.write_text(payload, encoding="utf-8")
    print(json.dumps({"status": "ok", "verify": verify, "rows": len(output)}))
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    build(parser.parse_args().verify)
