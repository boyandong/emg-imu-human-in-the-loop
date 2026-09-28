"""Seven-axis availability and three-axis new-v2 robustness envelope."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "ENVELOPE_PROTOCOL.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
SOURCE = ROOT / "FAMILY_SCREEN.csv"


def csv_text(rows: list[dict]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def build(verify: bool = False) -> dict:
    if hashlib.sha256((ROOT / "FAMILY_SCREEN_PROTOCOL.json").read_bytes()).hexdigest() != PROTOCOL["parent_family_screen_protocol_sha256"]:
        raise AssertionError("parent family screen protocol changed")
    audit = json.loads((ROOT / "FAMILY_SCREEN_AUDIT.json").read_text(encoding="utf-8"))
    if audit["rows"]["FAMILY_SCREEN.csv"] != 256:
        raise AssertionError("parent family screen audit unavailable")
    with SOURCE.open(newline="", encoding="utf-8") as stream:
        source = list(csv.DictReader(stream))
    if len(source) != 256:
        raise AssertionError("family screen size changed")
    output = []
    for phase in ("validation", "final"):
        for arm in PROTOCOL["arms"]:
            values, per_axis = [], {}
            for axis in PROTOCOL["observed_axes"]:
                cell = [row for row in source if row["axis"] == axis and
                        row["phase"] == phase and row["feature_family"] == arm and
                        row["scope"] == "pooled"]
                if len(cell) != 1:
                    raise AssertionError(f"expected one pooled cell for {axis}/{phase}/{arm}")
                value = float(cell[0]["macro_f1"])
                per_axis[axis] = value
                values.append(value)
            subcells = [row for row in source if row["axis"] in PROTOCOL["observed_axes"] and
                        row["phase"] == phase and row["feature_family"] == arm and
                        row["scope"] in ("subject", "condition")]
            if len(subcells) != 28:
                raise AssertionError("expected 28 observed subject/condition cells per phase and arm")
            worst = min(subcells, key=lambda row: float(row["macro_f1"]))
            minimum_axis = min(per_axis, key=per_axis.get)
            output.append({"phase": phase, "feature_bank": arm,
                           "R_force_macro_f1": per_axis["force"],
                           "R_wearing_macro_f1": per_axis["wearing"],
                           "R_day_macro_f1": per_axis["day"],
                           **{f"R_{axis}_macro_f1": "N/A" for axis in PROTOCOL["unobserved_axes"]},
                           "mean_R_available": sum(values) / len(values),
                           "R_min": min(values), "R_min_axis": minimum_axis,
                           "minimum_observed_cell_macro_f1": float(worst["macro_f1"]),
                           "minimum_observed_cell_axis": worst["axis"],
                           "minimum_observed_cell_scope": worst["scope"],
                           "minimum_observed_cell_subject": worst["subject"],
                           "minimum_observed_cell_condition": worst["condition"],
                           "observed_axes": 3, "unobserved_axes": 4,
                           "aggregation": "equal_axis_unweighted_descriptive"})
    if len(output) != 8:
        raise AssertionError("expected four arms by two phases")
    path = ROOT / "ROBUSTNESS_ENVELOPE.csv"
    content = csv_text(output)
    if verify:
        if path.read_bytes() != content.encode("utf-8"):
            raise AssertionError("robustness envelope changed")
    else:
        path.write_bytes(content.encode("utf-8"))
    report = {"status": "ok", "protocol_sha256": hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
              "family_screen_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              "rows": 8, "observed_axes": PROTOCOL["observed_axes"],
              "unobserved_axes": PROTOCOL["unobserved_axes"],
              "boundary": PROTOCOL["boundary"]}
    audit_path = ROOT / "ENVELOPE_AUDIT.json"
    audit_text = json.dumps(report, indent=2) + "\n"
    if verify:
        if audit_path.read_text(encoding="utf-8") != audit_text:
            raise AssertionError("robustness envelope audit changed")
    else:
        audit_path.write_text(audit_text, encoding="utf-8")
    print(json.dumps({"status": "ok", "verify": verify, "rows": len(output)}))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    build(parser.parse_args().verify)
