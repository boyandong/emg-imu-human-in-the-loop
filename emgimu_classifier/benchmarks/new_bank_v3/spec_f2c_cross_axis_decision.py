"""Read back frozen V3 F2c public-axis results and apply the existing guard."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


V3 = Path(__file__).resolve().parent
V2 = V3.parent / "new_bank_v2"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(data: dict, axis: str, phase: str, arm: str) -> dict:
    if arm == "candidate":
        return (data["scores"][phase]["pooled"] if axis == "wearing_shift" else
                data["scores"][axis][phase]["pooled"] if axis == "manus_session" else
                data["scores"][axis][phase])
    return (data["scores"][phase]["F0v2"]["pooled"] if axis != "grab_unseen_user" else
            data["scores"]["F0v2"][phase])


def main() -> None:
    protocol_path = V3 / "SPEC_F2C_CROSS_AXIS_PROTOCOL.json"
    protocol = read(protocol_path)
    wearing = read(V3 / "SPEC_F2C_WEARING_RESULTS.json")
    transfer = read(V3 / "SPEC_F2C_TRANSFER_RESULTS.json")
    parents = {
        "wearing_shift": (V2 / "F2A_DOCUMENT_WEARING_RESULTS.json", V2 / "WEARING_RESULTS.json", wearing),
        "manus_session": (V2 / "F2_MANUS_CANDIDATES_RESULTS.json", V2 / "F2_MANUS_CANDIDATES_RESULTS.json", transfer),
        "grab_unseen_user": (V2 / "F2_GRAB_CANDIDATES_RESULTS.json", V2 / "F2_GRAB_CANDIDATES_RESULTS.json", transfer),
    }
    rows = []
    violations = []
    for axis, (parent_path, f0_path, candidate) in parents.items():
        assert sha(parent_path) == protocol["parent_result_sha256"][axis]
        assert candidate["protocol_sha256"] == sha(protocol_path)
        f0 = read(f0_path)
        for phase in ("validation", "final"):
            base = score(f0, axis, phase, "F0v2")
            newer = score(candidate, axis, phase, "candidate")
            assert base["trials"] == newer["trials"]
            row = {
                "axis": axis, "phase": phase, "trials": base["trials"],
                "f0_macro_f1": base["macro_f1"], "f2c_macro_f1": newer["macro_f1"],
                "delta_macro_f1": newer["macro_f1"] - base["macro_f1"],
                "f0_log_loss": base["log_loss"], "f2c_log_loss": newer["log_loss"],
                "delta_log_loss": newer["log_loss"] - base["log_loss"],
            }
            rows.append(row)
            if phase == "validation":
                if row["delta_macro_f1"] < -1e-12:
                    violations.append(f"{axis}:macro_f1")
                if row["delta_log_loss"] > 1e-12:
                    violations.append(f"{axis}:log_loss")
    csv_path = V3 / "SPEC_F2C_CROSS_AXIS_CELLS.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    audit = {
        "status": "centered_v3_f2c_matched_public_guard",
        "protocol_sha256": sha(protocol_path),
        "source_results_sha256": {
            "SPEC_F2C_WEARING_RESULTS.json": sha(V3 / "SPEC_F2C_WEARING_RESULTS.json"),
            "SPEC_F2C_TRANSFER_RESULTS.json": sha(V3 / "SPEC_F2C_TRANSFER_RESULTS.json"),
        },
        "cells_sha256": sha(csv_path), "cells": len(rows),
        "validation_violations": violations,
        "candidate_eligible_for_public_default": not violations,
        "three_axis_default": "F0v2" if violations else protocol["candidate"],
        "boundary": "Previously inspected public data, validation guard only; finals descriptive. No prospective, own-device or live-recognition claim.",
    }
    (V3 / "SPEC_F2C_CROSS_AXIS_AUDIT.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
