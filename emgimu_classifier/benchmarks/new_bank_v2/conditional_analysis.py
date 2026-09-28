"""Compute Stage-2 conditional additions from frozen new-v2 family scores."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "CONDITIONAL_PROTOCOL.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
SOURCE = ROOT / "FAMILY_SCREEN.csv"
PAIRS = ROOT / "F2A_F3C_INTERACTION.csv"
KEYS = ("dataset", "axis", "phase", "scope", "subject", "condition")


def csv_text(rows: list[dict]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def build(verify: bool = False) -> dict:
    for name, expected in (("FAMILY_SCREEN_PROTOCOL.json", "parent_family_screen_protocol_sha256"),
                           ("PAIR_PROTOCOL.json", "parent_pair_protocol_sha256")):
        digest = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if digest != PROTOCOL[expected]:
            raise AssertionError(f"parent protocol changed: {name}")
    family_audit = json.loads((ROOT / "FAMILY_SCREEN_AUDIT.json").read_text(encoding="utf-8"))
    pair_audit = json.loads((ROOT / "PAIR_AUDIT.json").read_text(encoding="utf-8"))
    if family_audit["rows"]["FAMILY_SCREEN.csv"] != 256 or pair_audit["rows"]["F2A_F3C_INTERACTION.csv"] != 64:
        raise AssertionError("parent tables not audited")
    with SOURCE.open(newline="", encoding="utf-8") as stream:
        summary = list(csv.DictReader(stream))
    with PAIRS.open(newline="", encoding="utf-8") as stream:
        interaction = list(csv.DictReader(stream))
    if len(summary) != 256 or len(interaction) != 64:
        raise AssertionError("parent table lengths changed")
    cells = {}
    for row in summary:
        key = tuple(row[k] for k in KEYS)
        arms = cells.setdefault(key, {})
        arm = row["feature_family"]
        if arm in arms:
            raise AssertionError("duplicate family score cell")
        arms[arm] = row
    if len(cells) != 64:
        raise AssertionError("unexpected number of family score cells")
    by_key = {tuple(row[k] for k in KEYS): row for row in interaction}
    if set(by_key) != set(cells):
        raise AssertionError("conditional cells differ from matched prediction cells")
    output = []
    for key, arms in cells.items():
        if len(arms) != 4:
            raise AssertionError("incomplete four-arm family cell")
        parent = by_key[key]
        for comparison in PROTOCOL["comparisons"]:
            core = arms[comparison["core"]]
            increment = arms[comparison["increment"]]
            if core["evaluation_trials"] != increment["evaluation_trials"] or core["evaluation_trials"] != parent["evaluation_trials"]:
                raise AssertionError("unmatched trial count in conditional comparison")
            output.append({**{k: core[k] for k in KEYS},
                           "calibration_budget": 0,
                           "evaluation_unit": "whole_native_trial",
                           "evaluation_trials": core["evaluation_trials"],
                           "core_bank": comparison["core"],
                           "added_family": comparison["added_family"],
                           "increment_bank": comparison["increment"],
                           "delta_log_loss": float(core["log_loss"]) - float(increment["log_loss"]),
                           "delta_brier": float(core["brier"]) - float(increment["brier"]),
                           "delta_macro_f1": float(increment["macro_f1"]) - float(core["macro_f1"]),
                           "delta_accuracy": float(increment["accuracy"]) - float(core["accuracy"]),
                           "delta_ece": float(core["ece"]) - float(increment["ece"]),
                           "core_per_class_f1_json": core["per_class_f1_json"],
                           "increment_per_class_f1_json": increment["per_class_f1_json"]})
    if len(output) != 128:
        raise AssertionError("expected two conditional comparisons per cell")
    path = ROOT / "CONDITIONAL_VALUE.csv"
    content = csv_text(output)
    if verify:
        if path.read_bytes() != content.encode("utf-8"):
            raise AssertionError("conditional table changed")
    else:
        path.write_bytes(content.encode("utf-8"))
    audit = {"status": "ok", "protocol_sha256": hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
             "family_screen_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
             "pair_interaction_sha256": hashlib.sha256(PAIRS.read_bytes()).hexdigest(),
             "rows": len(output), "matched_cells": len(cells), "boundary": PROTOCOL["boundary"]}
    audit_path = ROOT / "CONDITIONAL_AUDIT.json"
    audit_text = json.dumps(audit, indent=2) + "\n"
    if verify:
        if audit_path.read_text(encoding="utf-8") != audit_text:
            raise AssertionError("conditional audit changed")
    else:
        audit_path.write_text(audit_text, encoding="utf-8")
    print(json.dumps({"status": "ok", "verify": verify, "rows": len(output)}))
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    build(parser.parse_args().verify)
