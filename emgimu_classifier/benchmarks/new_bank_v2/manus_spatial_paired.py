"""Matched conditional, complementarity and interaction study on frozen MANUS predictions."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmarks.new_bank_v1.paired_analysis import analyze_group, csv_text

ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "MANUS_SPATIAL_PAIRED_PROTOCOL.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
PARENT = json.loads((ROOT / "MANUS_SPATIAL_RESULTS.json").read_text(encoding="utf-8"))
NAMES = tuple(PROTOCOL["arms"])
CLASSES = [str(i) for i in range(6)]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(verify: bool = False) -> dict:
    for name, field in (("MANUS_SPATIAL_PROTOCOL.json", "parent_protocol_sha256"),
                        ("MANUS_SPATIAL_RESULTS.json", "parent_result_sha256"),
                        ("MANUS_SPATIAL_PREDICTIONS.csv", "parent_prediction_sha256")):
        if sha(ROOT / name) != PROTOCOL[field]:
            raise AssertionError(f"frozen MANUS parent changed: {name}")
    with (ROOT / "MANUS_SPATIAL_PREDICTIONS.csv").open(newline="", encoding="utf-8") as stream:
        raw = list(csv.DictReader(stream))
    if len(raw) != 864 or NAMES != ("F0", "F0+F2a", "F0+F3c", "F0+F2a+F3c"):
        raise AssertionError("unexpected saved MANUS predictions or arms")
    outputs = {"MANUS_SPATIAL_SCREEN.csv": [], "MANUS_SPATIAL_CONDITIONAL.csv": [],
               "MANUS_SPATIAL_COMPLEMENTARITY.csv": [], "MANUS_SPATIAL_INTERACTION.csv": []}
    checked = 0
    for phase in ("validation", "final"):
        phase_rows = [row for row in raw if row["phase"] == phase]
        groups = [("pooled", "ALL", "ALL")]
        groups.extend(("subject", str(user), "ALL")
                      for user in PARENT["protocol"]["users"])
        groups.extend(("condition", "ALL", speed)
                      for speed in PARENT["protocol"]["conditions"])
        for scope, subject, condition in groups:
            by_arm = {arm: [row for row in phase_rows if row["arm"] == arm and
                            (scope != "subject" or row["subject"] == subject) and
                            (scope != "condition" or row["condition"] == condition)]
                      for arm in NAMES}
            screen, conditional, errors, interaction, calculated = analyze_group(
                "manus", phase, scope, subject, condition, by_arm, CLASSES, NAMES)
            outputs["MANUS_SPATIAL_SCREEN.csv"].extend(screen)
            outputs["MANUS_SPATIAL_CONDITIONAL.csv"].extend(conditional)
            outputs["MANUS_SPATIAL_COMPLEMENTARITY.csv"].append(errors)
            outputs["MANUS_SPATIAL_INTERACTION.csv"].append(interaction)
            for arm in NAMES:
                parent_score = PARENT["scores"][phase][arm][
                    "pooled" if scope == "pooled" else "by_user" if scope == "subject" else "by_speed"]
                if scope != "pooled":
                    parent_score = parent_score[subject if scope == "subject" else condition]
                actual = calculated[arm]["score"]
                for metric in ("trials", "accuracy", "macro_f1", "log_loss", "brier"):
                    if not np.isclose(float(actual[metric]), float(parent_score[metric]),
                                      rtol=0, atol=1e-10):
                        raise AssertionError(f"parent score changed: {phase}/{scope}/{arm}/{metric}")
                checked += 1
    expected = {"MANUS_SPATIAL_SCREEN.csv": 80, "MANUS_SPATIAL_CONDITIONAL.csv": 40,
                "MANUS_SPATIAL_COMPLEMENTARITY.csv": 20, "MANUS_SPATIAL_INTERACTION.csv": 20}
    if {name: len(rows) for name, rows in outputs.items()} != expected or checked != 80:
        raise AssertionError("unexpected matched cell count")
    for name, rows in outputs.items():
        payload = csv_text(rows).encode("utf-8")
        target = ROOT / name
        if verify:
            if target.read_bytes() != payload:
                raise AssertionError(f"derived MANUS table changed: {name}")
        else:
            target.write_bytes(payload)
    audit = {"status": "ok", "analysis_protocol_sha256": sha(PROTOCOL_PATH),
             "parent_prediction_sha256": PROTOCOL["parent_prediction_sha256"],
             "parent_score_groups_replayed": checked, "matched_cells": 20,
             "rows": expected, "boundary": PROTOCOL["scope"]}
    payload = json.dumps(audit, indent=2) + "\n"
    target = ROOT / "MANUS_SPATIAL_PAIRED_AUDIT.json"
    if verify:
        if target.read_text(encoding="utf-8") != payload:
            raise AssertionError("derived MANUS audit changed")
    else:
        target.write_text(payload, encoding="utf-8")
    print(json.dumps({"status": "ok", "verify": verify, "rows": expected}))
    return audit


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    build(parser.parse_args().verify)
