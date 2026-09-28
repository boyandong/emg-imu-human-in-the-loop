"""Frozen-prediction four-arm paired analysis of the GRABMyo user axis."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmarks.new_bank_v1.paired_analysis import analyze_group, csv_text

ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "GRAB_USER_PAIRED_PROTOCOL.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
PARENT = json.loads((ROOT / "GRAB_USER_RESULTS.json").read_text(encoding="utf-8"))
NAMES = tuple(PROTOCOL["arms"])
CLASSES = [str(value) for value in PROTOCOL["classes"]]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(verify: bool = False) -> dict:
    for name, field in (("GRAB_USER_PROTOCOL.json", "parent_protocol_sha256"),
                        ("GRAB_USER_RESULTS.json", "parent_result_sha256"),
                        ("GRAB_USER_PREDICTIONS.csv", "parent_prediction_sha256")):
        if sha(ROOT / name) != PROTOCOL[field]:
            raise AssertionError(f"frozen same-day parent changed: {name}")
    with (ROOT / "GRAB_USER_PREDICTIONS.csv").open(newline="", encoding="utf-8") as stream:
        raw = list(csv.DictReader(stream))
    if len(raw) != 448 or NAMES != ("F0v2", "F0v2+F2a", "F0v2+F3c", "F0v2+F2a+F3c"):
        raise AssertionError("unexpected saved user-axis trial predictions")
    normalized = [{"arm": row["arm"], "phase": row["phase"], "trial_id": row["trial_id"],
                   "subject": row["subject"], "condition": "Day1", "label": row["gesture"],
                   **{f"p_{c}": row[f"p_{c}"] for c in CLASSES}} for row in raw]
    outputs = {"GRAB_USER_SCREEN.csv": [], "GRAB_USER_CONDITIONAL.csv": [],
               "GRAB_USER_COMPLEMENTARITY.csv": [], "GRAB_USER_INTERACTION.csv": []}
    checked = 0
    for phase in ("validation", "final"):
        users = PARENT["protocol"][f"{phase}_subjects"]
        groups = [("pooled", "ALL", "Day1")]
        groups.extend(("subject", str(user), "Day1") for user in users)
        for scope, subject, condition in groups:
            by_arm = {arm: [row for row in normalized if row["phase"] == phase and
                            row["arm"] == arm and (scope == "pooled" or row["subject"] == subject)]
                      for arm in NAMES}
            screen, conditional, errors, interaction, computed = analyze_group(
                "grab_user", phase, scope, subject, condition, by_arm, CLASSES, NAMES)
            outputs["GRAB_USER_SCREEN.csv"].extend(screen)
            outputs["GRAB_USER_CONDITIONAL.csv"].extend(conditional)
            outputs["GRAB_USER_COMPLEMENTARITY.csv"].append(errors)
            outputs["GRAB_USER_INTERACTION.csv"].append(interaction)
            for arm in NAMES:
                actual = computed[arm]["score"]
                saved = PARENT["scores"][arm][phase]
                if scope == "pooled":
                    for metric in ("accuracy", "macro_f1", "log_loss", "brier"):
                        if not np.isclose(actual[metric], saved[metric], rtol=0, atol=1e-10):
                            raise AssertionError(f"saved pooled metric changed: {phase}/{arm}/{metric}")
                    if actual["trials"] != saved["trials"]:
                        raise AssertionError("saved pooled record count changed")
                elif (actual["trials"] != 28 or not np.isclose(
                        actual["macro_f1"], saved["per_subject_macro_f1"][subject],
                        rtol=0, atol=1e-10)):
                    raise AssertionError(f"saved subject metric changed: {phase}/{arm}/{subject}")
                checked += 1
    expected = {"GRAB_USER_SCREEN.csv": 24, "GRAB_USER_CONDITIONAL.csv": 12,
                "GRAB_USER_COMPLEMENTARITY.csv": 6, "GRAB_USER_INTERACTION.csv": 6}
    if {name: len(rows) for name, rows in outputs.items()} != expected or checked != 24:
        raise AssertionError("unexpected user-axis matched cell count")
    for name, rows in outputs.items():
        payload = csv_text(rows).encode("utf-8")
        target = ROOT / name
        if verify:
            if target.read_bytes() != payload:
                raise AssertionError(f"derived user-axis table changed: {name}")
        else:
            target.write_bytes(payload)
    audit = {"status": "ok", "analysis_protocol_sha256": sha(PROTOCOL_PATH),
             "parent_prediction_sha256": PROTOCOL["parent_prediction_sha256"],
             "parent_score_groups_replayed": checked, "matched_cells": 6,
             "rows": expected, "boundary": PROTOCOL["boundary"]}
    payload = json.dumps(audit, indent=2) + "\n"
    path = ROOT / "GRAB_USER_PAIRED_AUDIT.json"
    if verify:
        if path.read_text(encoding="utf-8") != payload:
            raise AssertionError("derived user-axis audit changed")
    else:
        path.write_text(payload, encoding="utf-8")
    print(json.dumps({"status": "ok", "verify": verify, "rows": expected}))
    return audit


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    build(parser.parse_args().verify)
