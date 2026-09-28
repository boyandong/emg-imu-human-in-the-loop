"""Matched exact-bank MANUS speed analysis from frozen external-Rest predictions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmarks.new_bank_v1.paired_analysis import analyze_group, csv_text
from emgimu.datasets.semg_manus import PATH_RE

ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "MANUS_REST_TRANSFER_PAIRED_PROTOCOL.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
NAMES = tuple(PROTOCOL["arms"])
CLASSES = [str(value) for value in PROTOCOL["classes"]]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(verify: bool = False) -> dict:
    for name, field in (("MANUS_REST_TRANSFER_PROTOCOL.json", "parent_protocol_sha256"),
                        ("MANUS_REST_TRANSFER_RESULTS.json", "parent_result_sha256"),
                        ("MANUS_REST_TRANSFER_PREDICTIONS.csv", "parent_prediction_sha256")):
        if sha(ROOT / name) != PROTOCOL[field]:
            raise AssertionError(f"frozen external-Rest parent changed: {name}")
    parent = json.loads((ROOT / "MANUS_REST_TRANSFER_RESULTS.json").read_text(encoding="utf-8"))
    with (ROOT / "MANUS_REST_TRANSFER_PREDICTIONS.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 864 or NAMES != ("F0v2", "F0v2+F2a", "F0v2+F3c", "F0v2+F2a+F3c"):
        raise AssertionError("unexpected MANUS external-Rest prediction count or arms")
    for row in rows:
        native = PATH_RE.fullmatch(row["trial_id"])
        session = parent["protocol"][f"{row['phase']}_session"]
        if (native is None or int(native["user"]) != int(row["subject"])
                or int(native["session"]) != session
                or native["speed"] != row["condition"]
                or native["gesture"] != parent["protocol"]["gestures"][int(row["label"])]) :
            raise AssertionError("native MANUS identity mismatch")
        p = np.asarray([float(row[f"p_{c}"]) for c in CLASSES])
        if np.any(p < 0) or not np.isclose(p.sum(), 1, rtol=0, atol=1e-10):
            raise AssertionError("invalid saved MANUS probability")
    outputs = {"MANUS_REST_TRANSFER_SCREEN.csv": [],
               "MANUS_REST_TRANSFER_CONDITIONAL.csv": [],
               "MANUS_REST_TRANSFER_COMPLEMENTARITY.csv": [],
               "MANUS_REST_TRANSFER_INTERACTION.csv": []}
    replayed = 0
    for phase in ("validation", "final"):
        phase_rows = [row for row in rows if row["phase"] == phase]
        users, speeds = parent["protocol"]["users"], parent["protocol"]["conditions"]
        groups = [("pooled", "ALL", "ALL")]
        groups.extend(("subject", str(user), "ALL") for user in users)
        groups.extend(("condition", "ALL", speed) for speed in speeds)
        groups.extend(("subject_condition", str(user), speed) for user in users for speed in speeds)
        for scope, subject, condition in groups:
            by_arm = {arm: [row for row in phase_rows if row["arm"] == arm and
                            (subject == "ALL" or row["subject"] == subject) and
                            (condition == "ALL" or row["condition"] == condition)]
                      for arm in NAMES}
            screen, conditional, errors, interaction, calculated = analyze_group(
                "manus_rest_transfer", phase, scope, subject, condition,
                by_arm, CLASSES, NAMES)
            outputs["MANUS_REST_TRANSFER_SCREEN.csv"].extend(screen)
            outputs["MANUS_REST_TRANSFER_CONDITIONAL.csv"].extend(conditional)
            outputs["MANUS_REST_TRANSFER_COMPLEMENTARITY.csv"].append(errors)
            outputs["MANUS_REST_TRANSFER_INTERACTION.csv"].append(interaction)
            for arm in NAMES:
                actual = calculated[arm]["score"]
                parent_score = parent["scores"][phase][arm]
                expected = (parent_score["pooled"] if scope == "pooled" else
                            parent_score["by_user"][subject] if scope == "subject" else
                            parent_score["by_speed"][condition] if scope == "condition" else None)
                if expected is not None:
                    for metric in ("trials", "accuracy", "macro_f1", "log_loss", "brier"):
                        if not np.isclose(actual[metric], expected[metric], rtol=0, atol=1e-10):
                            raise AssertionError("saved MANUS speed score changed")
                    replayed += 1
                elif actual["trials"] != 6:
                    raise AssertionError("user-by-speed native trial count changed")
    expected_rows = {"MANUS_REST_TRANSFER_SCREEN.csv": 224,
                     "MANUS_REST_TRANSFER_CONDITIONAL.csv": 112,
                     "MANUS_REST_TRANSFER_COMPLEMENTARITY.csv": 56,
                     "MANUS_REST_TRANSFER_INTERACTION.csv": 56}
    if {name: len(items) for name, items in outputs.items()} != expected_rows or replayed != 80:
        raise AssertionError("unexpected MANUS external-Rest matched cells")
    for name, items in outputs.items():
        payload = csv_text(items).encode("utf-8")
        target = ROOT / name
        if verify:
            if target.read_bytes() != payload:
                raise AssertionError(f"derived MANUS table changed: {name}")
        else:
            target.write_bytes(payload)
    audit = {"status": "ok", "analysis_protocol_sha256": sha(PROTOCOL_PATH),
             "parent_prediction_sha256": PROTOCOL["parent_prediction_sha256"],
             "parent_score_groups_replayed": replayed, "matched_cells": 56,
             "rows": expected_rows, "boundary": PROTOCOL["scope"]}
    payload = json.dumps(audit, indent=2) + "\n"
    target = ROOT / "MANUS_REST_TRANSFER_PAIRED_AUDIT.json"
    if verify:
        if target.read_text(encoding="utf-8") != payload:
            raise AssertionError("derived MANUS audit changed")
    else:
        target.write_text(payload, encoding="utf-8")
    print(json.dumps({"status": "ok", "verify": verify, "rows": expected_rows}))
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    build(parser.parse_args().verify)
