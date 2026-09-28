"""Matched new-v2 posture analyses from frozen native-bout predictions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmarks.new_bank_v1.paired_analysis import analyze_group, csv_text

ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "ROAM_POSTURE_PAIRED_PROTOCOL.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
NAMES = tuple(PROTOCOL["arms"])
CLASSES = [str(value) for value in PROTOCOL["classes"]]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(verify: bool = False) -> dict:
    for name, key in (("ROAM_POSTURE_PROTOCOL.json", "parent_protocol_sha256"),
                      ("ROAM_POSTURE_RESULTS.json", "parent_result_sha256"),
                      ("ROAM_POSTURE_PREDICTIONS.csv", "parent_prediction_sha256")):
        if sha(ROOT / name) != PROTOCOL[key]:
            raise AssertionError(f"frozen ROAM parent changed: {name}")
    parent = json.loads((ROOT / "ROAM_POSTURE_RESULTS.json").read_text(encoding="utf-8"))
    with (ROOT / "ROAM_POSTURE_PREDICTIONS.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1440 or NAMES != ("F0v2", "F0v2+F2a", "F0v2+F3c", "F0v2+F2a+F3c"):
        raise AssertionError("unexpected ROAM prediction count or arms")
    if {r["label"] for r in rows} != set(CLASSES):
        raise AssertionError("unexpected ROAM label set")
    for row in rows:
        subject, posture, bout = row["trial_id"].split("_", 2)
        if (subject != f"s{row['subject']}" or posture != "static"
                or not bout.startswith(f"{row['condition']}_bout")
                or row["condition"] not in parent["protocol"]["target_postures"]):
            raise AssertionError("native bout identity does not match metadata")
        p = np.asarray([float(row[f"p_{c}"]) for c in CLASSES])
        if np.any(p < 0) or not np.isclose(p.sum(), 1, rtol=0, atol=1e-10):
            raise AssertionError("invalid saved class probability")
    outputs = {"ROAM_POSTURE_SCREEN.csv": [], "ROAM_POSTURE_CONDITIONAL.csv": [],
               "ROAM_POSTURE_COMPLEMENTARITY.csv": [], "ROAM_POSTURE_INTERACTION.csv": []}
    replayed = 0
    for phase in ("validation", "final"):
        users = parent["protocol"][f"{phase}_subjects"]
        postures = parent["protocol"]["target_postures"]
        phase_rows = [row for row in rows if row["phase"] == phase]
        groups = [("pooled", "ALL", "ALL")]
        groups.extend(("condition", "ALL", posture) for posture in postures)
        groups.extend(("subject", str(user), "ALL") for user in users)
        groups.extend(("subject_condition", str(user), posture)
                      for user in users for posture in postures)
        for scope, subject, posture in groups:
            by_arm = {arm: [row for row in phase_rows if row["arm"] == arm and
                            (subject == "ALL" or row["subject"] == subject) and
                            (posture == "ALL" or row["condition"] == posture)]
                      for arm in NAMES}
            screen, conditional, errors, interaction, computed = analyze_group(
                "roam_posture", phase, scope, subject, posture, by_arm, CLASSES, NAMES)
            for item in [*screen, *conditional, errors, interaction]:
                item["evaluation_unit"] = "native_label_bout"
            errors["comparison_type"] = "matched_new_v2_single_additions"
            outputs["ROAM_POSTURE_SCREEN.csv"].extend(screen)
            outputs["ROAM_POSTURE_CONDITIONAL.csv"].extend(conditional)
            outputs["ROAM_POSTURE_COMPLEMENTARITY.csv"].append(errors)
            outputs["ROAM_POSTURE_INTERACTION.csv"].append(interaction)
            for arm in NAMES:
                actual = computed[arm]["score"]
                saved = parent["scores"][arm][phase]
                expected = (saved if scope == "pooled" else
                            saved["by_posture"][posture] if scope == "condition" else None)
                if expected is not None:
                    expected_bouts = 180 if scope == "pooled" else 45
                    for metric in ("trials", "accuracy", "macro_f1", "log_loss", "brier"):
                        if not np.isclose(actual[metric],
                                          expected_bouts if metric == "trials" else expected[metric],
                                          rtol=0, atol=1e-10):
                            raise AssertionError(f"parent score changed: {phase}/{scope}/{arm}/{metric}")
                    replayed += 1
                if scope == "subject" and not np.isclose(
                        actual["macro_f1"], saved["per_subject_macro_f1"][subject],
                        rtol=0, atol=1e-10):
                    raise AssertionError("parent user macro-F1 changed")
    expected_rows = {"ROAM_POSTURE_SCREEN.csv": 240,
                     "ROAM_POSTURE_CONDITIONAL.csv": 120,
                     "ROAM_POSTURE_COMPLEMENTARITY.csv": 60,
                     "ROAM_POSTURE_INTERACTION.csv": 60}
    if {name: len(items) for name, items in outputs.items()} != expected_rows or replayed != 40:
        raise AssertionError("unexpected matched ROAM groups")
    for name, items in outputs.items():
        payload = csv_text(items).encode("utf-8")
        target = ROOT / name
        if verify:
            if target.read_bytes() != payload:
                raise AssertionError(f"derived ROAM table changed: {name}")
        else:
            target.write_bytes(payload)
    audit = {"status": "ok", "analysis_protocol_sha256": sha(PROTOCOL_PATH),
             "parent_prediction_sha256": PROTOCOL["parent_prediction_sha256"],
             "saved_score_groups_replayed": replayed, "matched_cells": 60,
             "rows": expected_rows, "boundary": PROTOCOL["boundary"]}
    payload = json.dumps(audit, indent=2) + "\n"
    target = ROOT / "ROAM_POSTURE_PAIRED_AUDIT.json"
    if verify:
        if target.read_text(encoding="utf-8") != payload:
            raise AssertionError("derived ROAM audit changed")
    else:
        target.write_text(payload, encoding="utf-8")
    print(json.dumps({"status": "ok", "verify": verify, "rows": expected_rows}))
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    build(parser.parse_args().verify)
