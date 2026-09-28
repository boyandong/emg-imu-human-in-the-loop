"""Matched new-v2 synthetic-quality analysis from frozen native-bout predictions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmarks.new_bank_v1.paired_analysis import analyze_group, csv_text

ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "ROAM_QUALITY_PAIRED_PROTOCOL.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
NAMES = tuple(PROTOCOL["arms"])
CLASSES = [str(value) for value in PROTOCOL["classes"]]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(verify: bool = False) -> dict:
    for name, key in (("ROAM_QUALITY_PROTOCOL.json", "parent_protocol_sha256"),
                      ("ROAM_QUALITY_RESULTS.json", "parent_result_sha256"),
                      ("ROAM_QUALITY_PREDICTIONS.csv", "parent_prediction_sha256")):
        if sha(ROOT / name) != PROTOCOL[key]:
            raise AssertionError(f"frozen ROAM quality parent changed: {name}")
    parent = json.loads((ROOT / "ROAM_QUALITY_RESULTS.json").read_text(encoding="utf-8"))
    with (ROOT / "ROAM_QUALITY_PREDICTIONS.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 5040 or NAMES != ("F0v2", "F0v2+F2a", "F0v2+F3c", "F0v2+F2a+F3c"):
        raise AssertionError("unexpected quality prediction count or arms")
    for row in rows:
        if (row["condition"] not in parent["protocol"]["conditions"]
                or row["label"] not in CLASSES
                or not row["trial_id"].startswith(f"s{row['subject']}_static_resting_bout")):
            raise AssertionError("quality bout identity or label mismatch")
        p = np.asarray([float(row[f"p_{c}"]) for c in CLASSES])
        if np.any(p < 0) or not np.isclose(p.sum(), 1, rtol=0, atol=1e-10):
            raise AssertionError("invalid saved class probability")
    outputs = {"ROAM_QUALITY_SCREEN.csv": [], "ROAM_QUALITY_CONDITIONAL.csv": [],
               "ROAM_QUALITY_COMPLEMENTARITY.csv": [], "ROAM_QUALITY_INTERACTION.csv": []}
    replayed = 0
    for phase in ("validation", "final"):
        users = parent["protocol"][f"{phase}_subjects"]
        for condition in parent["protocol"]["conditions"]:
            scene = [row for row in rows if row["phase"] == phase and row["condition"] == condition]
            groups = [("pooled", "ALL"), *(("subject", str(user)) for user in users)]
            for scope, subject in groups:
                by_arm = {arm: [row for row in scene if row["arm"] == arm and
                                (scope == "pooled" or row["subject"] == subject)]
                          for arm in NAMES}
                screen, conditional, errors, interaction, computed = analyze_group(
                    "roam_quality", phase, scope, subject, condition,
                    by_arm, CLASSES, NAMES)
                for item in [*screen, *conditional, errors, interaction]:
                    item["evaluation_unit"] = "native_label_bout"
                errors["comparison_type"] = "matched_new_v2_single_additions_synthetic_fault"
                outputs["ROAM_QUALITY_SCREEN.csv"].extend(screen)
                outputs["ROAM_QUALITY_CONDITIONAL.csv"].extend(conditional)
                outputs["ROAM_QUALITY_COMPLEMENTARITY.csv"].append(errors)
                outputs["ROAM_QUALITY_INTERACTION.csv"].append(interaction)
                for arm in NAMES:
                    actual = computed[arm]["score"]
                    saved = parent["scores"][arm][phase][condition]
                    if actual["trials"] != (45 if scope == "pooled" else 9):
                        raise AssertionError("native quality bout count changed")
                    if scope == "pooled":
                        for metric in ("accuracy", "macro_f1", "log_loss", "brier"):
                            if not np.isclose(actual[metric], saved[metric], rtol=0, atol=1e-10):
                                raise AssertionError("parent pooled quality score changed")
                        replayed += 1
                    elif not np.isclose(actual["macro_f1"],
                                        saved["per_subject_macro_f1"][subject],
                                        rtol=0, atol=1e-10):
                        raise AssertionError("parent subject quality score changed")
    expected = {"ROAM_QUALITY_SCREEN.csv": 672,
                "ROAM_QUALITY_CONDITIONAL.csv": 336,
                "ROAM_QUALITY_COMPLEMENTARITY.csv": 168,
                "ROAM_QUALITY_INTERACTION.csv": 168}
    if {name: len(items) for name, items in outputs.items()} != expected or replayed != 112:
        raise AssertionError("unexpected matched quality cells")
    for name, items in outputs.items():
        payload = csv_text(items).encode("utf-8")
        target = ROOT / name
        if verify:
            if target.read_bytes() != payload:
                raise AssertionError(f"derived quality table changed: {name}")
        else:
            target.write_bytes(payload)
    audit = {"status": "ok", "analysis_protocol_sha256": sha(PROTOCOL_PATH),
             "parent_prediction_sha256": PROTOCOL["parent_prediction_sha256"],
             "saved_pooled_score_groups_replayed": replayed, "matched_cells": 168,
             "rows": expected, "boundary": PROTOCOL["boundary"]}
    payload = json.dumps(audit, indent=2) + "\n"
    target = ROOT / "ROAM_QUALITY_PAIRED_AUDIT.json"
    if verify:
        if target.read_text(encoding="utf-8") != payload:
            raise AssertionError("derived quality audit changed")
    else:
        target.write_text(payload, encoding="utf-8")
    print(json.dumps({"status": "ok", "verify": verify, "rows": expected}))
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    build(parser.parse_args().verify)
