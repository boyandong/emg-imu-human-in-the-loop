"""Matched saved-trial error complementarity and two-family interaction."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from family_screen import ROOT, SPECS, _score

PROTOCOL_PATH = ROOT / "PAIR_PROTOCOL.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
ARMS = tuple(PROTOCOL["arms"])
PREDICTION_HASHES = json.loads((ROOT / "FAMILY_SCREEN_PROTOCOL.json").read_text(encoding="utf-8"))["prediction_sha256"]


def read_axis(axis: str, spec: dict) -> list[dict]:
    path = ROOT / spec["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != PREDICTION_HASHES[axis]:
        raise AssertionError(f"frozen predictions changed: {axis}")
    with path.open(newline="", encoding="utf-8") as stream:
        raw = list(csv.DictReader(stream))
    rows = []
    for item in raw:
        arm = item["arm"].replace("F0+", "F0v2+") if axis == "day" else item["arm"]
        if axis == "day" and arm == "F0":
            arm = "F0v2"
        if arm not in ARMS:
            continue
        phase = item[spec["phase"]]
        if phase not in ("validation", "final"):
            raise AssertionError("unexpected phase")
        rows.append({"phase": phase, "arm": arm, "trial_id": item["trial_id"],
                     "subject": item[spec["subject"]] if spec["subject"] else "Song",
                     "condition": item[spec["condition"]] if spec["condition"] else
                     ("session_2" if phase == "validation" else "session_3"),
                     "label": item[spec["label"]],
                     **{f"p_{c}": item[f"p_{c}"] for c in spec["classes"]}})
    expected = {"force": 4704, "wearing": 960, "day": 1792, "Song_same_day": 1136}[axis]
    if len(rows) != expected:
        raise AssertionError(f"unexpected {axis} trial-arm count")
    return rows


def analyze_cell(axis: str, spec: dict, phase: str, scope: str, subject: str,
                 condition: str, by_arm: dict[str, list[dict]]) -> tuple[list[dict], dict]:
    aligned = {}
    for arm, rows in by_arm.items():
        index = {}
        for row in rows:
            key = (row["subject"], row["condition"], row["trial_id"])
            if key in index:
                raise AssertionError(f"duplicate native trial: {axis}/{phase}/{arm}/{key}")
            index[key] = row
        aligned[arm] = index
    keys = sorted(aligned[ARMS[0]])
    if not keys or any(set(index) != set(keys) for index in aligned.values()):
        raise AssertionError(f"unmatched native trials: {axis}/{phase}/{scope}")
    labels = np.asarray([aligned[ARMS[0]][key]["label"] for key in keys])
    calculated = {}
    for arm in ARMS:
        ordered = [aligned[arm][key] for key in keys]
        if not np.array_equal(labels, [row["label"] for row in ordered]):
            raise AssertionError("matched native trial has conflicting labels")
        probs = np.asarray([[float(row[f"p_{c}"]) for c in spec["classes"]] for row in ordered])
        pred = np.asarray(spec["classes"])[probs.argmax(axis=1)]
        calculated[arm] = {"pred": pred, "correct": pred == labels,
                           "score": _score(ordered, spec["classes"])}
    common = {"dataset": spec["dataset"], "axis": axis, "phase": phase,
              "scope": scope, "subject": subject, "condition": condition,
              "calibration_budget": 0, "evaluation_unit": "whole_native_trial",
              "evaluation_trials": len(keys)}
    pairs = []
    for arm_a, arm_b in itertools.combinations(ARMS, 2):
        a, b = calculated[arm_a], calculated[arm_b]
        ea, eb = (~a["correct"]).astype(int), (~b["correct"]).astype(int)
        corr = (float(np.corrcoef(ea, eb)[0, 1]) if ea.std() and eb.std() else "N/A")
        acbw = int(np.sum(a["correct"] & ~b["correct"]))
        awbc = int(np.sum(~a["correct"] & b["correct"]))
        both_correct = int(np.sum(a["correct"] & b["correct"]))
        both_wrong = int(np.sum(~a["correct"] & ~b["correct"]))
        if acbw + awbc + both_correct + both_wrong != len(keys):
            raise AssertionError("error contingency does not sum to native trials")
        pairs.append({**common, "family_a": arm_a, "family_b": arm_b,
                      "comparison_type": "matched_saved_prediction_pair",
                      "error_correlation": corr,
                      "prediction_disagreement_rate": float(np.mean(a["pred"] != b["pred"])),
                      "a_correct_b_wrong": acbw, "a_wrong_b_correct": awbc,
                      "both_correct": both_correct, "both_wrong": both_wrong})
    b, a, c, joint = (calculated[arm]["score"] for arm in ARMS)
    interaction = {**common, "base_arm": ARMS[0], "family_a_arm": ARMS[1],
                   "family_b_arm": ARMS[2], "joint_arm": ARMS[3],
                   "interaction_neg_log_loss": -joint["log_loss"] + a["log_loss"] +
                   c["log_loss"] - b["log_loss"],
                   "interaction_macro_f1": joint["macro_f1"] - a["macro_f1"] -
                   c["macro_f1"] + b["macro_f1"],
                   "joint_minus_core_macro_f1": joint["macro_f1"] - b["macro_f1"],
                   "joint_minus_core_neg_log_loss": b["log_loss"] - joint["log_loss"],
                   "base_macro_f1": b["macro_f1"], "joint_macro_f1": joint["macro_f1"]}
    return pairs, interaction


def render_csv(rows: list[dict]) -> str:
    import io
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def build(verify: bool = False) -> dict:
    parent = ROOT / "FAMILY_SCREEN_PROTOCOL.json"
    if hashlib.sha256(parent.read_bytes()).hexdigest() != PROTOCOL["parent_family_screen_protocol_sha256"]:
        raise AssertionError("parent protocol changed")
    pair_rows, interaction_rows = [], []
    cells_by_axis = {}
    for axis, spec in SPECS.items():
        rows = read_axis(axis, spec)
        before = len(interaction_rows)
        for phase in ("validation", "final"):
            phase_rows = [row for row in rows if row["phase"] == phase]
            base = [row for row in phase_rows if row["arm"] == ARMS[0]]
            aggregate = (base[0]["condition"] if axis in ("day", "Song_same_day")
                         else spec["aggregate"])
            groups = [("pooled", "Song" if axis == "Song_same_day" else "ALL", aggregate)]
            if axis != "Song_same_day":
                groups += [("subject", subject, aggregate) for subject in
                           sorted({row["subject"] for row in base}, key=int)]
            if axis in ("force", "wearing"):
                groups += [("condition", "ALL", condition) for condition in
                           sorted({row["condition"] for row in base})]
            for scope, subject, condition in groups:
                selected = {}
                for arm in ARMS:
                    group = [row for row in phase_rows if row["arm"] == arm and
                             (scope != "subject" or row["subject"] == subject) and
                             (scope != "condition" or row["condition"] == condition)]
                    selected[arm] = group
                pairs, interaction = analyze_cell(axis, spec, phase, scope, subject,
                                                  condition, selected)
                pair_rows.extend(pairs)
                interaction_rows.append(interaction)
        cells_by_axis[axis] = len(interaction_rows) - before
    if cells_by_axis != {"force": 28, "wearing": 16, "day": 18, "Song_same_day": 2}:
        raise AssertionError(f"unexpected paired cells: {cells_by_axis}")
    outputs = {"PAIR_COMPLEMENTARITY.csv": pair_rows,
               "F2A_F3C_INTERACTION.csv": interaction_rows}
    if len(pair_rows) != 384 or len(interaction_rows) != 64:
        raise AssertionError("unexpected paired-analysis table sizes")
    for filename, rows in outputs.items():
        content = render_csv(rows)
        path = ROOT / filename
        if verify:
            if path.read_bytes() != content.encode("utf-8"):
                raise AssertionError(f"derived table changed: {filename}")
        else:
            path.write_bytes(content.encode("utf-8"))
    audit = {"status": "ok", "protocol_sha256": hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
             "parent_protocol_sha256": hashlib.sha256(parent.read_bytes()).hexdigest(),
             "source_predictions_sha256": PREDICTION_HASHES, "matched_cells_by_axis": cells_by_axis,
             "rows": {name: len(rows) for name, rows in outputs.items()},
             "interpretation": PROTOCOL["interpretation"], "boundary": PROTOCOL["boundary"]}
    path = ROOT / "PAIR_AUDIT.json"
    content = json.dumps(audit, indent=2) + "\n"
    if verify:
        if path.read_text(encoding="utf-8") != content:
            raise AssertionError("pair audit changed")
    else:
        path.write_text(content, encoding="utf-8")
    print(json.dumps({"status": "ok", "verify": verify, "cells": cells_by_axis,
                      "rows": audit["rows"]}))
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    build(parser.parse_args().verify)
