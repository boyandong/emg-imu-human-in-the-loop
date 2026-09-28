"""Paired F2a/F3c increments, error complementarity and interaction on frozen trials."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, log_loss

ROOT = Path(__file__).resolve().parent
PREDICTIONS = ROOT / "FORCE_TRIAL_PREDICTIONS.csv"
CLASSES = np.arange(7)
BASE = "F0v2"
A = "F0v2+F2a"
B = "F0v2+F3c"
JOINT = "F0v2+F2a+F3c"
ARMS = (BASE, A, B, JOINT)


def _csv_bytes(rows: list[dict]) -> bytes:
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return text.getvalue().encode("utf-8")


def _scores(entries: list[dict], arm: str) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    y = np.asarray([int(item[arm]["label"]) for item in entries])
    p = np.asarray([[float(item[arm][f"p_{c}"]) for c in CLASSES] for item in entries])
    pred = p.argmax(axis=1)
    f1 = float(f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0))
    ll = float(log_loss(y, p, labels=CLASSES))
    brier = float(np.mean(np.sum((p - np.eye(7)[y]) ** 2, axis=1)))
    return f1, ll, brier, pred, y


def build(verify: bool = False) -> dict:
    raw = PREDICTIONS.read_bytes()
    with PREDICTIONS.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    groups: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        key = (row["phase"], row["trial_id"])
        if row["arm"] in groups[key]:
            raise AssertionError(f"duplicate arm-trial prediction: {key}, {row['arm']}")
        groups[key][row["arm"]] = row
    if (len(rows) != 5880 or len(groups) != 1176 or
            any(set(arms) != set(ARMS) | {"F0"} for arms in groups.values())):
        raise AssertionError("frozen force arms do not share exactly 1,176 native trials")
    for arms in groups.values():
        if len({(r["label"], r["subject"], r["condition"]) for r in arms.values()}) != 1:
            raise AssertionError("paired trial labels or conditions differ")
    cells = []
    for phase in ("validation", "final"):
        phase_entries = [arms for (cell_phase, _), arms in groups.items() if cell_phase == phase]
        for scope, value in [("pooled", "ALL"),
                             *[("subject", subject) for subject in sorted({r[BASE]["subject"] for r in phase_entries})],
                             *[("condition", condition) for condition in sorted({r[BASE]["condition"] for r in phase_entries})]]:
            entries = [r for r in phase_entries if scope == "pooled" or
                       r[BASE]["subject" if scope == "subject" else "condition"] == value]
            if not entries:
                raise AssertionError("empty paired cell")
            cells.append((phase, scope, value, entries))
    if len(cells) != 28:
        raise AssertionError("expected pooled, subject and condition cells in two phases")
    increments, complementarity, interactions = [], [], []
    for phase, scope, value, entries in cells:
        scores = {arm: _scores(entries, arm) for arm in ARMS}
        context = {"phase": phase, "scope": scope, "cell": value, "trials": len(entries),
                   "core_bank": BASE}
        for arm, added in ((A, "F2a"), (B, "F3c"), (JOINT, "F2a+F3c")):
            increments.append({**context, "added_family": added,
                               "delta_logloss_improvement": scores[BASE][1] - scores[arm][1],
                               "delta_macro_f1": scores[arm][0] - scores[BASE][0],
                               "delta_brier_improvement": scores[BASE][2] - scores[arm][2]})
        left_wrong = scores[A][3] != scores[A][4]
        right_wrong = scores[B][3] != scores[B][4]
        if np.std(left_wrong) and np.std(right_wrong):
            correlation: float | str = float(np.corrcoef(left_wrong.astype(float), right_wrong.astype(float))[0, 1])
        else:
            correlation = "N/A"
        complementarity.append({**context, "family_a": "F2a_added", "family_b": "F3c_added",
                                "error_correlation": correlation,
                                "prediction_disagreement_rate": float(np.mean(scores[A][3] != scores[B][3])),
                                "a_correct_b_wrong": float(np.mean(~left_wrong & right_wrong)),
                                "a_wrong_b_correct": float(np.mean(left_wrong & ~right_wrong))})
        interactions.append({**context, "family_a": "F2a", "family_b": "F3c",
                             "negative_logloss_interaction":
                             -scores[JOINT][1] + scores[A][1] + scores[B][1] - scores[BASE][1],
                             "macro_f1_interaction":
                             scores[JOINT][0] - scores[A][0] - scores[B][0] + scores[BASE][0],
                             "joint_minus_core_logloss_improvement": scores[BASE][1] - scores[JOINT][1],
                             "joint_minus_core_macro_f1": scores[JOINT][0] - scores[BASE][0]})
    outputs = {"FORCE_CONDITIONAL.csv": increments,
               "FORCE_COMPLEMENTARITY.csv": complementarity,
               "FORCE_INTERACTION.csv": interactions}
    for name, table in outputs.items():
        content = _csv_bytes(table)
        path = ROOT / name
        if verify:
            if path.read_bytes() != content:
                raise AssertionError(f"paired table changed: {name}")
        else:
            path.write_bytes(content)
    audit = {"status": "ok", "source_predictions_sha256": hashlib.sha256(raw).hexdigest(),
             "paired_native_trials": len(groups), "cells": len(cells),
             "rows": {name: len(table) for name, table in outputs.items()},
             "formula": "P=-log_loss; interaction=P(B+F2a+F3c)-P(B+F2a)-P(B+F3c)+P(B)",
             "boundary": "matched saved predictions only; final subjects descriptive, no selection/refit"}
    path = ROOT / "FORCE_PAIRED_AUDIT.json"
    content = (json.dumps(audit, indent=2) + "\n").encode("utf-8")
    if verify:
        if path.read_bytes() != content:
            raise AssertionError("paired audit changed")
    else:
        path.write_bytes(content)
    print(json.dumps({"status": "ok", "verification": verify, "rows": audit["rows"]}))
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    build(parser.parse_args().verify)
