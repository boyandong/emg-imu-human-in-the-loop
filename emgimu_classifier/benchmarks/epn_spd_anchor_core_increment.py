"""Matched EPN replay of SPD personal anchors added to the frozen Core.

The Core and anchor predictions already exist.  This script never fits a model;
the Core/anchor mixture is fixed at 1/2 before evaluating final users.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from emgimu.datasets.epn612 import GESTURES
from emgimu.feature_bank.epn_study import _metrics


PHASE_USERS = {"validation": (16, 17, 18), "final": (19, 20, 21)}
BUDGETS = (1, 2, 5)
ARMS = ("F0", "Core", "SPD_Anchor", "Core_plus_SPD_Anchor")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probabilities(values: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if (matrix.ndim != 2 or matrix.shape[1] != len(GESTURES)
            or not np.isfinite(matrix).all() or np.any(matrix < 0)
            or not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-8)):
        raise ValueError(f"invalid {name} probability matrix")
    return matrix


def matched_arrays(core: dict, anchor_rows: list[dict], selection_rows: list[dict],
                   budget: int, phase: str = "final"):
    if budget not in BUDGETS:
        raise ValueError("unsupported budget")
    if phase not in PHASE_USERS:
        raise ValueError("unsupported phase")
    phase_users = PHASE_USERS[phase]
    labels = np.asarray(core["labels"], dtype=int)
    users = np.asarray(core["users"], dtype=int)
    trials = np.asarray(core["trials"], dtype=str)
    base = probabilities(core["baseline"], "F0")
    full = probabilities(core["full"], "Core")
    if not (len(labels) == len(users) == len(trials) == len(base) == len(full)):
        raise ValueError("Core arrays have unequal lengths")
    keys = [(int(user), str(trial)) for user, trial in zip(users, trials)]
    if len(set(keys)) != len(keys) or set(users) != set(phase_users):
        raise ValueError("Core user/trial keys are duplicated or unexpected")

    selected = set()
    for row in selection_rows:
        if row["phase"] != phase or int(row["subject"]) not in phase_users:
            raise ValueError("selection phase/user mismatch")
        if int(row["shots_per_class"]) == budget:
            key = (int(row["subject"]), row["trial_id"])
            if key in selected:
                raise ValueError("duplicate calibration trial")
            selected.add(key)
    if len(selected) != len(phase_users) * len(GESTURES) * budget or not selected <= set(keys):
        raise ValueError("calibration selection does not match frozen Core trials")

    anchor = {}
    for row in anchor_rows:
        if row["phase"] != phase or int(row["subject"]) not in phase_users:
            raise ValueError("anchor phase/user mismatch")
        if int(row["shots_per_class"]) == budget:
            key = (int(row["subject"]), row["trial_id"])
            if key in anchor:
                raise ValueError("duplicate anchor prediction")
            anchor[key] = row
    expected = set(keys) - selected
    if set(anchor) != expected:
        raise ValueError("anchor predictions do not equal noncalibration Core trials")
    indexes = np.asarray([index for index, key in enumerate(keys) if key in expected], dtype=int)
    target_keys = [keys[index] for index in indexes]
    if any(int(anchor[key]["true_label"]) != int(labels[index])
           for index, key in zip(indexes, target_keys)):
        raise ValueError("Core and anchor trial labels disagree")
    anchor_probability = probabilities(np.asarray([
        [float(anchor[key][f"p_{gesture}"]) for gesture in GESTURES]
        for key in target_keys]), "SPD Anchor")
    return labels[indexes], users[indexes], {
        "F0": base[indexes], "Core": full[indexes],
        "SPD_Anchor": anchor_probability,
        "Core_plus_SPD_Anchor": (full[indexes] + anchor_probability) / 2,
    }


def run(phase: str, core_path: Path, core_audit_path: Path, anchor_path: Path,
        anchor_audit_path: Path, selection_path: Path, output_path: Path) -> dict:
    if phase not in PHASE_USERS:
        raise ValueError("unsupported phase")
    core_audit = json.loads(core_audit_path.read_text(encoding="utf-8"))
    anchor_audit = json.loads(anchor_audit_path.read_text(encoding="utf-8"))
    core_hash = (core_audit["source_sha256"] if phase == "validation" else core_audit["output_sha256"])["heldout_predictions.npz"]
    if (sha(core_path) != core_hash
            or sha(anchor_path) != anchor_audit["phases"][phase]["output_sha256"]["trial_predictions.csv"]
            or sha(selection_path) != anchor_audit["phases"][phase]["selection_file_sha256"]):
        raise ValueError("frozen input hash mismatch")
    with np.load(core_path, allow_pickle=False) as source:
        core = {name: source[name].copy() for name in ("labels", "users", "trials", "baseline", "full")}
    with anchor_path.open(newline="", encoding="utf-8") as handle:
        anchor_rows = list(csv.DictReader(handle))
    with selection_path.open(newline="", encoding="utf-8") as handle:
        selection_rows = list(csv.DictReader(handle))

    rows = []
    for budget in BUDGETS:
        truth, users, arms = matched_arrays(core, anchor_rows, selection_rows, budget, phase)
        for subject in ("ALL", *PHASE_USERS[phase]):
            mask = np.ones(len(truth), dtype=bool) if subject == "ALL" else users == subject
            for arm in ARMS:
                rows.append({"phase": phase, "subject": subject, "shots_per_class": budget,
                             "model": arm, "evaluation_trials": int(mask.sum()),
                             **_metrics(truth[mask], arms[arm][mask], np.ones(mask.sum()))})
        print(f"{phase} cal{budget}: {len(truth)} matched native trials", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(output_path)
    result = {
        "completion_proven": False,
        "protocol": "source-frozen EPN F0+Ring+CSP+IMU Core; frozen source-SPD personal trial prototypes; exact saved calibration/evaluation splits; equal probability mixture with no final fitting or weight tuning",
        "phase": phase, "users": list(PHASE_USERS[phase]), "budgets": list(BUDGETS), "arms": list(ARMS),
        "script_sha256": sha(Path(__file__)),
        "inputs_sha256": {"core_predictions": sha(core_path), "core_audit": sha(core_audit_path),
                          "spd_anchor_predictions": sha(anchor_path),
                          "spd_anchor_audit": sha(anchor_audit_path), "selection": sha(selection_path)},
        "scores": rows,
        "boundary": "Matched native EPN trial replay of a fixed late-fusion increment. Validation users selected the Core and final users were already examined in other project analyses, so this is exploratory rather than a pristine confirmatory test. It is not a newly fitted concatenated-feature Core, calibrated fusion weight, historical RLCS, or own-device accuracy.",
    }
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=tuple(PHASE_USERS))
    for name in ("core", "core_audit", "anchor", "anchor_audit", "selection", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.phase, args.core, args.core_audit, args.anchor, args.anchor_audit, args.selection, args.output)
