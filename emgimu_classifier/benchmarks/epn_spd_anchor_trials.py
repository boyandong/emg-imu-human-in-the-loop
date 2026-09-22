"""Frozen-source EPN SPD personal prototypes on complete native trials.

This is an offline candidate study, not historical RLCS or device validation.
Selection CSVs are reused byte-for-byte from the corrected EPN anchor study.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from emgimu.datasets.epn612 import GESTURES, load_epn612_windows
from emgimu.feature_bank import SpdTangentFamily, SpdTangentPersonalAnchor
from emgimu.feature_bank.epn_study import _metrics


SEED = 20260915
PHASE_USERS = {"validation": (16, 17, 18), "final": (19, 20, 21)}
BUDGETS = (1, 2, 5)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _selection(path: Path, phase: str) -> dict[tuple[int, int], set[str]]:
    chosen = {(user, budget): set() for user in PHASE_USERS[phase] for budget in BUDGETS}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["phase"] != phase:
                raise ValueError("selection phase mismatch")
            key = (int(row["subject"]), int(row["shots_per_class"]))
            if key not in chosen or row["trial_id"] in chosen[key]:
                raise ValueError("unexpected or duplicate calibration trial")
            chosen[key].add(row["trial_id"])
    if any(len(chosen[key]) != 6 * key[1] for key in chosen):
        raise ValueError("selection does not contain the specified six-class budget")
    return chosen


def run(archive: Path, selection_root: Path, output_root: Path) -> None:
    print("[1/3] fitting source SPD reference: users 1-15", flush=True)
    source = load_epn612_windows(archive, users=range(1, 16))
    family = SpdTangentFamily().fit(source.batch)
    frozen = SpdTangentPersonalAnchor(family)
    reference_sha = frozen.source_reference_sha256_
    source_trial_count = len(np.unique(source.trials))
    for phase in PHASE_USERS:
        users = PHASE_USERS[phase]
        selection_file = (selection_root /
            f"feature_bank_epn612_anchor_temperature_{phase}_20260916_v2" /
            "calibration_trial_ids.csv")
        selected = _selection(selection_file, phase)
        print(f"[2/3] {phase}: loading native users {users}", flush=True)
        data = load_epn612_windows(archive, users=users)
        output = output_root / f"feature_bank_epn_spd_anchor_{phase}_20260922"
        output.mkdir(parents=True, exist_ok=True)
        scores, predictions = [], []
        for user in users:
            personal = data.take(np.flatnonzero(data.users == user))
            all_trials = set(personal.trials.tolist())
            trial_truth = {}
            for trial in all_trials:
                values = np.unique(personal.labels[personal.trials == trial])
                if len(values) != 1:
                    raise ValueError(f"mixed labels in native trial {trial}")
                trial_truth[trial] = int(values[0])
            for budget in BUDGETS:
                calibration_ids = selected[(user, budget)]
                if not calibration_ids <= all_trials:
                    raise ValueError("selected calibration trial absent from native data")
                if any(sum(trial_truth[t] == cls for t in calibration_ids) != budget for cls in range(6)):
                    raise ValueError("selected calibration labels do not match budget")
                cal = personal.take(np.flatnonzero(np.isin(personal.trials, list(calibration_ids))))
                evaluation = personal.take(np.flatnonzero(~np.isin(personal.trials, list(calibration_ids))))
                if set(cal.trials) & set(evaluation.trials):
                    raise ValueError("calibration and evaluation trials overlap")
                anchor = SpdTangentPersonalAnchor(family).fit_trials(cal.batch, cal.labels, cal.trials)
                if anchor.source_reference_sha256_ != reference_sha or anchor.anchor_.classes_.tolist() != list(range(6)):
                    raise ValueError("source reference or six-class prototype mismatch")
                ids, coordinates = anchor.transform_trials(evaluation.batch, evaluation.trials)
                distance = coordinates[:, :6].astype(np.float64)
                logits = -distance / anchor.anchor_.similarity_scale_
                logits -= logits.max(axis=1, keepdims=True)
                exp = np.exp(logits)
                probability = exp / exp.sum(axis=1, keepdims=True)
                truth = np.array([trial_truth[trial] for trial in ids], dtype=int)
                if len(ids) != len(all_trials) - len(calibration_ids):
                    raise ValueError("trial-level evaluation count mismatch")
                metrics = _metrics(truth, probability, np.ones(len(ids)))
                scores.append({"phase": phase, "subject": user, "shots_per_class": budget,
                               "calibration_trials": len(calibration_ids), "evaluation_trials": len(ids),
                               "source_reference_sha256": reference_sha, **metrics})
                for trial, label, prob in zip(ids, truth, probability):
                    predictions.append({"phase": phase, "subject": user, "shots_per_class": budget,
                                        "trial_id": str(trial), "true_label": int(label),
                                        "predicted_label": int(np.argmax(prob)),
                                        **{f"p_{name}": float(prob[index]) for index, name in enumerate(GESTURES)}})
                print(f"[3/3] {phase} user{user} cal{budget}: {len(ids)} trials, "
                      f"ACC={metrics['accuracy']:.3f}, F1={metrics['macro_f1']:.3f}", flush=True)
        pooled = []
        for budget in BUDGETS:
            rows = [row for row in predictions if row["shots_per_class"] == budget]
            truth = np.asarray([row["true_label"] for row in rows], dtype=int)
            probability = np.asarray([[row[f"p_{name}"] for name in GESTURES] for row in rows])
            pooled.append({"phase": phase, "shots_per_class": budget,
                           "subjects": len(users), "evaluation_trials": len(rows),
                           **_metrics(truth, probability, np.ones(len(rows)))})
        _write_csv(output / "subject_scores.csv", scores)
        _write_csv(output / "pooled_scores.csv", pooled)
        _write_csv(output / "trial_predictions.csv", predictions)
        manifest = {
            "phase": phase, "users": users, "source_users": list(range(1, 16)),
            "source_trials": source_trial_count, "source_reference_sha256": reference_sha,
            "selection_file_sha256": _sha(selection_file), "selection_file": str(selection_file),
            "archive": str(archive), "seed_of_reused_selection": SEED,
            "protocol": "frozen source SPD tangent; one mean per complete native trial; equal trial weights; own-user class prototypes only",
            "evaluation": "all target-user labeled trainingJSON trials outside selected calibration trials",
            "zero_shot": "N/A: personal class prototypes require calibration",
            "boundaries": ["offline EPN candidate", "not historical RLCS", "not multi-family Core", "not live device accuracy"],
            "source_sha256": {"spd_anchor.py": _sha(Path(__file__).resolve().parents[1] / "src/emgimu/feature_bank/spd_anchor.py"),
                              "epn_spd_anchor_trials.py": _sha(Path(__file__))},
            "outputs_sha256": {name: _sha(output / name) for name in
                               ("subject_scores.csv", "pooled_scores.csv", "trial_predictions.csv")},
        }
        (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.archive, arguments.selection_root, arguments.output_root)
