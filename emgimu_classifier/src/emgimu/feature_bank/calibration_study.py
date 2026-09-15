from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from emgimu.datasets.libemg_force import ForceWindows, load_libemg_force_windows

from .calibration import PersonalAnchor
from .screening import CLASSES, FAMILY_FACTORIES, SEED, metrics


FROZEN_FAMILIES = ("F0", "F2b_CSP", "F1_X1H")
CONDITIONS = ("20P", "30P", "40P", "50P", "60P", "70P", "80P", "MVC", "Light", "Medium", "Hard")


def _extract(train: ForceWindows, target: ForceWindows) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    train_parts, target_parts, dimensions = [], [], {}
    for index, name in enumerate(FROZEN_FAMILIES, 1):
        print(f"[2/5] frozen feature {index}/{len(FROZEN_FAMILIES)} {name}", flush=True)
        family = FAMILY_FACTORIES[name]()
        train_part = family.fit_transform(train.batch, train.labels)
        target_part = family.transform(target.batch)
        dimensions[name] = train_part.shape[1]
        train_parts.append(train_part)
        target_parts.append(target_part)
    return np.concatenate(train_parts, axis=1), np.concatenate(target_parts, axis=1), dimensions


def _anchor_probability(calibration_x: np.ndarray, calibration_y: np.ndarray, evaluation_x: np.ndarray) -> np.ndarray:
    anchor = PersonalAnchor(metric="standardized_euclidean").fit(calibration_x, calibration_y)
    distances = anchor.transform(evaluation_x)[:, :len(CLASSES)]
    calibration_distances = anchor.transform(calibration_x)[:, :len(CLASSES)]
    scale = max(float(np.median(calibration_distances)), 1e-10)
    logits = -distances.astype(np.float64) / scale
    logits -= logits.max(axis=1, keepdims=True)
    probability = np.exp(logits)
    return probability / probability.sum(axis=1, keepdims=True)


def run(root: Path, output: Path, subjects: tuple[int, ...], phase: str) -> None:
    print(f"[1/5] loading {phase} target subjects and unopened population trials", flush=True)
    if phase == "validation" and set(subjects) != {7, 8}:
        raise ValueError("validation phase is frozen to subjects 7 and 8")
    if phase == "final" and set(subjects) != {9, 10}:
        raise ValueError("final phase is frozen to subjects 9 and 10")
    train = load_libemg_force_windows(root, subjects=range(1, 7), conditions=("Ramp",))
    target = load_libemg_force_windows(root, subjects=subjects, conditions=CONDITIONS)
    train_x, target_x, dimensions = _extract(train, target)

    scaler = StandardScaler().fit(train_x, sample_weight=train.sample_weight)
    population = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=SEED)
    population.fit(scaler.transform(train_x), train.labels, sample_weight=train.sample_weight)
    standardized_target = scaler.transform(target_x)
    population_probability = population.predict_proba(standardized_target)

    rows: list[dict] = []
    selections: list[dict] = []
    rng = np.random.default_rng(SEED)
    jobs = [(subject, condition, shots) for subject in subjects for condition in CONDITIONS for shots in (0, 1, 2, 5)]
    for job_index, (subject, condition, shots) in enumerate(jobs, 1):
        print(f"[3/5] calibration {job_index}/{len(jobs)} S{subject} {condition} {shots}-shot", flush=True)
        cell = (target.subjects == subject) & (target.conditions == condition)
        trial_ids_by_class = {
            int(label): np.unique(target.trials[cell & (target.labels == label)]).tolist() for label in CLASSES
        }
        available = min(len(ids) for ids in trial_ids_by_class.values())
        if shots > 0 and (shots >= available or shots == 5):
            rows.append({
                "phase": phase, "dataset": "libemg_contraction_intensity", "subject": subject,
                "condition": condition, "shots_per_class": shots, "feature_bank": "+".join(FROZEN_FAMILIES),
                "mode": "unsupported", "reason": f"only {available} trials/class; calibration must leave evaluation trials",
                "evaluation_trials": 0, "macro_f1": "", "accuracy": "", "log_loss": "", "brier": "", "ece": "",
                "per_class_f1_json": "",
            })
            continue

        calibration_trials: list[str] = []
        if shots:
            for label in CLASSES:
                candidates = np.asarray(trial_ids_by_class[int(label)], dtype=object)
                calibration_trials.extend(rng.permutation(candidates)[:shots].tolist())
        calibration_mask = cell & np.isin(target.trials, calibration_trials)
        evaluation_mask = cell & ~calibration_mask
        if shots:
            anchor_probability = _anchor_probability(
                standardized_target[calibration_mask], target.labels[calibration_mask], standardized_target[evaluation_mask]
            )
            personal_weight = shots / (2.0 + shots)
            probability = (1.0 - personal_weight) * population_probability[evaluation_mask] + personal_weight * anchor_probability
            probability /= probability.sum(axis=1, keepdims=True)
            mode = "population_plus_F7_anchor"
        else:
            probability = population_probability[evaluation_mask]
            mode = "population_zero_shot"
        values = metrics(target.labels[evaluation_mask], probability, target.sample_weight[evaluation_mask])
        rows.append({
            "phase": phase, "dataset": "libemg_contraction_intensity", "subject": subject,
            "condition": condition, "shots_per_class": shots, "feature_bank": "+".join(FROZEN_FAMILIES),
            "mode": mode, "reason": "", "evaluation_trials": len(np.unique(target.trials[evaluation_mask])), **values,
        })
        for trial in calibration_trials:
            selections.append({"phase": phase, "subject": subject, "condition": condition, "shots_per_class": shots, "trial_id": trial})

    print("[4/5] aggregating supported calibration budgets", flush=True)
    for shots in (0, 1, 2):
        masks, probabilities = [], []
        for subject in subjects:
            for condition in CONDITIONS:
                chosen = [row["trial_id"] for row in selections if row["subject"] == subject and row["condition"] == condition and row["shots_per_class"] == shots]
                if shots and not chosen:
                    # The condition does not have enough trials to calibrate and retain a test trial.
                    continue
                cell = (target.subjects == subject) & (target.conditions == condition) & ~np.isin(target.trials, chosen)
                masks.append(cell)
                if shots:
                    calibration = (target.subjects == subject) & (target.conditions == condition) & np.isin(target.trials, chosen)
                    anchor_probability = _anchor_probability(standardized_target[calibration], target.labels[calibration], standardized_target[cell])
                    weight = shots / (2.0 + shots)
                    blended = (1.0 - weight) * population_probability[cell] + weight * anchor_probability
                    probabilities.append(blended / blended.sum(axis=1, keepdims=True))
                else:
                    probabilities.append(population_probability[cell])
        combined_mask = np.logical_or.reduce(masks)
        probability = np.concatenate(probabilities)
        # Concatenation follows the same subject/condition order as this explicit index list.
        indices = np.concatenate([np.flatnonzero(mask) for mask in masks])
        values = metrics(target.labels[indices], probability, target.sample_weight[indices])
        rows.append({
            "phase": phase, "dataset": "libemg_contraction_intensity", "subject": "ALL", "condition": "ALL",
            "shots_per_class": shots, "feature_bank": "+".join(FROZEN_FAMILIES),
            "mode": "population_zero_shot" if shots == 0 else "population_plus_F7_anchor", "reason": "",
            "evaluation_trials": len(np.unique(target.trials[combined_mask])), **values,
        })
    rows.append({
        "phase": phase, "dataset": "libemg_contraction_intensity", "subject": "ALL", "condition": "ALL",
        "shots_per_class": 5, "feature_bank": "+".join(FROZEN_FAMILIES), "mode": "unsupported",
        "reason": "only 4 trials/class and at least one must remain for evaluation", "evaluation_trials": 0,
        "macro_f1": "", "accuracy": "", "log_loss": "", "brier": "", "ece": "", "per_class_f1_json": "",
    })

    print("[5/5] writing leakage-auditable calibration evidence", flush=True)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "calibration_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with (output / "calibration_trial_ids.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["phase", "subject", "condition", "shots_per_class", "trial_id"])
        writer.writeheader(); writer.writerows(selections)
    (output / "run_manifest.json").write_text(json.dumps({
        "phase": phase, "seed": SEED, "population_train_subjects": list(range(1, 7)),
        "population_train_condition": "Ramp", "target_subjects": list(subjects),
        "target_conditions": list(CONDITIONS), "families": list(FROZEN_FAMILIES), "dimensions": dimensions,
        "personal_anchor_shrinkage": "shots/(2+shots)", "test_trials_exclude_all_calibration_trial_ids": True,
        "protocol": "Force-ProductMode", "temperature_fit": "explicit calibration distances only",
    }, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "phase": phase, "rows": len(rows), "output": str(output)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage-safe personal calibration study")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--phase", choices=("validation", "final"), required=True)
    parser.add_argument("--subjects", type=int, nargs="+", required=True)
    args = parser.parse_args()
    run(args.dataset, args.output, tuple(args.subjects), args.phase)


if __name__ == "__main__":
    main()
