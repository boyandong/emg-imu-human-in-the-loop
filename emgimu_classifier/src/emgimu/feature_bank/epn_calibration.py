from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from emgimu.datasets.epn612 import GESTURES, load_epn612_windows

from .calibration import PersonalAnchor
from .epn_study import _metrics, aggregate_trials
from .screening import FAMILY_FACTORIES, SEED


CLASSES = np.arange(len(GESTURES))
FROZEN_FAMILIES = ("F0", "F3_Ring")


def _anchor_probability(calibration_x: np.ndarray, calibration_y: np.ndarray, evaluation_x: np.ndarray) -> np.ndarray:
    anchor = PersonalAnchor(metric="standardized_euclidean").fit(calibration_x, calibration_y)
    distance = anchor.transform(evaluation_x)[:, :len(CLASSES)]
    # A fixed calibration-only scale keeps each evaluation row independent of
    # other queries. Never estimate this temperature from evaluation distances.
    logits = -distance / anchor.similarity_scale_
    logits -= logits.max(axis=1, keepdims=True)
    probability = np.exp(logits)
    return probability / probability.sum(axis=1, keepdims=True)


def run(archive: Path, output: Path, subjects: tuple[int, ...], phase: str, method: str = "anchor") -> None:
    if phase not in ("validation", "final"):
        raise ValueError("phase must be validation or final")
    if method not in ("anchor", "finetune"):
        raise ValueError("method must be anchor or finetune")
    expected = {16, 17, 18} if phase == "validation" else {19, 20, 21}
    if set(subjects) != expected:
        raise ValueError(f"{phase} subjects are frozen to {sorted(expected)}")
    print(f"[1/5] loading EPN612 {phase} calibration split", flush=True)
    train = load_epn612_windows(archive, users=range(1, 16))
    target = load_epn612_windows(archive, users=subjects)
    train_parts, target_parts, dimensions, family_states = [], [], {}, {}
    train_labels = train_users = train_trials = train_weights = None
    target_labels = target_users = target_trials = target_weights = None
    for index, name in enumerate(FROZEN_FAMILIES, 1):
        print(f"[2/5] feature {index}/{len(FROZEN_FAMILIES)} {name}", flush=True)
        family = FAMILY_FACTORIES[name]()
        train_part = family.fit_transform(train.batch, train.labels)
        target_part = family.transform(target.batch)
        family_states[name] = family
        train_part, train_labels, train_users, train_trials, train_weights = aggregate_trials(train_part, train)
        target_part, target_labels, target_users, target_trials, target_weights = aggregate_trials(target_part, target)
        train_parts.append(train_part); target_parts.append(target_part); dimensions[name] = train_part.shape[1]
    train_x, target_x = np.concatenate(train_parts, axis=1), np.concatenate(target_parts, axis=1)
    scaler = StandardScaler().fit(train_x, sample_weight=train_weights)
    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=SEED)
    model.fit(scaler.transform(train_x), train_labels, sample_weight=train_weights)
    standardized = scaler.transform(target_x)
    population_probability = model.predict_proba(standardized)

    def calibrated_probability(calibration: np.ndarray, evaluation: np.ndarray, shots: int) -> np.ndarray:
        if shots == 0:
            return population_probability[evaluation]
        if method == "anchor":
            personal = _anchor_probability(standardized[calibration], target_labels[calibration], standardized[evaluation])
            weight = shots / (2.0 + shots)
            probability = (1.0 - weight) * population_probability[evaluation] + weight * personal
        elif method == "finetune":
            augmented_x = np.concatenate((scaler.transform(train_x), standardized[calibration]))
            augmented_y = np.concatenate((train_labels, target_labels[calibration]))
            augmented_weight = np.concatenate((train_weights, np.full(calibration.sum(), 15.0 / shots)))
            adapted = LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=SEED)
            adapted.fit(augmented_x, augmented_y, sample_weight=augmented_weight)
            probability = adapted.predict_proba(standardized[evaluation])
        else:
            raise ValueError(f"unknown calibration method: {method}")
        return probability / probability.sum(axis=1, keepdims=True)

    rows, selections, saved, split_ids = [], [], {}, {}
    rng = np.random.default_rng(SEED)
    for index, (subject, shots) in enumerate(((s, k) for s in subjects for k in (0, 1, 2, 5)), 1):
        print(f"[3/5] personal calibration {index}/{len(subjects) * 4} user{subject} {shots}-shot", flush=True)
        user_mask = target_users == subject
        chosen = []
        if shots:
            for label in CLASSES:
                trials = np.unique(target_trials[user_mask & (target_labels == label)])
                chosen.extend(rng.permutation(trials)[:shots].tolist())
        calibration = user_mask & np.isin(target_trials, chosen)
        evaluation = user_mask & ~calibration
        probability = calibrated_probability(calibration, evaluation, shots)
        key = f'user{subject}_shots{shots}'
        saved[f'{key}_probability'] = probability
        saved[f'{key}_labels'] = target_labels[evaluation]
        saved[f'{key}_weights'] = target_weights[evaluation]
        saved[f'{key}_trials'] = target_trials[evaluation]
        split_ids[key] = {'calibration': sorted(set(target_trials[calibration])),
                          'evaluation': sorted(set(target_trials[evaluation]))}
        rows.append({"phase": phase, "dataset": "emg_epn612", "subject": subject, "condition": "cross_user",
                     "shots_per_class": shots, "feature_bank": "+".join(FROZEN_FAMILIES),
                     "mode": "population_zero_shot" if shots == 0 else ("population_plus_F7_anchor" if method == "anchor" else "population_weighted_finetune"),
                     "evaluation_trials": len(np.unique(target_trials[evaluation])), **_metrics(target_labels[evaluation], probability, target_weights[evaluation])})
        selections.extend({"phase": phase, "subject": subject, "shots_per_class": shots, "trial_id": trial} for trial in chosen)

    print("[4/5] aggregating calibration budgets", flush=True)
    for shots in (0, 1, 2, 5):
        indices, probabilities = [], []
        for subject in subjects:
            chosen = [row["trial_id"] for row in selections if row["subject"] == subject and row["shots_per_class"] == shots]
            calibration = (target_users == subject) & np.isin(target_trials, chosen)
            evaluation = (target_users == subject) & ~calibration
            index = np.flatnonzero(evaluation); indices.append(index)
            probabilities.append(calibrated_probability(calibration, evaluation, shots))
        ordered = np.concatenate(indices); probability = np.concatenate(probabilities)
        rows.append({"phase": phase, "dataset": "emg_epn612", "subject": "ALL", "condition": "cross_user",
                     "shots_per_class": shots, "feature_bank": "+".join(FROZEN_FAMILIES),
                     "mode": "population_zero_shot" if shots == 0 else ("population_plus_F7_anchor" if method == "anchor" else "population_weighted_finetune"),
                     "evaluation_trials": len(np.unique(target_trials[ordered])), **_metrics(target_labels[ordered], probability, target_weights[ordered])})

    print("[5/5] writing leakage-auditable calibration evidence", flush=True)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "calibration_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (output / "calibration_trial_ids.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["phase", "subject", "shots_per_class", "trial_id"]); writer.writeheader(); writer.writerows(selections)
    np.savez_compressed(output / 'heldout_predictions.npz', **saved)
    (output / 'fitted_source_state.pkl').write_bytes(pickle.dumps({'families': family_states,
        'scaler': scaler, 'classifier': model}))
    (output / 'split_trial_ids.json').write_text(json.dumps({'source': sorted(set(train_trials)),
        'target_cases': split_ids}, indent=2), encoding='utf-8')
    (output / "run_manifest.json").write_text(json.dumps({"phase": phase, "seed": SEED, "train_users": list(range(1, 16)),
        "target_users": list(subjects), "families": list(FROZEN_FAMILIES), "dimensions": dimensions,
        "post_onset_windows_per_trial": 4, "calibration_method": method,
        "native_sample_rate_hz": 200, "native_emg_channels": 8, "native_imu_channels": 6,
        "native_classes": list(GESTURES), "source_trial_ids_artifact": "split_trial_ids.json",
        "prediction_artifact": "heldout_predictions.npz", "source_state_artifact": "fitted_source_state.pkl",
        "classifier": {"type": "LogisticRegression", "C": 1.0, "class_weight": "balanced", "max_iter": 1000, "random_state": SEED},
        "scaler": "StandardScaler fit on source users1-15 only",
        "personal_anchor_shrinkage": "shots/(2+shots)" if method == "anchor" else None,
        "anchor_temperature_fit": "calibration-only median prototype distance" if method == "anchor" else None,
        "finetune": {"C": 0.1, "calibration_sample_weight": "15/shots"} if method == "finetune" else None,
        "evaluation_excludes_complete_calibration_trials": True}, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "phase": phase, "rows": len(rows), "output": str(output)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="EPN612 0/1/2/5-shot personal calibration")
    parser.add_argument("archive", type=Path); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("validation", "final"), required=True); parser.add_argument("--subjects", type=int, nargs="+", required=True)
    parser.add_argument("--method", choices=("anchor", "finetune"), default="anchor")
    args = parser.parse_args(); run(args.archive, args.output, tuple(args.subjects), args.phase, args.method)


if __name__ == "__main__":
    main()
