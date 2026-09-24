"""Public-v8 DS2 gesture-only subject-held-out family increments.

This is a new bounded experiment. It is not the historical force B0/X1-H/X2
reproduction: force labels and exact old input identity are unavailable.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.families import (
    LocalDetailFamily, ScalePatternFamily, TraceCovarianceFamily,
    SpdTangentFamily, SpectralStateFamily,
)


SEED = 20260924
WINDOW_INDICES = (0, 40, 80)
WINDOW_STARTS = tuple(3000 + 75 * i for i in WINDOW_INDICES)
WINDOW_SAMPLES = 375
SPLITS = {
    "train": tuple(range(1, 13)),
    "validation": tuple(range(13, 17)),
    "final": tuple(range(17, 21)),
}
FAMILIES = {
    "F0": LocalDetailFamily,
    "F1_scale_pattern": ScalePatternFamily,
    "F2a_trace_covariance": TraceCovarianceFamily,
    "F2c_SPD": SpdTangentFamily,
    "F4_spectral": SpectralStateFamily,
}
ARMS = {"F0": ("F0",), **{f"F0_plus_{name}": ("F0", name)
                             for name in FAMILIES if name != "F0"}}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def trial_probabilities(window_probability: np.ndarray, trial_ids: np.ndarray,
                        labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids, truth, probabilities = [], [], []
    for trial in np.unique(trial_ids):
        selected = trial_ids == trial
        unique = np.unique(labels[selected])
        if len(unique) != 1 or np.sum(selected) != len(WINDOW_INDICES):
            raise ValueError("Trial windows or labels changed")
        ids.append(int(trial))
        truth.append(int(unique[0]))
        probabilities.append(window_probability[selected].mean(axis=0))
    return np.asarray(ids), np.asarray(truth), np.asarray(probabilities)


def metrics(truth: np.ndarray, probability: np.ndarray, classes: np.ndarray) -> dict:
    prediction = classes[np.argmax(probability, axis=1)]
    target = np.eye(len(classes))[np.searchsorted(classes, truth)]
    return {
        "trials": len(truth),
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, labels=classes,
                                   average="macro", zero_division=0)),
        "log_loss": float(log_loss(truth, probability, labels=classes)),
        "brier": float(np.mean(np.sum((probability - target) ** 2, axis=1))),
        "recall_by_code": {str(int(c)): float(recall_score(
            truth, prediction, labels=[c], average="macro", zero_division=0))
            for c in classes},
        "confusion": confusion_matrix(truth, prediction, labels=classes).tolist(),
    }


def run(raw_path: Path, tdms_join_path: Path, window_join_path: Path,
        output: Path) -> dict:
    classifier_root = Path(__file__).resolve().parents[3]
    print("[1/5] verify DS2 source and build unique-subject trial index", flush=True)
    tdms_rows = list(csv.DictReader(tdms_join_path.open(encoding="utf-8", newline="")))
    label_rows = list(csv.DictReader(window_join_path.open(encoding="utf-8", newline="")))
    if len(tdms_rows) != 2833 or len(label_rows) != 2863:
        raise ValueError("Public DS2 join artifacts changed")
    eligible = []
    excluded_mixed = []
    for row in tdms_rows:
        trial = int(row["raw_trial_index_zero_based"])
        subject = int(row["subject_folder"])
        if not 1 <= subject <= 20 or trial != int(label_rows[trial]["raw_trial_index_zero_based"]):
            raise ValueError("Trial/subject join inconsistent")
        label = label_rows[trial]["gesture_label_if_uniform"]
        if row["gesture_code_if_uniform"] != label:
            raise ValueError("TDMS and window label join disagree")
        if label == "N/A":
            excluded_mixed.append(trial)
        else:
            eligible.append((trial, subject, int(label)))
    if len({trial for trial, _, _ in eligible}) != len(eligible) or len(eligible) != 2832:
        raise ValueError("Expected 2832 uniquely matched, uniformly labelled trials")
    if excluded_mixed != [209]:
        raise ValueError("Unexpected mixed-label trial")
    raw = loadmat(raw_path, variable_names=["data_final_all"])["data_final_all"]
    if raw.shape != (2863, 3, 15000):
        raise ValueError("Raw MAT shape changed")
    trial_index = np.asarray([item[0] for item in eligible], dtype=np.int32)
    subjects = np.asarray([item[1] for item in eligible], dtype=np.int8)
    labels = np.asarray([item[2] for item in eligible], dtype=np.int8)
    windows = np.empty((len(eligible) * len(WINDOW_STARTS), WINDOW_SAMPLES, 3),
                       dtype=np.float32)
    for position, trial in enumerate(trial_index):
        for window, start in enumerate(WINDOW_STARTS):
            windows[position * len(WINDOW_STARTS) + window] = raw[
                trial, :, start:start + WINDOW_SAMPLES].T
    del raw
    repeated_trials = np.repeat(trial_index, len(WINDOW_STARTS))
    repeated_subjects = np.repeat(subjects, len(WINDOW_STARTS))
    repeated_labels = np.repeat(labels, len(WINDOW_STARTS))
    split_masks = {name: np.isin(repeated_subjects, ids)
                   for name, ids in SPLITS.items()}
    if np.sum([mask.astype(int) for mask in split_masks.values()], axis=0).min() != 1 or \
            np.sum([mask.astype(int) for mask in split_masks.values()], axis=0).max() != 1:
        raise ValueError("Subject partition is not disjoint and exhaustive")
    if any(set(np.unique(repeated_labels[mask])) != set(range(5))
           for mask in split_masks.values()):
        raise ValueError("A subject partition lacks a gesture code")
    batches = {name: FeatureBatch(windows[mask], 1500.0)
               for name, mask in split_masks.items()}

    print("[2/5] fit feature-family state only on train subjects", flush=True)
    features = {}
    feature_counts = {}
    for name, factory in FAMILIES.items():
        family = factory().fit(batches["train"], repeated_labels[split_masks["train"]])
        features[name] = {split: family.transform(batch) for split, batch in batches.items()}
        feature_counts[name] = len(family.feature_names)
        print(f"  {name}: {feature_counts[name]} features", flush=True)
    del batches, windows

    print("[3/5] train fixed logistic arms on train subjects", flush=True)
    models = {}
    for arm, names in ARMS.items():
        x_train = np.concatenate([features[name]["train"] for name in names], axis=1)
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000, random_state=SEED))
        model.fit(x_train, repeated_labels[split_masks["train"]])
        if not np.array_equal(model[-1].classes_, np.arange(5)):
            raise ValueError("Unexpected trained class order")
        models[arm] = model
        print(f"  trained {arm}", flush=True)

    print("[4/5] score held-out subjects, one vote per trial", flush=True)
    classes = np.arange(5)
    scores, subject_scores, prediction_rows, by_split_arm = {}, {}, [], {}
    subject_by_trial = dict(zip(trial_index, subjects))
    for split in ("validation", "final"):
        mask = split_masks[split]
        scores[split] = {}
        subject_scores[split] = {}
        by_split_arm[split] = {}
        for arm, names in ARMS.items():
            x = np.concatenate([features[name][split] for name in names], axis=1)
            ids, truth, probability = trial_probabilities(
                models[arm].predict_proba(x), repeated_trials[mask], repeated_labels[mask])
            if not np.array_equal(ids, np.sort(np.unique(repeated_trials[mask]))):
                raise ValueError("Trial identity changed during score aggregation")
            scores[split][arm] = metrics(truth, probability, classes)
            by_split_arm[split][arm] = (ids, truth, probability)
            for subject in SPLITS[split]:
                selected = np.asarray([subject_by_trial[i] == subject for i in ids])
                subject_scores[split].setdefault(str(subject), {})[arm] = metrics(
                    truth[selected], probability[selected], classes)
            for i, label, prob in zip(ids, truth, probability):
                prediction_rows.append({
                    "split": split, "arm": arm, "raw_trial_index_zero_based": int(i),
                    "subject_folder": int(subject_by_trial[i]), "gesture_code": int(label),
                    **{f"p{c}": float(prob[c]) for c in classes},
                })
    increments = {split: {arm: {
        "delta_log_loss": scores[split]["F0"]["log_loss"] - score["log_loss"],
        "delta_brier": scores[split]["F0"]["brier"] - score["brier"],
        "delta_macro_f1": score["macro_f1"] - scores[split]["F0"]["macro_f1"],
    } for arm, score in scores[split].items() if arm != "F0"}
        for split in ("validation", "final")}
    complementarity = {}
    for split in ("validation", "final"):
        complementarity[split] = {}
        ids0, truth0, p0 = by_split_arm[split]["F0"]
        correct0 = classes[np.argmax(p0, axis=1)] == truth0
        for arm in ARMS:
            if arm == "F0":
                continue
            ids, truth, probability = by_split_arm[split][arm]
            if not np.array_equal(ids, ids0) or not np.array_equal(truth, truth0):
                raise ValueError("Paired arms do not cover the same trials")
            correct = classes[np.argmax(probability, axis=1)] == truth
            errors0 = ~correct0
            errors = ~correct
            correlation = np.corrcoef(errors0.astype(float), errors.astype(float))[0, 1] \
                if np.any(errors0) and np.any(correct0) and np.any(errors) and np.any(correct) \
                else float("nan")
            complementarity[split][arm] = {
                "trials": len(ids),
                "disagreement_rate": float(np.mean(correct0 != correct)),
                "error_correlation": float(correlation) if np.isfinite(correlation) else None,
                "both_correct": int(np.sum(correct0 & correct)),
                "F0_only_correct": int(np.sum(correct0 & ~correct)),
                "candidate_only_correct": int(np.sum(~correct0 & correct)),
                "both_wrong": int(np.sum(~correct0 & ~correct)),
            }

    print("[5/5] write split-locked scores and trial probabilities", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    predictions = output / "TRIAL_PREDICTIONS.csv"
    with predictions.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(prediction_rows)
    conditional_path = output / "CONDITIONAL_INCREMENTAL.csv"
    with conditional_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "split", "candidate_arm", "trials", "delta_log_loss", "delta_brier",
            "delta_macro_f1"), lineterminator="\n")
        writer.writeheader()
        writer.writerows({"split": split, "candidate_arm": arm,
                          "trials": scores[split][arm]["trials"], **values}
                         for split, arms in increments.items() for arm, values in arms.items())
    complementarity_path = output / "ERROR_COMPLEMENTARITY.csv"
    with complementarity_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "split", "candidate_arm", "trials", "disagreement_rate",
            "error_correlation", "both_correct", "F0_only_correct",
            "candidate_only_correct", "both_wrong"), lineterminator="\n")
        writer.writeheader()
        writer.writerows({"split": split, "candidate_arm": arm, **values}
                         for split, arms in complementarity.items() for arm, values in arms.items())
    result = {
        "status": "public_v8_gesture_only_subject_held_out_exploratory",
        "source_sha256": {"raw_mat": sha256(raw_path),
                          "tdms_join_csv": sha256(tdms_join_path),
                          "window_join_csv": sha256(window_join_path),
                          "runner": sha256(Path(__file__)),
                          "feature_families": sha256(classifier_root / "src/emgimu/feature_bank/families.py"),
                          "feature_batch": sha256(classifier_root / "src/emgimu/feature_bank/core.py")},
        "runtime_versions": {name: importlib.metadata.version(name) for name in
                             ("numpy", "scipy", "scikit-learn")},
        "split_subject_folders": {name: list(ids) for name, ids in SPLITS.items()},
        "excluded_unmatched_raw_trials": 30,
        "excluded_mixed_gesture_label_trials": excluded_mixed,
        "eligible_trials": len(eligible),
        "window_indices_within_published_action": list(WINDOW_INDICES),
        "window_starts_zero_based": list(WINDOW_STARTS),
        "window_samples": WINDOW_SAMPLES,
        "windows_per_trial": len(WINDOW_STARTS),
        "sample_rate_hz": 1500,
        "feature_counts": feature_counts,
        "arms": {name: list(families) for name, families in ARMS.items()},
        "classifier": "StandardScaler + balanced multinomial LogisticRegression(C=1.0, max_iter=2000, random_state=20260924); fit on train subjects only",
        "scores": scores,
        "subject_scores": subject_scores,
        "increments_vs_F0": increments,
        "error_complementarity_vs_F0": complementarity,
        "trial_predictions": predictions.name,
        "trial_predictions_sha256": sha256(predictions),
        "conditional_incremental_csv": conditional_path.name,
        "conditional_incremental_sha256": sha256(conditional_path),
        "error_complementarity_csv": complementarity_path.name,
        "error_complementarity_sha256": sha256(complementarity_path),
        "boundary": "Public DS2 v8 gesture codes only; 3-channel raw EMG, not Song 8-channel input. One vote per complete trial and disjoint train/validation/final subject folders. Unknown-subject and mixed-label trials are excluded. No force-condition labels, historical B0/X1-H/X2 identity, live-device result or cross-day estimate is inferred. Fixed models were specified before final-subject scoring; this remains exploratory because the public candidate and families were previously inspected.",
    }
    (output / "RESULTS.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"eligible_trials": len(eligible),
                      "validation_trials": scores["validation"]["F0"]["trials"],
                      "final_trials": scores["final"]["F0"]["trials"]}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-mat", required=True, type=Path)
    parser.add_argument("--tdms-join", required=True, type=Path)
    parser.add_argument("--window-join", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.raw_mat, args.tdms_join, args.window_join, args.output)
