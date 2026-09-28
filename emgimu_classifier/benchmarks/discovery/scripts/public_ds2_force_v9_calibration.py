"""Frozen public-DS2 personal calibration budgets with force-safe target splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.families import LocalDetailFamily, ScalePatternFamily
from public_ds2_force_v9 import ARMS, MODES, SPLITS, WINDOW_SAMPLES, WINDOW_STARTS, sha256
from public_ds2_subject_gesture_incremental import metrics, trial_probabilities


SHOTS = (0, 1, 2, 5)
SCHEDULE = {"unseen_high": (1, 0, 1, 0, 1),
            "product_all": (1, 0, 2, 1, 0)}


def calibration_assignment(ids: np.ndarray, subjects: np.ndarray,
                           gestures: np.ndarray, forces: np.ndarray,
                           mode: str, subject: int) -> dict[int, tuple[int, ...]]:
    assignments = {}
    for gesture in range(4):
        used = set()
        selected = []
        for force in SCHEDULE[mode]:
            candidates = (int(trial) for trial in ids[
                (subjects == subject) & (gestures == gesture) & (forces == force)]
                if int(trial) not in used)
            ordered = sorted(candidates, key=lambda trial: hashlib.sha256(
                f"20260928|{subject}|{gesture}|{trial}".encode()).digest())
            if not ordered:
                raise ValueError(f"not enough calibration trials for {mode}/{subject}/{gesture}/{force}")
            chosen = ordered[0]
            selected.append(chosen)
            used.add(chosen)
        assignments[gesture] = tuple(selected)
    return assignments


def score_group(truth: np.ndarray, probability: np.ndarray,
                force: np.ndarray, subject: np.ndarray) -> dict:
    classes = np.arange(4)
    return {
        "all": metrics(truth, probability, classes),
        "by_force_code": {str(level): metrics(truth[force == level],
                                               probability[force == level], classes)
                          for level in sorted(set(force))},
        "by_subject": {str(person): metrics(truth[subject == person],
                                             probability[subject == person], classes)
                       for person in sorted(set(subject))},
    }


def run(raw_path: Path, join_path: Path, force_audit_path: Path,
        base_dir: Path, protocol_path: Path, output: Path) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    audit = json.loads(force_audit_path.read_text(encoding="utf-8"))
    baseline = json.loads((base_dir / "RESULTS.json").read_text(encoding="utf-8"))
    if (sha256(raw_path) != audit["source_sha256"]["v8_raw_mat"] or
            sha256(join_path) != audit["trial_join_csv_sha256"] or
            baseline["source_sha256"]["force_label_audit"] != sha256(force_audit_path) or
            protocol["budgets_per_gesture"] != list(SHOTS) or
            {key: list(values) for key, values in SCHEDULE.items()} != protocol["calibration_schedule"]):
        raise ValueError("frozen source, parent study or calibration protocol changed")
    previous = {}
    with (base_dir / "TRIAL_PREDICTIONS.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["training_mode"], row["arm"], row["split"],
                   int(row["raw_trial_index_zero_based"]))
            previous[key] = np.array([float(row[f"p{k}"]) for k in range(4)])
    if len(previous) != 5454 or sha256(base_dir / "TRIAL_PREDICTIONS.csv") != baseline["trial_predictions_sha256"]:
        raise ValueError("parent frozen probabilities changed")

    print("[1/4] verify trial identities and construct fixed 375-sample windows", flush=True)
    rows = list(csv.DictReader(join_path.open(encoding="utf-8", newline="")))
    eligible = [row for row in rows if row["subject_folder"] != "N/A"
                and row["gesture_code_if_uniform"] != "N/A"
                and int(row["gesture_code_if_uniform"]) in range(4)]
    if len(eligible) != 2297:
        raise ValueError("eligible active-trial count changed")
    ids = np.array([int(row["raw_trial_index_zero_based"]) for row in eligible], dtype=np.int32)
    subjects = np.array([int(row["subject_folder"]) for row in eligible], dtype=np.int8)
    gestures = np.array([int(row["gesture_code_if_uniform"]) for row in eligible], dtype=np.int8)
    forces = np.array([int(row["force_code"]) for row in eligible], dtype=np.int8)
    raw = loadmat(raw_path, variable_names=["data_final_all"])["data_final_all"]
    if raw.shape != (2863, 3, 15000):
        raise ValueError("raw MAT shape changed")
    windows = np.empty((len(ids) * 3, WINDOW_SAMPLES, 3), dtype=np.float32)
    for position, trial in enumerate(ids):
        for slot, start in enumerate(WINDOW_STARTS):
            windows[position * 3 + slot] = raw[trial, :, start:start + WINDOW_SAMPLES].T
    del raw
    window_ids = np.repeat(ids, 3)
    window_subjects = np.repeat(subjects, 3)
    window_gestures = np.repeat(gestures, 3)
    window_forces = np.repeat(forces, 3)
    trial_position = {int(trial): position for position, trial in enumerate(ids)}
    output.mkdir(parents=True, exist_ok=True)
    assignment_rows = []
    assignments = {}
    for split in ("validation", "final_descriptive"):
        for subject in SPLITS[split]:
            for mode in MODES:
                assigned = calibration_assignment(ids, subjects, gestures, forces, mode, subject)
                assignments[(mode, subject)] = assigned
                for gesture, selected in assigned.items():
                    for shot, trial in enumerate(selected, 1):
                        assignment_rows.append({"split": split, "training_mode": mode,
                                                "subject_folder": subject, "gesture_code": gesture,
                                                "shot_slot": shot, "raw_trial_index_zero_based": trial,
                                                "force_code": int(forces[trial_position[trial]])})
    assignment_file = output / "CALIBRATION_ASSIGNMENT.csv"
    with assignment_file.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(assignment_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(assignment_rows)

    print("[2/4] replay population models before calibration", flush=True)
    prediction_rows = []
    scores = {}
    maximum_parent_difference = 0.0
    for mode, allowed_forces in MODES.items():
        train_mask = np.isin(window_subjects, SPLITS["train"]) & np.isin(window_forces, allowed_forces)
        train_batch = FeatureBatch(windows[train_mask], 1500.0)
        families = {"F0": LocalDetailFamily(), "F1": ScalePatternFamily()}
        transformed = {}
        for name, family in families.items():
            family.fit(train_batch, window_gestures[train_mask])
            transformed[name] = {"train": family.transform(train_batch),
                                 **{split: family.transform(FeatureBatch(
                                     windows[np.isin(window_subjects, SPLITS[split])], 1500.0))
                                    for split in ("validation", "final_descriptive")}}
        scores[mode] = {}
        for arm, names in ARMS.items():
            x_train = np.concatenate([transformed[name]["train"] for name in names], axis=1)
            model = make_pipeline(StandardScaler(), LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=2000, random_state=20260928))
            model.fit(x_train, window_gestures[train_mask])
            if not np.array_equal(model[-1].classes_, np.arange(4)):
                raise ValueError("population class order changed")
            scaled_source_trials = model[0].transform(x_train).reshape(-1, 3, x_train.shape[1]).mean(axis=1)
            source_labels = window_gestures[train_mask][::3]
            source_centroids = np.stack([scaled_source_trials[source_labels == k].mean(axis=0)
                                         for k in range(4)])
            source_residual = scaled_source_trials - source_centroids[source_labels]
            source_variance = max(float(np.median(np.sum(source_residual ** 2, axis=1) /
                                                  x_train.shape[1])), 1e-8)
            scores[mode][arm] = {}
            for split in ("validation", "final_descriptive"):
                mask = np.isin(window_subjects, SPLITS[split])
                x = np.concatenate([transformed[name][split] for name in names], axis=1)
                split_ids, split_truth, base_probability = trial_probabilities(
                    model.predict_proba(x), window_ids[mask], window_gestures[mask])
                scaled_trials = model[0].transform(x).reshape(-1, 3, x.shape[1]).mean(axis=1)
                expected = np.stack([previous[(mode, arm, split, int(trial))] for trial in split_ids])
                difference = float(np.max(np.abs(base_probability - expected)))
                maximum_parent_difference = max(maximum_parent_difference, difference)
                if difference > 1e-12:
                    raise ValueError(f"population baseline failed exact replay: {mode}/{arm}/{split} {difference}")
                split_subjects = np.array([subjects[trial_position[int(trial)]] for trial in split_ids])
                split_forces = np.array([forces[trial_position[int(trial)]] for trial in split_ids])
                combined_by_shot = {shot: [] for shot in SHOTS}
                for subject in SPLITS[split]:
                    own = split_subjects == subject
                    own_ids = split_ids[own]
                    own_features = scaled_trials[own]
                    own_source_probability = base_probability[own]
                    own_labels = split_truth[own]
                    own_forces = split_forces[own]
                    chosen = assignments[(mode, subject)]
                    reserved = {trial for selections in chosen.values() for trial in selections}
                    evaluation = (own_forces == 2 if mode == "unseen_high" else
                                  ~np.isin(own_ids, tuple(reserved)))
                    if not np.any(evaluation):
                        raise ValueError("subject has no evaluation trials")
                    for shot in SHOTS:
                        probability = own_source_probability[evaluation].copy()
                        if shot:
                            prototypes = np.stack([np.mean([
                                own_features[np.where(own_ids == trial)[0][0]]
                                for trial in chosen[gesture][:shot]], axis=0)
                                for gesture in range(4)])
                            delta = (own_features[evaluation, None, :].astype(np.float64) -
                                     prototypes[None, :, :].astype(np.float64))
                            distances = np.sum(delta * delta, axis=2) / x.shape[1]
                            personal = softmax(-distances / (2 * source_variance), axis=1)
                            alpha = shot / (shot + 5)
                            probability = (1 - alpha) * probability + alpha * personal
                            probability /= probability.sum(axis=1, keepdims=True)
                        selected_ids = own_ids[evaluation]
                        selected_truth = own_labels[evaluation]
                        selected_force = own_forces[evaluation]
                        combined_by_shot[shot].append((selected_truth, probability,
                                                       selected_force, np.full(len(selected_ids), subject)))
                        for trial, truth, force, probs in zip(selected_ids, selected_truth,
                                                              selected_force, probability):
                            prediction_rows.append({
                                "training_mode": mode, "arm": arm, "split": split,
                                "subject_folder": int(subject), "shots_per_gesture": shot,
                                "raw_trial_index_zero_based": int(trial),
                                "gesture_code": int(truth), "force_code": int(force),
                                **{f"p{k}": float(probs[k]) for k in range(4)},
                            })
                scores[mode][arm][split] = {}
                for shot, pieces in combined_by_shot.items():
                    truth, probs, force, person = (np.concatenate([part[index] for part in pieces])
                                                   for index in range(4))
                    scores[mode][arm][split][str(shot)] = score_group(truth, probs, force, person)
            print(f"  {mode}/{arm}: 0/1/2/5-shot", flush=True)

    print("[3/4] save fixed-budget native-trial predictions", flush=True)
    prediction_file = output / "CALIBRATION_TRIAL_PREDICTIONS.csv"
    with prediction_file.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(prediction_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(prediction_rows)
    result = {
        "status": "public_ds2_v9_force_personal_calibration_descriptive",
        "protocol_sha256": sha256(protocol_path),
        "source_sha256": {"raw_mat": sha256(raw_path), "trial_join": sha256(join_path),
                          "force_audit": sha256(force_audit_path),
                          "parent_results": sha256(base_dir / "RESULTS.json"),
                          "parent_predictions": sha256(base_dir / "TRIAL_PREDICTIONS.csv"),
                          "runner": sha256(Path(__file__))},
        "population_max_probability_difference_vs_parent": maximum_parent_difference,
        "calibration_assignment": assignment_file.name,
        "calibration_assignment_sha256": sha256(assignment_file),
        "trial_predictions": prediction_file.name,
        "trial_predictions_sha256": sha256(prediction_file),
        "prediction_rows": len(prediction_rows),
        "scores": scores,
        "boundary": "Personal calibration only. Unseen-high calibration uses force codes 0/1 and evaluates code 2, unchanged across budgets. Product mode reserves five trials/class before any budget and evaluates the same remaining trials at 0/1/2/5 shots. Source model/scaler/variance are source-only; target evaluation features and labels never fit calibration. Previously inspected final subjects are descriptive; no historical X1-H or own-device inference.",
    }
    (output / "CALIBRATION_RESULTS.json").write_bytes(
        (json.dumps(result, indent=2) + "\n").encode("utf-8"))
    print("[4/4] calibration curve complete", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-mat", required=True, type=Path)
    parser.add_argument("--trial-join", required=True, type=Path)
    parser.add_argument("--force-audit", required=True, type=Path)
    default = Path(__file__).resolve().parents[1] / "public_ds2_force_v9"
    parser.add_argument("--base-dir", type=Path, default=default)
    parser.add_argument("--protocol", type=Path, default=default / "CALIBRATION_PROTOCOL.json")
    parser.add_argument("--output", type=Path, default=default)
    args = parser.parse_args()
    run(args.raw_mat, args.trial_join, args.force_audit,
        args.base_dir, args.protocol, args.output)


if __name__ == "__main__":
    main()
