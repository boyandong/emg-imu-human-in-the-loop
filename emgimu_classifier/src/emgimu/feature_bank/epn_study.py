from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.preprocessing import StandardScaler

from emgimu.datasets.epn612 import GESTURES, EpnWindows, load_epn612_windows

from .families import BodyContextFamily
from .screening import FAMILY_FACTORIES, SEED, expected_calibration_error


CLASSES = np.arange(len(GESTURES))
FACTORIES = {**FAMILY_FACTORIES, "F6_IMU": BodyContextFamily}


def _metrics(y: np.ndarray, probability: np.ndarray, weights: np.ndarray) -> dict:
    prediction = probability.argmax(axis=1)
    one_hot = np.eye(len(CLASSES))[y]
    per_class = f1_score(y, prediction, labels=CLASSES, average=None, sample_weight=weights, zero_division=0)
    return {
        "macro_f1": float(f1_score(y, prediction, labels=CLASSES, average="macro", sample_weight=weights, zero_division=0)),
        "accuracy": float(accuracy_score(y, prediction, sample_weight=weights)),
        "log_loss": float(log_loss(y, probability, labels=CLASSES, sample_weight=weights)),
        "brier": float(np.average(np.mean((probability - one_hot) ** 2, axis=1), weights=weights)),
        "ece": expected_calibration_error(y, probability, weights),
        "per_class_f1_json": json.dumps({GESTURES[i]: float(value) for i, value in enumerate(per_class)}, separators=(",", ":")),
    }


def aggregate_trials(features: np.ndarray, data: EpnWindows) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    trials = np.unique(data.trials)
    output, labels, users = [], [], []
    for trial in trials:
        selected = data.trials == trial
        trial_labels = np.unique(data.labels[selected])
        trial_users = np.unique(data.users[selected])
        if len(trial_labels) != 1 or len(trial_users) != 1:
            raise ValueError(f"inconsistent EPN612 trial metadata: {trial}")
        output.append(features[selected].mean(axis=0)); labels.append(trial_labels[0]); users.append(trial_users[0])
    return np.stack(output), np.asarray(labels), np.asarray(users), trials, np.ones(len(trials), dtype=float)


def _fit(train_x: np.ndarray, train_y: np.ndarray, train_weight: np.ndarray, validation_x: np.ndarray) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    scaler = StandardScaler().fit(train_x, sample_weight=train_weight)
    classifier = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=SEED)
    classifier.fit(scaler.transform(train_x), train_y, sample_weight=train_weight)
    return classifier.predict_proba(scaler.transform(validation_x)), time.perf_counter() - started


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def run(archive: Path, output: Path) -> None:
    print("[1/4] loading EPN612 users 1-15 train and 16-18 validation", flush=True)
    train = load_epn612_windows(archive, users=range(1, 16))
    validation = load_epn612_windows(archive, users=range(16, 19))
    train_features, validation_features, dimensions = {}, {}, {}
    train_labels = train_users = train_trials = train_weights = None
    validation_labels = validation_users = validation_trials = validation_weights = None
    for index, (name, factory) in enumerate(FACTORIES.items(), 1):
        print(f"[2/4] feature {index}/{len(FACTORIES)} {name}", flush=True)
        family = factory()
        train_window_features = family.fit_transform(train.batch, train.labels)
        validation_window_features = family.transform(validation.batch)
        train_features[name], train_labels, train_users, train_trials, train_weights = aggregate_trials(train_window_features, train)
        validation_features[name], validation_labels, validation_users, validation_trials, validation_weights = aggregate_trials(validation_window_features, validation)
        dimensions[name] = train_features[name].shape[1]
    specs = {"B_F0": ("F0",), **{f"B_plus_{name}": ("F0", name) for name in FACTORIES if name != "F0"}}
    probabilities, rows = {}, []
    for index, (name, members) in enumerate(specs.items(), 1):
        print(f"[3/4] classifier {index}/{len(specs)} {name}", flush=True)
        probability, seconds = _fit(np.concatenate([train_features[x] for x in members], axis=1), train_labels, train_weights,
                                    np.concatenate([validation_features[x] for x in members], axis=1))
        probabilities[name] = probability
        for user in ("ALL", 16, 17, 18):
            selected = np.ones(len(validation_labels), dtype=bool) if user == "ALL" else validation_users == user
            rows.append({"dataset": "emg_epn612", "subject": user, "condition": "cross_user", "feature_family": "+".join(members),
                         "calibration_budget": 0, **_metrics(validation_labels[selected], probability[selected], validation_weights[selected]),
                         "feature_dimension": sum(dimensions[x] for x in members), "training_seconds": seconds})
    baseline = next(row for row in rows if row["subject"] == "ALL" and row["feature_family"] == "F0")
    incremental = []
    for name in FACTORIES:
        if name == "F0": continue
        candidate = next(row for row in rows if row["subject"] == "ALL" and row["feature_family"] == f"F0+{name}")
        incremental.append({"core_bank": "F0", "added_family": name, "condition": "cross_user",
                            "delta_logloss": float(baseline["log_loss"]) - float(candidate["log_loss"]),
                            "delta_macro_f1": float(candidate["macro_f1"]) - float(baseline["macro_f1"]),
                            "delta_brier": float(baseline["brier"]) - float(candidate["brier"])})
    print("[4/4] writing EPN612 development evidence", flush=True)
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "feature_family_results.csv", rows)
    _write(output / "conditional_incremental.csv", incremental)
    (output / "run_manifest.json").write_text(json.dumps({"seed": SEED, "cohort": "trainingJSON", "source_split": "trainingSamples",
        "train_users": list(range(1, 16)), "validation_users": [16, 17, 18], "test_users_unopened": [19, 20, 21],
        "post_onset_windows_per_trial": 4, "dimensions": dimensions}, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "train_trials": len(train_labels), "validation_trials": len(validation_labels), "output": str(output)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="EPN612 Feature Bank cross-user screening")
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(); run(args.archive, args.output)


if __name__ == "__main__":
    main()
