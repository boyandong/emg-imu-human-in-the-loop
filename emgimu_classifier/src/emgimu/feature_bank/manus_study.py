from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.preprocessing import StandardScaler

from emgimu.datasets.semg_manus import ManusWindows, load_semg_manus_windows

from .families import BodyContextFamily
from .screening import FAMILY_FACTORIES, SEED, expected_calibration_error


GESTURES = ("flexext_fist", "flexext_thumb", "flexext_index", "flexext_middle", "flexext_ring", "flexext_pinky")
FACTORIES = {**FAMILY_FACTORIES, "F6_IMU": BodyContextFamily}


def _aggregate(features: np.ndarray, data: ManusWindows) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    trials = np.unique(data.trials); output = []; labels = []; users = []; sessions = []; speeds = []
    for trial in trials:
        selected = data.trials == trial
        output.append(features[selected].mean(axis=0)); labels.append(np.unique(data.labels[selected]).item())
        users.append(np.unique(data.users[selected]).item()); sessions.append(np.unique(data.sessions[selected]).item())
        speeds.append(np.unique(data.speeds[selected]).item())
    return np.stack(output), np.asarray(labels), np.asarray(users), np.asarray(sessions), np.asarray(speeds), trials


def _metrics(y: np.ndarray, probability: np.ndarray) -> dict:
    prediction = probability.argmax(axis=1); one_hot = np.eye(len(GESTURES))[y]; weights = np.ones(len(y))
    per_class = f1_score(y, prediction, labels=np.arange(len(GESTURES)), average=None, zero_division=0)
    return {"macro_f1": float(f1_score(y, prediction, labels=np.arange(len(GESTURES)), average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y, prediction)), "log_loss": float(log_loss(y, probability, labels=np.arange(len(GESTURES)))),
        "brier": float(np.mean((probability - one_hot) ** 2)), "ece": expected_calibration_error(y, probability, weights),
        "per_class_f1_json": json.dumps({name: float(value) for name, value in zip(GESTURES, per_class)}, separators=(",", ":"))}


def run(archive: Path, output: Path, phase: str = "validation") -> None:
    users = (3, 4, 5, 6, 7, 8)
    target_session = 2 if phase == "validation" else 3
    print(f"[1/4] loading sEMG-MANUS session 1 train and session {target_session} {phase}", flush=True)
    train = load_semg_manus_windows(archive, users=users, sessions=(1,), gestures=GESTURES)
    validation = load_semg_manus_windows(archive, users=users, sessions=(target_session,), gestures=GESTURES)
    active_factories = FACTORIES if phase == "validation" else {"F0": FACTORIES["F0"], "F2c_SPD": FACTORIES["F2c_SPD"]}
    train_features, validation_features, dimensions = {}, {}, {}
    train_y = validation_y = validation_speed = None
    for index, (name, factory) in enumerate(active_factories.items(), 1):
        print(f"[2/4] feature {index}/{len(active_factories)} {name}", flush=True)
        family = factory(); a = family.fit_transform(train.batch, train.labels); b = family.transform(validation.batch)
        train_features[name], train_y, _, _, _, _ = _aggregate(a, train)
        validation_features[name], validation_y, _, _, validation_speed, _ = _aggregate(b, validation)
        dimensions[name] = train_features[name].shape[1]
    specs = {"B_F0": ("F0",), **{f"B_plus_{name}": ("F0", name) for name in active_factories if name != "F0"}}
    rows = []
    for index, (model_name, members) in enumerate(specs.items(), 1):
        print(f"[3/4] classifier {index}/{len(specs)} {model_name}", flush=True)
        a = np.concatenate([train_features[x] for x in members], axis=1); b = np.concatenate([validation_features[x] for x in members], axis=1)
        scaler = StandardScaler().fit(a); model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=SEED).fit(scaler.transform(a), train_y)
        probability = model.predict_proba(scaler.transform(b))
        for speed in ("ALL", "slow", "medium", "fast"):
            selected = np.ones(len(validation_y), dtype=bool) if speed == "ALL" else validation_speed == speed
            rows.append({"dataset": "semg_manus", "subject": "SAME_USERS", "session/domain": f"session_{target_session}", "condition": speed,
                "feature_family": "+".join(members), "calibration_budget": 0, **_metrics(validation_y[selected], probability[selected]),
                "feature_dimension": sum(dimensions[x] for x in members)})
    print("[4/4] writing cross-session and speed evidence", flush=True)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "feature_family_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (output / "run_manifest.json").write_text(json.dumps({"seed": SEED, "users": list(users), "gestures": list(GESTURES),
        "phase": phase, "train_session": 1, "target_session": target_session,
        "test_session_unopened": 3 if phase == "validation" else None, "speed_cells": ["slow", "medium", "fast"],
        "evaluation_unit": "trial", "windows_per_trial": 8, "dimensions": dimensions}, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "train_trials": len(train_y), "validation_trials": len(validation_y), "output": str(output)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="sEMG-MANUS session and speed Feature Bank screen")
    parser.add_argument("archive", type=Path); parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--phase", choices=("validation", "final"), default="validation")
    args = parser.parse_args(); run(args.archive, args.output, args.phase)


if __name__ == "__main__":
    main()
