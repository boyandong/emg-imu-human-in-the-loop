"""Own-device 250 Hz, one-person exploratory trial-level new-bank screen."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmarks.song_real8_study import _join_batches, load_session
from benchmarks.song_spd_increment_study import trial_probabilities
from emgimu.feature_bank.families import LocalDetailFamily
from emgimu.feature_bank.new_bank_v1 import NEW_BANK_V1


ROOT = Path(__file__).resolve().parent
SOURCE = Path("E:/qxy/emg_meta/emg_meta/data/Song")
PROTOCOL = json.loads((ROOT / "SONG_PROTOCOL.json").read_text(encoding="utf-8"))
FACTORIES = {"F0": LocalDetailFamily, **NEW_BANK_V1}


def metrics(truth, probability, classes):
    prediction = classes[probability.argmax(axis=1)]
    one_hot = (truth[:, None] == classes[None, :]).astype(float)
    return {"trials": int(len(truth)), "accuracy": float(accuracy_score(truth, prediction)),
            "macro_f1": float(f1_score(truth, prediction, labels=classes, average="macro", zero_division=0)),
            "log_loss": float(log_loss(truth, probability, labels=classes)),
            "brier": float(np.mean(np.sum((probability - one_hot) ** 2, axis=1))),
            "recall": {str(label): float(value) for label, value in zip(classes,
                recall_score(truth, prediction, labels=classes, average=None, zero_division=0))}}


def run():
    data = {sid: load_session(SOURCE / f"2026-09-18_{sid}", sid, "causal")
            for sid in ("S01", "S02", "S03", "S04")}
    source_batch = _join_batches([data["S01"], data["S02"]])
    source_y = np.concatenate([data[sid]["hand"] for sid in ("S01", "S02")])
    names = tuple(dict.fromkeys(part for arm in PROTOCOL["arms"] for part in arm.split("+")))
    feature_values = {}
    feature_dimensions = {}
    for name in names:
        family = FACTORIES[name]().fit(source_batch, source_y)
        feature_values[name] = {sid: family.transform(value["batch"]) for sid, value in data.items()}
        feature_dimensions[name] = int(feature_values[name]["S01"].shape[1])
    classes = np.array(PROTOCOL["classes"])
    models = {}
    for arm in PROTOCOL["arms"]:
        members = arm.split("+")
        x = np.concatenate([np.concatenate([feature_values[name][sid] for name in members], axis=1)
                            for sid in PROTOCOL["train_sessions"]])
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000, random_state=20260924))
        model.fit(x, source_y)
        np.testing.assert_array_equal(model[-1].classes_, classes)
        models[arm] = model
    results = {"protocol": PROTOCOL, "source_hdf5_sha256": {
        sid: value["audit"]["sha256"] for sid, value in data.items()},
        "feature_dimensions": feature_dimensions, "sessions": {}}
    predictions = []
    for phase, sid in (("validation", PROTOCOL["validation_session"]),
                       ("final", PROTOCOL["final_session"])):
        arm_results = {}
        expected_ids = expected_truth = None
        for arm in PROTOCOL["arms"]:
            members = arm.split("+")
            x = np.concatenate([feature_values[name][sid] for name in members], axis=1)
            ids, truth, probability = trial_probabilities(
                data[sid]["hand"], models[arm].predict_proba(x), data[sid]["trial"], classes)
            if expected_ids is not None:
                np.testing.assert_array_equal(expected_ids, ids)
                np.testing.assert_array_equal(expected_truth, truth)
            expected_ids, expected_truth = ids, truth
            arm_results[arm] = metrics(truth, probability, classes)
            for trial, label, prob in zip(ids, truth, probability):
                predictions.append({"phase": phase, "session": sid, "arm": arm,
                                    "trial_id": str(trial), "label": str(label),
                                    **{f"p_{name}": float(value) for name, value in zip(classes, prob)}})
            print(f"Song {phase} {arm}: F1={arm_results[arm]['macro_f1']:.4f}", flush=True)
        results["sessions"][phase] = {"session": sid, "arms": arm_results,
            "trial_ids": expected_ids.tolist(),
            "validation_best_descriptive_arm": min(arm_results, key=lambda arm: (
                -arm_results[arm]["macro_f1"], arm_results[arm]["log_loss"]))}
    results["validation_selected_arm"] = results["sessions"]["validation"]["validation_best_descriptive_arm"]
    (ROOT / "SONG_RESULTS.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    with (ROOT / "SONG_TRIAL_PREDICTIONS.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)


if __name__ == "__main__":
    run()
