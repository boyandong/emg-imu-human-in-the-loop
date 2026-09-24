"""Frozen one-person 250 Hz Song trial-level screen for new-bank v2."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmarks.new_bank_v1.song_run import metrics
from benchmarks.song_real8_study import _join_batches, load_session
from benchmarks.song_spd_increment_study import trial_probabilities
from emgimu.feature_bank.families import LocalDetailFamily
from emgimu.feature_bank.new_bank_v2 import (
    RestNoiseDetailV2, RingRelativeCovarianceV2, TraceCovarianceV2,
)


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "SONG_PROTOCOL.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
SOURCE = Path("E:/qxy/emg_meta/emg_meta/data/Song")
FACTORIES = {
    "legacy_F0": LocalDetailFamily,
    "F0v2": lambda: RestNoiseDetailV2(rest_label="neutral"),
    "F2a": lambda: TraceCovarianceV2(shrinkage=0.05),
    "F3c": lambda: RingRelativeCovarianceV2(shrinkage=0.05),
}


def check_protocol() -> None:
    if (PROTOCOL["train_sessions"] != ["S01", "S02"]
            or PROTOCOL["validation_session"] != "S03"
            or PROTOCOL["final_session"] != "S04"
            or PROTOCOL["reference_arm"] != "legacy_F0"
            or PROTOCOL["arms"] != ["F0v2", "F0v2+F2a", "F0v2+F3c", "F0v2+F2a+F3c"]):
        raise ValueError("frozen Song v2 protocol changed")


def evaluate(source: Path = SOURCE) -> dict:
    check_protocol()
    data = {sid: load_session(source / f"2026-09-18_{sid}", sid, "causal")
            for sid in ("S01", "S02", "S03", "S04")}
    source_batch = _join_batches([data[sid] for sid in PROTOCOL["train_sessions"]])
    source_y = np.concatenate([data[sid]["hand"] for sid in PROTOCOL["train_sessions"]])
    classes = np.asarray(PROTOCOL["classes"])
    feature_values = {}
    feature_dimensions = {}
    for name, factory in FACTORIES.items():
        family = factory().fit(source_batch, source_y)
        feature_values[name] = {sid: family.transform(value["batch"])
                                for sid, value in data.items()}
        feature_dimensions[name] = int(feature_values[name]["S01"].shape[1])
        print(f"Song v2 {name}: {feature_dimensions[name]} features", flush=True)
    arms = (PROTOCOL["reference_arm"], *PROTOCOL["arms"])
    result = {
        "protocol": PROTOCOL,
        "protocol_sha256": hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
        "source_hdf5_sha256": {sid: value["audit"]["sha256"] for sid, value in data.items()},
        "source_rest_windows": int((source_y == "neutral").sum()),
        "feature_dimensions": feature_dimensions,
        "sessions": {},
    }
    rows = []
    for arm in arms:
        members = arm.split("+")
        x_source = np.concatenate([
            np.concatenate([feature_values[name][sid] for name in members], axis=1)
            for sid in PROTOCOL["train_sessions"]], axis=0)
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000, random_state=20260924))
        model.fit(x_source, source_y)
        np.testing.assert_array_equal(model[-1].classes_, classes)
        if np.max(model[-1].n_iter_) >= 2000:
            raise RuntimeError(f"classifier did not converge: {arm}")
        for phase, sid in (("validation", "S03"), ("final", "S04")):
            x = np.concatenate([feature_values[name][sid] for name in members], axis=1)
            trial_ids, truth, probability = trial_probabilities(
                data[sid]["hand"], model.predict_proba(x), data[sid]["trial"], classes)
            phase_result = result["sessions"].setdefault(phase, {
                "session": sid, "trial_ids": trial_ids.tolist(), "arms": {}})
            if phase_result["trial_ids"] != trial_ids.tolist():
                raise ValueError(f"trial coverage differs for {arm}/{phase}")
            phase_result["arms"][arm] = metrics(truth, probability, classes)
            for trial_id, label, vector in zip(trial_ids, truth, probability):
                rows.append({"phase": phase, "session": sid, "arm": arm,
                             "trial_id": str(trial_id), "label": str(label),
                             **{f"p_{name}": float(p) for name, p in zip(classes, vector)}})
            print(f"Song v2 {phase} {arm}: F1={phase_result['arms'][arm]['macro_f1']:.4f}", flush=True)
    scores = result["sessions"]["validation"]["arms"]
    result["validation_selected_arm"] = min(PROTOCOL["arms"], key=lambda arm: (
        -scores[arm]["macro_f1"], scores[arm]["log_loss"], PROTOCOL["arms"].index(arm)))
    result["prediction_rows"] = len(rows)
    (ROOT / "SONG_RESULTS.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (ROOT / "SONG_TRIAL_PREDICTIONS.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result


if __name__ == "__main__":
    evaluate()
