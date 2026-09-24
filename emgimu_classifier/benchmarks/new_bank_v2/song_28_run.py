"""Frozen source-only 28-state Song evaluation of the independent v2 families."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import scipy
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmarks.song_28_spd_increment_study import detailed_metrics
from benchmarks.song_real8_study import ARMS, HANDS, _join_batches, load_session
from benchmarks.song_spd_increment_study import trial_probabilities
from emgimu.feature_bank.families import BodyContextFamily, LocalDetailFamily
from emgimu.feature_bank.new_bank_v2 import (
    RestNoiseDetailV2, RingRelativeCovarianceV2, TraceCovarianceV2,
)


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "SONG_28_PROTOCOL.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
SOURCE = Path("E:/qxy/emg_meta/emg_meta/data/Song")
FACTORIES = {
    "legacy_F0": LocalDetailFamily,
    "F0v2": lambda: RestNoiseDetailV2(rest_label="neutral"),
    "IMU": BodyContextFamily,
    "F2a": lambda: TraceCovarianceV2(shrinkage=0.05),
    "F3c": lambda: RingRelativeCovarianceV2(shrinkage=0.05),
}


def check_protocol() -> None:
    if (PROTOCOL["train_sessions"] != ["S01", "S02"]
            or PROTOCOL["validation_session"] != "S03"
            or PROTOCOL["final_session"] != "S04"
            or PROTOCOL["reference_arm"] != "legacy_F0+IMU"
            or PROTOCOL["arms"] != ["F0v2+IMU", "F0v2+IMU+F2a", "F0v2+IMU+F3c", "F0v2+IMU+F2a+F3c"]):
        raise ValueError("frozen 28-state Song v2 protocol changed")


def evaluate(source: Path = SOURCE) -> dict:
    check_protocol()
    prior_runtime = json.loads((ROOT.parent / "song_real8" / "SPD_28_STATE_RESULTS.json").read_text(
        encoding="utf-8"))["runtime_versions"]
    current_runtime = {"python": sys.version.split()[0], "numpy": np.__version__,
                       "scipy": scipy.__version__, "scikit_learn": sklearn.__version__}
    if current_runtime != prior_runtime:
        raise RuntimeError(f"28-state baseline requires its saved runtime {prior_runtime}; "
                           f"current runtime is {current_runtime}")
    data = {sid: load_session(source / f"2026-09-18_{sid}", sid, "causal")
            for sid in ("S01", "S02", "S03", "S04")}
    source_batch = _join_batches([data[sid] for sid in PROTOCOL["train_sessions"]])
    source_y = np.concatenate([data[sid]["composite"] for sid in PROTOCOL["train_sessions"]])
    source_hand = np.concatenate([data[sid]["hand"] for sid in PROTOCOL["train_sessions"]])
    classes = np.unique(source_y)
    if len(classes) != 28 or set(classes) != {f"{arm}_{hand}" for arm in ARMS for hand in HANDS}:
        raise ValueError("source does not cover the native 28 states")
    features = {}
    dimensions = {}
    for name, factory in FACTORIES.items():
        family = factory().fit(source_batch, source_hand)
        features[name] = {sid: family.transform(item["batch"]) for sid, item in data.items()}
        dimensions[name] = int(features[name]["S01"].shape[1])
        print(f"Song 28 {name}: {dimensions[name]} features", flush=True)
    baseline_file = ROOT.parent / "song_real8" / "CAUSAL_RESULTS.json"
    baseline = json.loads(baseline_file.read_text(encoding="utf-8"))
    baseline_hashes = {row["session"]: row["sha256"] for row in baseline["source_audit"]}
    source_hashes = {sid: item["audit"]["sha256"] for sid, item in data.items()}
    if baseline_hashes != source_hashes or baseline["filter_mode"] != "causal":
        raise ValueError("source hashes/filter do not match saved causal baseline")
    result = {
        "protocol": PROTOCOL,
        "protocol_sha256": hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
        "saved_causal_baseline_sha256": hashlib.sha256(baseline_file.read_bytes()).hexdigest(),
        "runtime_versions": current_runtime,
        "source_hdf5_sha256": source_hashes,
        "source_rest_windows": int((source_hand == "neutral").sum()),
        "classes": classes.tolist(), "feature_dimensions": dimensions, "sessions": {},
    }
    rows = []
    all_arms = (PROTOCOL["reference_arm"], *PROTOCOL["arms"])
    for arm in all_arms:
        members = arm.split("+")
        x_source = np.concatenate([
            np.concatenate([features[name][sid] for name in members], axis=1)
            for sid in PROTOCOL["train_sessions"]], axis=0)
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000, random_state=0))
        model.fit(x_source, source_y)
        np.testing.assert_array_equal(model[-1].classes_, classes)
        if np.max(model[-1].n_iter_) >= 2000:
            raise RuntimeError(f"classifier did not converge: {arm}")
        for phase, sid in (("validation", "S03"), ("final", "S04")):
            x = np.concatenate([features[name][sid] for name in members], axis=1)
            trial_ids, truth, probability = trial_probabilities(
                data[sid]["composite"], model.predict_proba(x), data[sid]["trial"], classes)
            phase_result = result["sessions"].setdefault(phase, {
                "session": sid, "trial_ids": trial_ids.tolist(), "arms": {}})
            if phase_result["trial_ids"] != trial_ids.tolist():
                raise ValueError(f"trial coverage differs for {arm}/{phase}")
            scores = detailed_metrics(truth, probability, classes)
            phase_result["arms"][arm] = scores
            if arm == PROTOCOL["reference_arm"]:
                saved = baseline["secondary_28_state"]["validation" if phase == "validation" else "test"]
                for key in ("trials", "accuracy", "macro_f1"):
                    if not np.isclose(scores[key], saved[key], rtol=0, atol=1e-12):
                        raise ValueError(f"old 28-state baseline mismatch: {phase}/{key}: "
                                         f"current={scores[key]!r} saved={saved[key]!r}")
            for trial_id, label, vector in zip(trial_ids, truth, probability):
                rows.append({"phase": phase, "session": sid, "arm": arm,
                             "trial_id": str(trial_id), "label": str(label),
                             **{f"p_{name}": float(p) for name, p in zip(classes, vector)}})
            print(f"Song 28 {phase} {arm}: joint F1={scores['macro_f1']:.4f}, "
                  f"arm acc={scores['arm_accuracy']:.4f}, hand acc={scores['hand_accuracy']:.4f}", flush=True)
    scores = result["sessions"]["validation"]["arms"]
    result["validation_selected_arm"] = min(PROTOCOL["arms"], key=lambda arm: (
        -scores[arm]["macro_f1"], scores[arm]["log_loss"], PROTOCOL["arms"].index(arm)))
    result["prediction_rows"] = len(rows)
    (ROOT / "SONG_28_RESULTS.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (ROOT / "SONG_28_TRIAL_PREDICTIONS.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result


if __name__ == "__main__":
    evaluate()
