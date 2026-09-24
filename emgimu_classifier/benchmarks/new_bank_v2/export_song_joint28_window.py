"""Export and independently replay the frozen source-only Song 28-state model.

The bundle accepts already causally filtered, aligned windows. It is deliberately
not enabled in the collection UI until stream alignment has a separate audit.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmarks.new_bank_v2.song_arm_calibration import joint_probabilities
from benchmarks.song_real8_study import ARMS, HANDS, _join_batches, load_session, parse_label
from benchmarks.song_spd_increment_study import trial_probabilities
from emgimu.feature_bank.families import BodyContextFamily
from emgimu.feature_bank.new_bank_v2 import RestNoiseDetailV2, RingRelativeCovarianceV2, TraceCovarianceV2


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
BUNDLE = REPO / "collection/emg_meta/emg_meta/model_assets/song_joint28_window"
SOURCE = Path("E:/qxy/emg_meta/emg_meta/data/Song")
RESULTS = ROOT / "SONG_ARM_CAL_RESULTS.json"
TRIAL_CSV = ROOT / "SONG_ARM_CAL_TRIAL_PREDICTIONS.csv"
RUNTIME_FILE = REPO / "collection/emg_meta/emg_meta/emgforce/inference/song_joint28_local.py"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_state(pipeline) -> dict:
    scaler, classifier = pipeline.steps[0][1], pipeline.steps[1][1]
    return {"scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
            "coef": classifier.coef_.tolist(), "intercept": classifier.intercept_.tolist()}


def _runtime_class():
    spec = importlib.util.spec_from_file_location("song_joint28_local", RUNTIME_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SongJoint28WindowRuntime


def run(source: Path = SOURCE, bundle: Path = BUNDLE) -> dict:
    prior = json.loads(RESULTS.read_text(encoding="utf-8"))
    if prior["protocol"]["source_train_sessions"] != ["S01", "S02"]:
        raise ValueError("source training sessions changed")
    import scipy, sklearn, sys
    versions = {"python": sys.version.split()[0], "numpy": np.__version__,
                "scipy": scipy.__version__, "scikit_learn": sklearn.__version__}
    if versions != prior["runtime_versions"]:
        raise ValueError("model export requires the frozen experiment runtime")
    data = {sid: load_session(source / f"2026-09-18_{sid}", sid, "causal")
            for sid in ("S01", "S02", "S03", "S04")}
    hashes = {sid: data[sid]["audit"]["sha256"] for sid in data}
    if hashes != prior["source_hdf5_sha256"]:
        raise ValueError("source recording digest changed")
    batch = _join_batches([data[sid] for sid in ("S01", "S02")])
    hand_truth = np.concatenate([data[sid]["hand"] for sid in ("S01", "S02")])
    arm_truth = np.asarray([parse_label(str(label))[0] for sid in ("S01", "S02")
                            for label in data[sid]["composite"]])
    hand_families = (
        RestNoiseDetailV2(rest_label="neutral").fit(batch, hand_truth),
        TraceCovarianceV2(shrinkage=.05).fit(batch),
        RingRelativeCovarianceV2(shrinkage=.05).fit(batch),
    )
    imu_family = BodyContextFamily().fit(batch)
    hand_features = {sid: np.concatenate([family.transform(item["batch"])
                                          for family in hand_families], axis=1)
                     for sid, item in data.items()}
    arm_features = {sid: imu_family.transform(item["batch"])
                    for sid, item in data.items()}

    def fit(features: dict, labels: np.ndarray):
        pipeline = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000, random_state=20260924))
        pipeline.fit(np.concatenate([features[sid] for sid in ("S01", "S02")]), labels)
        return pipeline

    hand_model, arm_model = fit(hand_features, hand_truth), fit(arm_features, arm_truth)
    hand_classes, arm_classes = hand_model[-1].classes_, arm_model[-1].classes_
    if set(hand_classes) != set(HANDS) or set(arm_classes) != set(ARMS):
        raise ValueError("incomplete source class coverage")
    joint_classes = json.loads((ROOT / "SONG_28_RESULTS.json").read_text(encoding="utf-8"))["classes"]
    joint_indices = [[int(np.flatnonzero(arm_classes == arm)[0]),
                      int(np.flatnonzero(hand_classes == hand)[0])]
                     for arm, hand in map(parse_label, joint_classes)]
    artifact = {
        "format_version": 1, "model_kind": "song_28_source_factorized_f0v2_f2a_f3c_f6",
        "emg_rate_hz": 250, "imu_rate_hz": 112, "emg_channels": 8, "imu_channels": 6,
        "emg_window_samples": 50, "imu_window_samples": 22, "covariance_shrinkage": .05,
        "input_emg": "continuous causal 40Hz highpass and 50/100Hz notch, float32",
        "imu_alignment": "last 22 native IMU samples whose emg_sample_index <= end EMG index",
        "hand_classes": hand_classes.tolist(), "arm_classes": arm_classes.tolist(),
        "joint_classes": joint_classes, "joint_indices": joint_indices,
        "hand": {**_model_state(hand_model), "rest_thresholds": hand_families[0].thresholds_.tolist()},
        "arm": _model_state(arm_model),
    }
    bundle.mkdir(parents=True, exist_ok=True)
    model_path = bundle / "song_joint28_model.json"
    model_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {"format_version": 1, "artifact": model_path.name, "sha256": _sha(model_path),
                "model_status": "exploratory_one_person_one_day_offline_only",
                "source_hdf5_sha256": {sid: hashes[sid] for sid in ("S01", "S02")},
                "reference_results_sha256": _sha(RESULTS), "reference_trial_csv_sha256": _sha(TRIAL_CSV),
                "training_runtime": versions,
                "limitation": "Cued stable windows, one person/day; no continuous or physical-device validation."}
    manifest_path = bundle / "song_joint28_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    runtime = _runtime_class()(bundle)
    if tuple(runtime.joint_classes) != tuple(joint_classes):
        raise ValueError("exported class order changed")

    with TRIAL_CSV.open(newline="", encoding="utf-8") as stream:
        frozen = {(row["session"], row["trial_id"]): row for row in csv.DictReader(stream)
                  if row["arm"] == "source_arm_x_same_hand"}
    max_feature_error = max_branch_error = max_joint_error = max_trial_error = 0.0
    counts = {}
    for sid in ("S03", "S04"):
        item = data[sid]
        emg, imu = item["batch"].emg, item["batch"].imu
        replay_hand = np.empty((len(emg), 4)); replay_arm = np.empty((len(emg), 7))
        for i in range(len(emg)):
            hand_feature = runtime.hand_features(emg[i])
            arm_feature = runtime.arm_features(imu[i])
            max_feature_error = max(max_feature_error,
                                    float(np.max(np.abs(hand_feature - hand_features[sid][i]))),
                                    float(np.max(np.abs(arm_feature - arm_features[sid][i]))))
            replay_hand[i], replay_arm[i], joint = runtime.predict_filtered_window(emg[i], imu[i])
            max_joint_error = max(max_joint_error, abs(float(joint.sum()) - 1.0))
        native_hand = hand_model.predict_proba(hand_features[sid])
        native_arm = arm_model.predict_proba(arm_features[sid])
        max_branch_error = max(max_branch_error,
                               float(np.max(np.abs(replay_hand - native_hand))),
                               float(np.max(np.abs(replay_arm - native_arm))))
        arm_labels = np.asarray([parse_label(str(label))[0] for label in item["composite"]])
        hand_ids, _, trial_hand = trial_probabilities(item["hand"], replay_hand,
                                                      item["trial"], hand_classes)
        arm_ids, _, trial_arm = trial_probabilities(arm_labels, replay_arm,
                                                   item["trial"], arm_classes)
        if not np.array_equal(hand_ids, arm_ids):
            raise ValueError(f"trial alignment changed: {sid}")
        trial_joint = joint_probabilities(trial_hand, trial_arm,
                                          hand_classes, arm_classes, np.asarray(joint_classes))
        for trial_id, probability in zip(hand_ids, trial_joint):
            reference = frozen[(sid, str(trial_id))]
            expected = np.asarray([float(reference[f"p_{label}"]) for label in joint_classes])
            max_trial_error = max(max_trial_error, float(np.max(np.abs(probability - expected))))
        counts[sid] = {"windows": len(emg), "trials": len(hand_ids)}
        print(f"{sid}: replayed {len(emg)} windows and {len(hand_ids)} trials", flush=True)
    if max_feature_error > 1e-6 or max_branch_error > 1e-6 or max_trial_error > 1e-6:
        raise ValueError(f"export replay mismatch: feature={max_feature_error}, branch={max_branch_error}, trial={max_trial_error}")
    audit = {"status": "offline_window_replay_exact_within_1e-6",
             "model_sha256": _sha(model_path), "manifest_sha256": _sha(manifest_path),
             "runtime_source_sha256": _sha(RUNTIME_FILE), "reference_results_sha256": _sha(RESULTS),
             "reference_trial_csv_sha256": _sha(TRIAL_CSV), "source_hdf5_sha256": hashes,
             "heldout": counts, "maximum_feature_absolute_error": max_feature_error,
             "maximum_branch_probability_absolute_error": max_branch_error,
             "maximum_joint_normalization_error": max_joint_error,
             "maximum_frozen_trial_probability_absolute_error": max_trial_error,
             "boundary": "No stream timing, continuous recognition, hardware use or new-person/day accuracy proved."}
    (bundle / "song_joint28_replay_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return audit


if __name__ == "__main__":
    print(json.dumps(run(), indent=2), flush=True)
