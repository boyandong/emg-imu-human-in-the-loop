"""Fixed paired source-only versus guided-session IMU arm calibration study."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmarks.song_28_spd_increment_study import detailed_metrics
from benchmarks.song_real8_study import ARMS, HANDS, IMU_WINDOW, WINDOW, _join_batches, _text, load_session, parse_label
from benchmarks.song_spd_increment_study import trial_probabilities
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.families import BodyContextFamily
from emgimu.feature_bank.new_bank_v2 import RestNoiseDetailV2, RingRelativeCovarianceV2, TraceCovarianceV2


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "SONG_ARM_CAL_PROTOCOL.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
SOURCE = Path("E:/qxy/emg_meta/emg_meta/data/Song")


def calibration_imu_windows(folder: Path, session: str) -> tuple[dict[str, np.ndarray], dict]:
    """Read only pre-formal guided calibration intervals, with native IMU indices."""
    with h5py.File(folder / "session.h5") as handle:
        indices = handle["streams/imu/emg_sample_index"][:]
        imu = np.column_stack((handle["streams/imu/accel"][:], handle["streams/imu/gyro"][:]))
        rows = handle["calibration_blocks"][:]
        formal = handle["trials"][:]
    if np.any(np.diff(indices) < 0) or len(indices) != len(imu):
        raise ValueError(f"nonmonotone or misaligned IMU stream: {session}")
    first_formal = min(int(row["trial_start_sample"]) for row in formal
                       if _text(row["trial_kind"]) == "formal")
    required = set(PROTOCOL["calibration_blocks"])
    blocks = {}
    audit = {}
    for row in rows:
        name = _text(row["label"])
        if name not in required:
            continue
        start, end = int(row["stable_start_sample"]), int(row["stable_end_sample"])
        trial_start, trial_end = int(row["trial_start_sample"]), int(row["trial_end_sample"])
        if (name in blocks or not bool(row["valid"])
                or _text(row["completion_status"]) != "completed"
                or not 0 <= trial_start <= start < end <= trial_end <= first_formal):
            raise ValueError(f"invalid or duplicate pre-formal calibration: {session}/{name}")
        left, right = np.searchsorted(indices, (start, end), side="left")
        segment = imu[left:right]
        windows = [segment[i:i + IMU_WINDOW]
                   for i in range(0, len(segment) - IMU_WINDOW + 1, IMU_WINDOW // 2)]
        if not windows:
            raise ValueError(f"calibration interval shorter than 22 IMU samples: {session}/{name}")
        blocks[name] = np.stack(windows)
        audit[name] = {"trial_id": int(row["trial_id"]),
                       "trial_start_sample": trial_start, "trial_end_sample": trial_end,
                       "emg_stable_start": start,
                       "emg_stable_end": end, "imu_samples": int(len(segment)),
                       "descriptor_windows": len(windows), "preformal": True}
    if set(blocks) != required:
        raise ValueError(f"missing guided calibration blocks: {session}: {sorted(required - set(blocks))}")
    return blocks, audit


def imu_features(family: BodyContextFamily, imu_windows: np.ndarray) -> np.ndarray:
    dummy_emg = np.zeros((len(imu_windows), WINDOW, 8), dtype=np.float32)
    return family.transform(FeatureBatch(dummy_emg, 250.0, imu_windows))


def prototypes(family: BodyContextFamily, blocks: dict[str, np.ndarray], classes: np.ndarray):
    values = {name: imu_features(family, windows) for name, windows in blocks.items()}
    rest = np.median(np.concatenate((values["calibration_rest_initial"],
                                     values["calibration_rest_final"])), axis=0)
    directions = []
    for arm in classes:
        directions.append(np.zeros_like(rest) if arm == "still" else
                          np.median(values[f"calibration_arm_{arm}"], axis=0) - rest)
    return rest, np.stack(directions)


def joint_probabilities(hand: np.ndarray, arm: np.ndarray, hand_classes: np.ndarray,
                        arm_classes: np.ndarray, joint_classes: np.ndarray) -> np.ndarray:
    result = np.empty((len(hand), len(joint_classes)), dtype=np.float64)
    for index, label in enumerate(joint_classes):
        cue, gesture = parse_label(str(label))
        result[:, index] = (hand[:, np.flatnonzero(hand_classes == gesture)[0]] *
                            arm[:, np.flatnonzero(arm_classes == cue)[0]])
    return result / result.sum(axis=1, keepdims=True)


def run(source: Path = SOURCE) -> dict:
    if (PROTOCOL["source_train_sessions"] != ["S01", "S02"]
            or PROTOCOL["validation_session"] != "S03"
            or PROTOCOL["final_session"] != "S04"
            or PROTOCOL["arms"] != ["source_arm_x_same_hand", "guided_cal_arm_x_same_hand"]):
        raise ValueError("frozen guided arm protocol changed")
    prior_path = ROOT / "SONG_28_RESULTS.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    import sys, scipy, sklearn
    current_runtime = {"python": sys.version.split()[0], "numpy": np.__version__,
                       "scipy": scipy.__version__, "scikit_learn": sklearn.__version__}
    if current_runtime != prior["runtime_versions"]:
        raise RuntimeError(f"paired 28-state baseline requires {prior['runtime_versions']}; got {current_runtime}")
    data = {sid: load_session(source / f"2026-09-18_{sid}", sid, "causal")
            for sid in ("S01", "S02", "S03", "S04")}
    source_hashes = {sid: item["audit"]["sha256"] for sid, item in data.items()}
    if source_hashes != prior["source_hdf5_sha256"]:
        raise ValueError("source HDF5 files differ from previous 28-state study")
    source_batch = _join_batches([data[sid] for sid in PROTOCOL["source_train_sessions"]])
    source_hand = np.concatenate([data[sid]["hand"] for sid in PROTOCOL["source_train_sessions"]])
    source_arm = np.asarray([parse_label(label)[0] for sid in PROTOCOL["source_train_sessions"]
                             for label in data[sid]["composite"]])
    hand_families = {
        "F0v2": RestNoiseDetailV2(rest_label="neutral").fit(source_batch, source_hand),
        "F2a": TraceCovarianceV2(shrinkage=0.05).fit(source_batch),
        "F3c": RingRelativeCovarianceV2(shrinkage=0.05).fit(source_batch),
    }
    hand_features = {sid: np.concatenate([family.transform(item["batch"])
                                          for family in hand_families.values()], axis=1)
                     for sid, item in data.items()}
    imu_family = BodyContextFamily().fit(source_batch)
    arm_features = {sid: imu_family.transform(item["batch"]) for sid, item in data.items()}
    hand_model = make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=2000, random_state=20260924))
    hand_model.fit(np.concatenate([hand_features[sid] for sid in ("S01", "S02")]), source_hand)
    arm_model = make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=2000, random_state=20260924))
    arm_model.fit(np.concatenate([arm_features[sid] for sid in ("S01", "S02")]), source_arm)
    hand_classes, arm_classes = hand_model[-1].classes_, arm_model[-1].classes_
    joint_classes = np.asarray(prior["classes"])
    if set(hand_classes) != set(HANDS) or set(arm_classes) != set(ARMS):
        raise ValueError("source hand or arm class coverage incomplete")
    print("Song arm calibration: source hand and arm models fitted", flush=True)
    blocks = {}; calibration_audit = {}; rest = {}; guide = {}
    for sid in data:
        blocks[sid], calibration_audit[sid] = calibration_imu_windows(
            source / f"2026-09-18_{sid}", sid)
        rest[sid], guide[sid] = prototypes(imu_family, blocks[sid], arm_classes)
        print(f"{sid}: {sum(len(v) for v in blocks[sid].values())} pre-formal calibration descriptor windows", flush=True)
    calibration_burden = {}
    for sid, audit in calibration_audit.items():
        all_blocks = list(audit.values())
        arm_blocks = [value for name, value in audit.items() if name.startswith("calibration_arm_")]
        calibration_burden[sid] = {
            "selected_eight_block_span_seconds": (max(item["trial_end_sample"] for item in all_blocks)
                - min(item["trial_start_sample"] for item in all_blocks)) / 250.0,
            "six_guided_arm_block_span_seconds": (max(item["trial_end_sample"] for item in arm_blocks)
                - min(item["trial_start_sample"] for item in arm_blocks)) / 250.0,
        }
    source_scale = np.maximum(np.std(np.concatenate([guide[sid] for sid in ("S01", "S02")]), axis=0), .05)
    rows = []; results = {
        "protocol": PROTOCOL, "protocol_sha256": hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
        "runtime_versions": current_runtime, "source_hdf5_sha256": source_hashes,
        "reference_28_results_sha256": hashlib.sha256(prior_path.read_bytes()).hexdigest(),
        "source_arm_coordinate_scale": source_scale.tolist(),
        "calibration_blocks": calibration_audit,
        "calibration_burden": calibration_burden, "sessions": {},
    }
    for phase, sid in (("validation", "S03"), ("final", "S04")):
        item = data[sid]
        hand_ids, hand_truth, p_hand = trial_probabilities(
            item["hand"], hand_model.predict_proba(hand_features[sid]), item["trial"], hand_classes)
        arm_labels = np.asarray([parse_label(label)[0] for label in item["composite"]])
        arm_ids, arm_truth, p_arm_source = trial_probabilities(
            arm_labels, arm_model.predict_proba(arm_features[sid]), item["trial"], arm_classes)
        distance = np.sum(((arm_features[sid][:, None, :] - rest[sid][None, None, :]
                            - guide[sid][None, :, :]) / source_scale[None, None, :]) ** 2, axis=2)
        calibrated_window_probability = softmax(-0.5 * distance, axis=1)
        guided_ids, guided_truth, p_arm_guided = trial_probabilities(
            arm_labels, calibrated_window_probability, item["trial"], arm_classes)
        joint_ids, joint_truth, _ = trial_probabilities(
            item["composite"], np.ones((len(item["trial"]), len(joint_classes))) / len(joint_classes),
            item["trial"], joint_classes)
        if not (np.array_equal(hand_ids, arm_ids) and np.array_equal(hand_ids, guided_ids)
                and np.array_equal(hand_ids, joint_ids) and np.array_equal(arm_truth, guided_truth)):
            raise ValueError(f"trial identities differ between calibrated and source branches: {sid}")
        candidate_probabilities = {
            "source_arm_x_same_hand": joint_probabilities(p_hand, p_arm_source,
                hand_classes, arm_classes, joint_classes),
            "guided_cal_arm_x_same_hand": joint_probabilities(p_hand, p_arm_guided,
                hand_classes, arm_classes, joint_classes),
        }
        result = {"session": sid, "trial_ids": joint_ids.tolist(), "arms": {}}
        for name, probability in candidate_probabilities.items():
            result["arms"][name] = detailed_metrics(joint_truth, probability, joint_classes)
            for trial, label, vector in zip(joint_ids, joint_truth, probability):
                rows.append({"phase": phase, "session": sid, "arm": name,
                             "trial_id": str(trial), "label": str(label),
                             **{f"p_{klass}": float(p) for klass, p in zip(joint_classes, vector)}})
            print(f"Song arm {phase} {name}: joint F1={result['arms'][name]['macro_f1']:.4f}", flush=True)
        results["sessions"][phase] = result
    results["prediction_rows"] = len(rows)
    (ROOT / "SONG_ARM_CAL_RESULTS.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    with (ROOT / "SONG_ARM_CAL_TRIAL_PREDICTIONS.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return results


if __name__ == "__main__":
    run()
