"""Frozen whole-stream comparison for the source-only signed-axis arm candidate."""
from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmarks.new_bank_v2.song_arm_signed_study import signed_features
from benchmarks.new_bank_v2.song_joint28_continuous_replay import (
    BUNDLE, ROOT, RUNTIME_FILE, SOURCE, _intervals, _runtime_classes, _scores, _sha,
)
from benchmarks.song_real8_study import ARMS, _join_batches, load_session, parse_label
from benchmarks.song_spd_increment_study import trial_probabilities
from emgimu.feature_bank.families import BodyContextFamily


PROTOCOL_FILE = ROOT / "SONG_ARM_SIGNED_CONTINUOUS_PROTOCOL.json"
MODEL_FILE = ROOT / "SONG_ARM_SIGNED_ARM_MODEL.json"
OUTPUT_JSON = ROOT / "SONG_ARM_SIGNED_CONTINUOUS_RESULTS.json"
OUTPUT_CSV = ROOT / "SONG_ARM_SIGNED_CONTINUOUS_FRAMES.csv"
BASELINE_CSV = ROOT / "SONG_JOINT28_CONTINUOUS_FRAMES.csv"


class SignedArmWindow:
    def __init__(self, baseline, arm_model):
        self.baseline = baseline
        self.arm_model = arm_model
        self.joint_classes = baseline.joint_classes
        if tuple(arm_model[-1].classes_) != baseline.arm_classes:
            raise ValueError("source arm class order differs from tracked runtime")

    def predict_filtered_window(self, emg: np.ndarray, imu: np.ndarray):
        hand, _, _ = self.baseline.predict_filtered_window(emg, imu)
        features = np.concatenate((self.baseline.arm_features(imu),
                                   signed_features(np.asarray(imu)[None, :, :])[0]))
        scaler, classifier = self.arm_model.steps[0][1], self.arm_model.steps[1][1]
        arm = softmax(classifier.coef_ @ ((features - scaler.mean_) / scaler.scale_)
                      + classifier.intercept_)
        indices = self.baseline.joint_indices
        joint = arm[indices[:, 0]] * hand[indices[:, 1]]
        joint /= joint.sum()
        return hand, arm, joint


def _fit_arm(source: Path, expected_hashes: dict) -> tuple[object, dict]:
    data = {sid: load_session(source / f"2026-09-18_{sid}", sid, "causal")
            for sid in ("S01", "S02")}
    if {sid: item["audit"]["sha256"] for sid, item in data.items()} != {
            sid: expected_hashes[sid] for sid in data}:
        raise ValueError("source recording digest changed")
    family = BodyContextFamily().fit(_join_batches(list(data.values())))
    features = np.concatenate([
        np.column_stack((family.transform(data[sid]["batch"]),
                         signed_features(data[sid]["batch"].imu)))
        for sid in ("S01", "S02")])
    labels = np.asarray([parse_label(str(label))[0] for sid in ("S01", "S02")
                         for label in data[sid]["composite"]])
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=2000, random_state=20260924))
    model.fit(features, labels)
    scaler, classifier = model.steps[0][1], model.steps[1][1]
    state = {"source_sessions": ["S01", "S02"],
             "source_hdf5_sha256": {sid: expected_hashes[sid] for sid in ("S01", "S02")},
             "feature": "F6_13 + signed_gyro_mean_3 + signed_accel_late_minus_early_3",
             "classes": classifier.classes_.tolist(),
             "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
             "coef": classifier.coef_.tolist(), "intercept": classifier.intercept_.tolist()}
    return model, state


def _baseline_rows() -> dict[tuple[str, int], dict]:
    with BASELINE_CSV.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    result = {(row["session"], int(row["emg_end_index"])): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate baseline frame")
    return result


def _verify_trial_candidate(source: Path, model, expected_hashes: dict,
                            trial_study: dict) -> float:
    prediction_path = ROOT / "SONG_ARM_SIGNED_TRIAL_PREDICTIONS.csv"
    if _sha(prediction_path) != trial_study["prediction_csv_sha256"]:
        raise ValueError("signed-arm trial predictions changed")
    with prediction_path.open(newline="", encoding="utf-8") as stream:
        saved = {(row["session"], row["trial_id"]): row for row in csv.DictReader(stream)}
    if len(saved) != trial_study["prediction_rows"]:
        raise ValueError("signed-arm trial count changed")
    max_error = 0.0
    for sid in ("S03", "S04"):
        data = load_session(source / f"2026-09-18_{sid}", sid, "causal")
        if data["audit"]["sha256"] != expected_hashes[sid]:
            raise ValueError(f"trial replay recording changed: {sid}")
        family = BodyContextFamily().fit(data["batch"])
        features = np.column_stack((family.transform(data["batch"]),
                                    signed_features(data["batch"].imu)))
        truth = np.asarray([parse_label(str(label))[0] for label in data["composite"]])
        ids, _, probabilities = trial_probabilities(
            truth, model.predict_proba(features), data["trial"], model[-1].classes_)
        for trial, probability in zip(ids, probabilities):
            row = saved.pop((sid, str(trial)), None)
            if row is None:
                raise ValueError(f"missing signed-arm native trial: {sid}/{trial}")
            recorded = np.asarray([sum(float(row[f"p_candidate_{joint}"])
                                       for joint in trial_study["joint_classes"]
                                       if parse_label(joint)[0] == arm)
                                   for arm in model[-1].classes_])
            max_error = max(max_error, float(np.max(np.abs(probability - recorded))))
    if saved or max_error > 1e-6:
        raise ValueError(f"candidate trial replay mismatch: {len(saved)} unverified, {max_error} maximum error")
    return max_error


def run(source: Path = SOURCE) -> dict:
    protocol = json.loads(PROTOCOL_FILE.read_text(encoding="utf-8"))
    if (protocol["sessions"] != ["S03", "S04"]
            or protocol["source_train_sessions"] != ["S01", "S02"]):
        raise ValueError("continuous signed-arm split changed")
    trial_study = json.loads((ROOT / "SONG_ARM_SIGNED_RESULTS.json").read_text(encoding="utf-8"))
    if _sha(ROOT / "SONG_ARM_SIGNED_PROTOCOL.json") != trial_study["protocol_sha256"]:
        raise ValueError("trial-level candidate protocol changed")
    frozen = json.loads((ROOT / "SONG_JOINT28_CONTINUOUS_RESULTS.json").read_text(encoding="utf-8"))
    if _sha(BASELINE_CSV) != frozen["frame_csv_sha256"]:
        raise ValueError("baseline continuous frames changed")
    Window, Stream = _runtime_classes()
    baseline = Window(BUNDLE)
    if baseline.sha256 != frozen["model_sha256"]:
        raise ValueError("tracked baseline bundle changed")
    import scipy, sklearn, sys
    runtime = {"python": sys.version.split()[0], "numpy": np.__version__,
               "scipy": scipy.__version__, "scikit_learn": sklearn.__version__}
    if runtime != trial_study["runtime_versions"]:
        raise RuntimeError("candidate requires the recorded scientific runtime")
    arm_model, state = _fit_arm(source, trial_study["source_hdf5_sha256"])
    if set(state["classes"]) != set(ARMS):
        raise ValueError("source arm class coverage is incomplete")
    trial_candidate_error = _verify_trial_candidate(
        source, arm_model, trial_study["source_hdf5_sha256"], trial_study)
    MODEL_FILE.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    candidate = SignedArmWindow(baseline, arm_model)
    old_rows = _baseline_rows()
    rows = []
    sessions = {}
    max_hand_marginal_error = 0.0
    for sid in protocol["sessions"]:
        path = source / f"2026-09-18_{sid}" / "session.h5"
        if _sha(path) != trial_study["source_hdf5_sha256"][sid]:
            raise ValueError(f"recording digest changed: {sid}")
        with h5py.File(path) as handle:
            raw = handle["streams/emg/raw"][:]
            emg_indices = handle["streams/emg/sample_index"][:]
            imu_indices = handle["streams/imu/emg_sample_index"][:]
            accel = handle["streams/imu/accel"][:]
            gyro = handle["streams/imu/gyro"][:]
            formal = _intervals(handle["trials"][:], trial_kind="formal")
            rest = _intervals(handle["calibration_blocks"][:],
                              {"calibration_rest_initial", "calibration_rest_final"},
                              trial_kind="calibration")
        if (raw.shape[1] != 8 or accel.shape != gyro.shape or accel.shape[1] != 3
                or not np.array_equal(emg_indices, np.arange(len(raw)))
                or np.any(np.diff(imu_indices) < 0)):
            raise ValueError(f"native stream shape/index invalid: {sid}")
        stream = Stream(candidate)
        emitted = []
        imu_offset = 0
        started = time.perf_counter()
        for left in range(0, len(raw), 25):
            right = min(left + 25, len(raw))
            gap, frames = stream.ingest_emg(raw[left:right], emg_indices[left:right])
            if gap:
                raise ValueError(f"unexpected EMG index gap: {sid}")
            emitted.extend(frames)
            last = int(np.searchsorted(imu_indices, right, side="right"))
            emitted.extend(stream.ingest_imu(accel[imu_offset:last], gyro[imu_offset:last],
                                             imu_indices[imu_offset:last]))
            imu_offset = last
        if imu_offset < len(imu_indices):
            emitted.extend(stream.ingest_imu(accel[imu_offset:], gyro[imu_offset:],
                                             imu_indices[imu_offset:]))
        duration = time.perf_counter() - started
        frame_truth, frame_pred, trial_probability = [], [], defaultdict(list)
        rest_frames = rest_active_argmax = rest_active_display = 0
        event_count = 0
        event_intervals = Counter()
        candidate_label = active = None
        candidate_count = 0
        by_label = Counter()
        for end, probability in emitted:
            key = (sid, int(end))
            old = old_rows.get(key)
            if old is None:
                raise ValueError(f"candidate emitted frame absent from baseline: {key}")
            peak = int(np.argmax(probability))
            name = candidate.joint_classes[peak]
            by_label[name] += 1
            threshold_name = name if probability[peak] >= .15 else None
            if threshold_name == candidate_label:
                candidate_count += 1
            else:
                candidate_label, candidate_count = threshold_name, 1
            if candidate_count >= 3 and threshold_name != active:
                active = threshold_name
                event_fired = active is not None
                event_count += int(event_fired)
            else:
                event_fired = False
            begin = end - 49
            matching_formal = [interval for interval in formal
                               if interval[0] <= begin and end < interval[1]]
            matching_rest = [interval for interval in rest
                             if interval[0] <= begin and end < interval[1]]
            if len(matching_formal) + len(matching_rest) > 1:
                raise ValueError(f"overlapping scored intervals: {key}")
            kind, trial_id, label = "unlabelled", "", ""
            if matching_formal:
                _, _, trial_id, label = matching_formal[0]
                kind = "formal_stable"
                frame_truth.append(label); frame_pred.append(name)
                trial_probability[(trial_id, label)].append(probability)
                old_joint = np.asarray(json.loads(old["joint_probabilities"]))
                for index in range(4):
                    observed = probability[candidate.baseline.joint_indices[:, 1] == index].sum()
                    reference = old_joint[candidate.baseline.joint_indices[:, 1] == index].sum()
                    max_hand_marginal_error = max(max_hand_marginal_error, abs(float(observed - reference)))
            elif matching_rest:
                _, _, trial_id, label = matching_rest[0]
                kind = "calibration_rest_stable"
                rest_frames += 1
                rest_active_argmax += int(parse_label(name)[1] != "neutral")
                rest_active_display += int(active is not None and parse_label(active)[1] != "neutral")
            if (kind != old["interval_kind"] or str(trial_id) != old["trial_id"]
                    or label != old["cue_label"]):
                raise ValueError(f"candidate and baseline interval mismatch: {key}")
            if event_fired:
                event_intervals[kind] += 1
            rows.append({"session": sid, "emg_end_index": end,
                         "interval_kind": kind, "trial_id": trial_id, "cue_label": label,
                         "peak_label": name, "peak_probability": float(probability[peak]),
                         "display_label": active or "", "event_fired": int(event_fired),
                         "joint_probabilities": json.dumps(probability.tolist(), separators=(",", ":"))
                         if kind == "formal_stable" else ""})
        trial_truth = [label for _, label in trial_probability]
        trial_pred = [candidate.joint_classes[int(np.argmax(np.mean(values, axis=0)))]
                      for values in trial_probability.values()]
        scored_trial_ids = {trial_id for trial_id, _ in trial_probability}
        unscored = [{"trial_id": trial_id, "label": label, "stable_samples": end - start}
                    for start, end, trial_id, label in formal if trial_id not in scored_trial_ids]
        hand_truth = [parse_label(value)[1] for value in frame_truth]
        hand_pred = [parse_label(value)[1] for value in frame_pred]
        arm_truth = [parse_label(value)[0] for value in frame_truth]
        arm_pred = [parse_label(value)[0] for value in frame_pred]
        sessions[sid] = {
            "raw_emg_samples": len(raw), "raw_imu_samples": len(imu_indices),
            "replay_cpu_seconds_not_live_latency": duration,
            "emitted_frames": len(emitted), "dropped_frames": stream.dropped_frames,
            "formal_stable_intervals": len(formal), "calibration_rest_stable_intervals": len(rest),
            "scored_formal_trials": len(trial_probability),
            "formal_intervals_without_full_emitted_window": unscored,
            "stable_windows": _scores(frame_truth, frame_pred, candidate.joint_classes),
            "stable_hand_accuracy": float(accuracy_score(hand_truth, hand_pred)),
            "stable_arm_accuracy": float(accuracy_score(arm_truth, arm_pred)),
            "stable_hand_recall": {name: float(value) for name, value in zip(
                sorted({parse_label(x)[1] for x in candidate.joint_classes}),
                recall_score(hand_truth, hand_pred,
                             labels=sorted({parse_label(x)[1] for x in candidate.joint_classes}),
                             average=None, zero_division=0))},
            "stable_arm_recall": {name: float(value) for name, value in zip(
                sorted(ARMS), recall_score(arm_truth, arm_pred, labels=sorted(ARMS),
                                           average=None, zero_division=0))},
            "trial_mean_joint_probability": _scores(trial_truth, trial_pred, candidate.joint_classes),
            "rest_frames": rest_frames,
            "rest_active_hand_argmax_frames": rest_active_argmax,
            "rest_active_hand_argmax_fraction": rest_active_argmax / rest_frames if rest_frames else None,
            "rest_active_hand_display_frames": rest_active_display,
            "rest_active_hand_display_fraction": rest_active_display / rest_frames if rest_frames else None,
            "continuous_state_transitions_to_non_null": event_count,
            "state_transitions_by_scored_interval": dict(event_intervals),
            "unlabelled_frames": int(sum(row["interval_kind"] == "unlabelled"
                                           for row in rows if row["session"] == sid)),
            "all_frame_peak_counts": dict(by_label),
        }
        old_result = frozen["sessions"][sid]
        for field in ("emitted_frames", "dropped_frames", "scored_formal_trials", "rest_frames"):
            if sessions[sid][field] != old_result[field]:
                raise ValueError(f"candidate/baseline frame population differs: {sid}/{field}")
        print(f"{sid}: {len(emitted)} frames, joint F1={sessions[sid]['stable_windows']['macro_f1']:.4f}, "
              f"rest displayed active={rest_active_display}/{rest_frames}", flush=True)
    if {(row["session"], int(row["emg_end_index"])) for row in rows} != set(old_rows):
        raise ValueError("candidate and baseline frame grids differ")
    if max_hand_marginal_error > 1e-6:
        raise ValueError(f"baseline hand marginal changed: {max_hand_marginal_error}")
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    result = {"status": "exploratory_continuous_same_person_day_replay",
              "protocol_sha256": _sha(PROTOCOL_FILE),
              "signed_trial_results_sha256": _sha(ROOT / "SONG_ARM_SIGNED_RESULTS.json"),
              "baseline_continuous_results_sha256": _sha(ROOT / "SONG_JOINT28_CONTINUOUS_RESULTS.json"),
              "baseline_frame_csv_sha256": _sha(BASELINE_CSV),
              "baseline_model_sha256": baseline.sha256,
              "candidate_arm_model_sha256": _sha(MODEL_FILE),
              "runtime_source_sha256": _sha(RUNTIME_FILE),
              "source_hdf5_sha256": {sid: trial_study["source_hdf5_sha256"][sid]
                                      for sid in ("S01", "S02", "S03", "S04")},
              "frame_csv_sha256": _sha(OUTPUT_CSV), "frame_rows": len(rows),
              "maximum_saved_trial_candidate_arm_probability_error": trial_candidate_error,
              "maximum_baseline_hand_marginal_error": max_hand_marginal_error,
              "sessions": sessions,
              "boundary": "Only cue-labelled stable windows and explicit rest blocks scored; no new-day/person or physical-live claim."}
    OUTPUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    run()
