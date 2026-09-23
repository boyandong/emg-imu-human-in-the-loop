"""Exploratory, split-locked study of the user's four Song HDF5 v3 sessions.

Raw recordings stay at the user-supplied path. This is offline cue-labelled
classification, not an online latency or deployment validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, sosfilt, tf2sos
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.families import BodyContextFamily, LocalDetailFamily, ScalePatternFamily

ROLES = {"S01": "train", "S02": "train", "S03": "val", "S04": "test"}
HANDS = ("neutral", "index_pinch", "fist", "open_hand")
ARMS = ("still", "up", "down", "left", "right", "forward", "backward")
CALIBRATION_BLOCKS = {
    "calibration_rest_initial": ("neutral", 1), "calibration_rest_final": ("neutral", 2),
    "calibration_pinch_1": ("index_pinch", 1), "calibration_pinch_2": ("index_pinch", 2),
    "calibration_fist_1": ("fist", 1), "calibration_fist_2": ("fist", 2),
    "calibration_open_1": ("open_hand", 1), "calibration_open_2": ("open_hand", 2),
}
WINDOW = 50  # 200 ms at 250 Hz
IMU_WINDOW = 22  # ~196 ms at 112 Hz, ending no later than EMG window


def _text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def window_starts(start: int, end: int, width: int = WINDOW) -> np.ndarray:
    """At most three non-overlapping, evenly located windows per stable trial."""
    count = min(3, (end - start) // width)
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    return np.linspace(start, end - width, count, dtype=np.int64)


def parse_label(value: str) -> tuple[str, str]:
    for arm in ARMS:
        prefix = arm + "_"
        if value.startswith(prefix) and value[len(prefix):] in HANDS:
            return arm, value[len(prefix):]
    raise ValueError(f"unknown 28-state label: {value}")


def _filter_emg(raw: np.ndarray, mode: str = "zero_phase") -> np.ndarray:
    # Fixed filters and no target-session fitted state. Causal mode carries
    # filter memory across the continuous recorded session.
    x = np.asarray(raw, dtype=np.float64)
    if mode not in ("zero_phase", "causal"):
        raise ValueError(f"unknown EMG filter mode: {mode}")
    if mode == "causal":
        x = sosfilt(butter(4, 40.0, btype="highpass", fs=250.0, output="sos"), x, axis=0)
    else:
        b, a = butter(4, 40.0, btype="highpass", fs=250.0)
        x = filtfilt(b, a, x, axis=0)
    for hz in (50.0, 100.0):
        b, a = iirnotch(hz, Q=30.0, fs=250.0)
        x = sosfilt(tf2sos(b, a), x, axis=0) if mode == "causal" else filtfilt(b, a, x, axis=0)
    return x.astype(np.float32)


def load_session(folder: Path, expected_id: str, filter_mode: str = "zero_phase"):
    path = folder / "session.h5"
    readiness = json.loads((folder / "SESSION_COLLECTION_READINESS.json").read_text(encoding="utf-8"))
    sha = _hash(path)
    if sha != readiness["hdf5_sha256"]:
        raise ValueError(f"source hash mismatch: {expected_id}")
    with h5py.File(path) as f:
        meta = f["meta"].attrs
        if (_text(meta["schema_version"]) != "3.0" or
            _text(meta["protocol_version"]) != "jilv_music_28_v2" or
            _text(meta["session_id"]) != expected_id or
            _text(meta["dataset_split"]) != ROLES[expected_id] or
            int(meta["emg_nominal_rate_hz"]) != 250 or
            int(meta["imu_nominal_rate_hz"]) != 112 or
            int(meta["num_emg_channels"]) != 8):
            raise ValueError(f"metadata/split mismatch: {expected_id}")
        raw = f["streams/emg/raw"][:]
        if raw.ndim != 2 or raw.shape[1] != 8:
            raise ValueError(f"wrong EMG shape: {expected_id}")
        indices = f["streams/emg/sample_index"][:]
        if not np.array_equal(indices, np.arange(len(raw))):
            raise ValueError(f"noncontiguous EMG indices: {expected_id}")
        imu_indices = f["streams/imu/emg_sample_index"][:]
        if np.any(np.diff(imu_indices) < 0):
            raise ValueError(f"nonmonotone IMU alignment: {expected_id}")
        imu = np.column_stack((f["streams/imu/accel"][:], f["streams/imu/gyro"][:]))
        trials = f["trials"][:]
        calibration_rows = f["calibration_blocks"][:]
    print(f"{expected_id}: filter {len(raw)} EMG samples, {len(imu)} IMU samples", flush=True)
    emg = _filter_emg(raw, filter_mode)
    windows, imus, trial_keys, hands, composite, exclusions = [], [], [], [], [], Counter()
    valid_trial_ids = set()
    for row in trials:
        if _text(row["trial_kind"]) != "formal":
            continue
        if not bool(row["valid"]) or _text(row["completion_status"]) != "completed":
            exclusions["invalid_or_incomplete"] += 1
            continue
        start, end = int(row["stable_start_sample"]), int(row["stable_end_sample"])
        if not (0 <= start < end <= len(emg)):
            exclusions["invalid_stable_interval"] += 1
            continue
        label = _text(row["label"])
        _, hand = parse_label(label)
        trial_id = int(row["trial_id"])
        if trial_id in valid_trial_ids:
            raise ValueError(f"duplicate formal trial ID: {expected_id}/{trial_id}")
        valid_trial_ids.add(trial_id)
        starts = window_starts(start, end)
        if not len(starts):
            exclusions["stable_interval_under_200ms"] += 1
            continue
        for left in starts:
            right = int(left) + WINDOW
            last_imu = int(np.searchsorted(imu_indices, right - 1, side="right"))
            if last_imu < IMU_WINDOW:
                exclusions["imu_history_unavailable_window"] += 1
                continue
            windows.append(emg[left:right])
            imus.append(imu[last_imu - IMU_WINDOW:last_imu])
            trial_keys.append(f"{expected_id}:{trial_id}")
            hands.append(hand)
            composite.append(label)
    if not windows:
        raise ValueError(f"no usable windows: {expected_id}")
    first_formal_start = min(int(row["trial_start_sample"]) for row in trials
                             if _text(row["trial_kind"]) == "formal")
    cal_windows, cal_hands, cal_shots = [], [], []
    seen_calibration = set()
    selected_calibration_intervals = []
    for row in calibration_rows:
        name = _text(row["label"])
        if name not in CALIBRATION_BLOCKS:
            continue
        if name in seen_calibration or not bool(row["valid"]) or _text(row["completion_status"]) != "completed":
            raise ValueError(f"missing/duplicate/invalid calibration block: {expected_id}/{name}")
        seen_calibration.add(name)
        start, end = int(row["stable_start_sample"]), int(row["stable_end_sample"])
        if not (0 <= start < end <= first_formal_start <= len(emg)):
            raise ValueError(f"invalid calibration interval: {expected_id}/{name}")
        hand, shot = CALIBRATION_BLOCKS[name]
        selected_calibration_intervals.append((int(row["trial_start_sample"]),
                                               int(row["trial_end_sample"]), shot))
        starts = window_starts(start, end)
        if not len(starts):
            raise ValueError(f"calibration shorter than one window: {expected_id}/{name}")
        for left in starts:
            cal_windows.append(emg[left:int(left) + WINDOW])
            cal_hands.append(hand)
            cal_shots.append(shot)
    if seen_calibration != set(CALIBRATION_BLOCKS):
        raise ValueError(f"incomplete calibration protocol: {expected_id}")
    return {
        "batch": FeatureBatch(np.stack(windows), 250.0, np.stack(imus)),
        "trial": np.asarray(trial_keys), "hand": np.asarray(hands),
        "composite": np.asarray(composite),
        "calibration_batch": FeatureBatch(np.stack(cal_windows), 250.0),
        "calibration_hand": np.asarray(cal_hands),
        "calibration_shot": np.asarray(cal_shots),
        "calibration_elapsed_seconds": {
            str(budget): (max(end for _, end, shot in selected_calibration_intervals if shot <= budget)
                          - min(start for start, _, shot in selected_calibration_intervals if shot <= budget)) / 250.0
            for budget in (1, 2)},
        "audit": {"session": expected_id, "role": ROLES[expected_id], "sha256": sha,
                  "readiness": readiness["status"], "formal_trials": int(sum(_text(r["trial_kind"]) == "formal" for r in trials)),
                  "usable_trials": len(set(trial_keys)), "windows": len(windows),
                  "exclusions": dict(exclusions)},
    }


def _join_batches(sessions):
    return FeatureBatch(np.concatenate([s["batch"].emg for s in sessions]), 250.0,
                        np.concatenate([s["batch"].imu for s in sessions]))


def _trial_metrics(y, probabilities, trials, classes):
    true, pred = [], []
    for trial in dict.fromkeys(trials):
        idx = np.flatnonzero(trials == trial)
        if len(set(y[idx])) != 1:
            raise ValueError(f"inconsistent within-trial labels: {trial}")
        true.append(y[idx[0]])
        pred.append(classes[int(np.argmax(probabilities[idx].mean(axis=0)))])
    labels = list(classes)
    return {"trials": len(true), "accuracy": float(accuracy_score(true, pred)),
            "macro_f1": float(f1_score(true, pred, labels=labels, average="macro", zero_division=0)),
            "recall": {label: float(value) for label, value in zip(labels, recall_score(true, pred, labels=labels, average=None, zero_division=0))},
            "confusion_matrix": confusion_matrix(true, pred, labels=labels).tolist(), "labels": labels,
            "true_support": dict(Counter(true)), "predicted_support": dict(Counter(pred))}


def run(root: Path, filter_mode: str = "zero_phase"):
    data = {}
    for session_id in ROLES:
        data[session_id] = load_session(root / f"2026-09-18_{session_id}", session_id, filter_mode)
    train = _join_batches([data["S01"], data["S02"]])
    features = {}
    for family_id, family in (("F0", LocalDetailFamily()), ("F1", ScalePatternFamily()), ("F6_imu", BodyContextFamily())):
        print(f"fit source feature family {family_id}", flush=True)
        family.fit(train)
        features[family_id] = {sid: family.transform(s["batch"]) for sid, s in data.items()}
    configurations = {"F0": ("F0",), "F0_F1": ("F0", "F1"), "F0_F6imu": ("F0", "F6_imu")}
    validation = {}
    trained = {}
    for name, families in configurations.items():
        print(f"validation: {name}", flush=True)
        matrix = {sid: np.concatenate([features[f][sid] for f in families], axis=1) for sid in data}
        x_train = np.concatenate([matrix["S01"], matrix["S02"]])
        y_train = np.concatenate([data[sid]["hand"] for sid in ("S01", "S02")])
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=0))
        model.fit(x_train, y_train)
        classes = model[-1].classes_
        p_val = model.predict_proba(matrix["S03"])
        validation[name] = _trial_metrics(data["S03"]["hand"], p_val, data["S03"]["trial"], classes)
        trained[name] = (model, matrix)
    selected = max(configurations, key=lambda key: (validation[key]["macro_f1"], key == "F0"))
    model, matrix = trained[selected]
    print(f"frozen selection {selected}; final test S04", flush=True)
    test = _trial_metrics(data["S04"]["hand"], model.predict_proba(matrix["S04"]), data["S04"]["trial"], model[-1].classes_)
    # Secondary 28-state endpoint: fixed F0+IMU design, with no test-based choice.
    full_matrix = trained["F0_F6imu"][1]
    full_train_y = np.concatenate([data[sid]["composite"] for sid in ("S01", "S02")])
    full_model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=0))
    full_model.fit(np.concatenate([full_matrix["S01"], full_matrix["S02"]]), full_train_y)
    full_labels = full_model[-1].classes_
    full_validation = _trial_metrics(data["S03"]["composite"], full_model.predict_proba(full_matrix["S03"]), data["S03"]["trial"], full_labels)
    full_test = _trial_metrics(data["S04"]["composite"], full_model.predict_proba(full_matrix["S04"]), data["S04"]["trial"], full_labels)
    return {"status": "exploratory_single_participant_not_deployment_validated", "protocol": "jilv_music_28_v2",
            "filter_mode": filter_mode,
            "split": ROLES, "window_samples": WINDOW, "imu_window_samples": IMU_WINDOW,
            "preprocessing": f"40Hz 4th-order {filter_mode} high-pass, 50/100Hz Q30 notch on continuous sessions; no target-global centering; source-only fitted feature thresholds and StandardScaler",
            "selection_rule": "best S03 trial-level 4-hand macro-F1; fixed C=1 balanced multinomial logistic; S04 viewed once after selection",
            "source_audit": [data[sid]["audit"] for sid in ROLES], "validation": validation,
            "selected": selected, "test": test,
            "secondary_28_state": {"features": "F0_F6imu", "validation": full_validation, "test": full_test,
                                   "note": "fixed secondary design; 28 labels = 7 arm postures x 4 hand states"},
            "limitations": ["one participant and one collection date", "S01-S03 collection readiness failed; valid trials used only for exploratory work", "S04 readiness passed", "cue labels are instructions, not verified physiological onset", "stable cued trial windows are not continuous online recognition; no measured end-to-end latency", "no clinical or cross-participant generalization claim"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--filter-mode", choices=("zero_phase", "causal"), default="zero_phase")
    args = parser.parse_args()
    result = run(args.source, args.filter_mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
