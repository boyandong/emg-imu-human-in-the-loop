"""Read back every saved Song continuous frame and its native cue assignment."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score

from benchmarks.song_real8_study import ARMS, HANDS, parse_label


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
SOURCE = Path("E:/qxy/emg_meta/emg_meta/data/Song")
RESULT = ROOT / "SONG_JOINT28_CONTINUOUS_RESULTS.json"
FRAMES = ROOT / "SONG_JOINT28_CONTINUOUS_FRAMES.csv"
PROTOCOL = ROOT / "SONG_JOINT28_CONTINUOUS_PROTOCOL.json"
MODEL = REPO / "collection/emg_meta/emg_meta/model_assets/song_joint28_window/song_joint28_model.json"
RUNTIME = REPO / "collection/emg_meta/emg_meta/emgforce/inference/song_joint28_local.py"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _intervals(records, kind: str, labels: set[str] | None = None):
    return [(int(row["stable_start_sample"]), int(row["stable_end_sample"]),
             int(row["trial_id"]), _text(row["label"]))
            for row in records if bool(row["valid"])
            and _text(row["completion_status"]) == "completed"
            and _text(row["trial_kind"]) == kind
            and (labels is None or _text(row["label"]) in labels)]


def _verify_session(sid: str, saved: dict, rows: list[dict], classes: tuple[str, ...]) -> dict:
    path = SOURCE / f"2026-09-18_{sid}" / "session.h5"
    with h5py.File(path) as handle:
        raw_count = len(handle["streams/emg/raw"])
        imu_count = len(handle["streams/imu/emg_sample_index"])
        formal = _intervals(handle["trials"][:], "formal")
        rest = _intervals(handle["calibration_blocks"][:], "calibration",
                          {"calibration_rest_initial", "calibration_rest_final"})
    expected_indices = list(range(49, raw_count, 25))
    actual_indices = [int(row["emg_end_index"]) for row in rows]
    if actual_indices != expected_indices:
        raise ValueError(f"emitted stream grid or loss changed: {sid}")
    truth, pred, trials = [], [], defaultdict(list)
    rest_frames = rest_active_peak = rest_active_display = 0
    events = Counter()
    peaks = Counter()
    candidate = active = None
    candidate_count = 0
    for row in rows:
        end = int(row["emg_end_index"])
        begin = end - 49
        matches = [("formal_stable", item) for item in formal
                   if item[0] <= begin and end < item[1]]
        matches += [("calibration_rest_stable", item) for item in rest
                    if item[0] <= begin and end < item[1]]
        if len(matches) > 1:
            raise ValueError(f"overlapping cue intervals: {sid}/{end}")
        kind = matches[0][0] if matches else "unlabelled"
        trial_id = str(matches[0][1][2]) if matches else ""
        label = matches[0][1][3] if matches else ""
        if (row["interval_kind"] != kind or row["trial_id"] != trial_id
                or row["cue_label"] != label):
            raise ValueError(f"native cue assignment changed: {sid}/{end}")
        name = row["peak_label"]
        peak = float(row["peak_probability"])
        if name not in classes or not 0 <= peak <= 1:
            raise ValueError(f"invalid peak row: {sid}/{end}")
        peaks[name] += 1
        threshold_name = name if peak >= .15 else None
        if threshold_name == candidate:
            candidate_count += 1
        else:
            candidate, candidate_count = threshold_name, 1
        fired = False
        if candidate_count >= 3 and threshold_name != active:
            active = threshold_name
            if active is not None:
                fired = True
                events[kind] += 1
        if row["display_label"] != (active or "") or int(row["event_fired"]) != int(fired):
            raise ValueError(f"online display/event replay changed: {sid}/{end}")
        if kind == "formal_stable":
            probability = np.asarray(json.loads(row["joint_probabilities"]), dtype=np.float64)
            if (probability.shape != (28,) or not np.isfinite(probability).all()
                    or np.any(probability < 0) or abs(probability.sum() - 1) > 1e-8
                    or classes[int(np.argmax(probability))] != name
                    or abs(float(probability.max()) - peak) > 1e-12):
                raise ValueError(f"invalid saved 28-way vector: {sid}/{end}")
            truth.append(label)
            pred.append(name)
            trials[(trial_id, label)].append(probability)
        elif row["joint_probabilities"]:
            raise ValueError(f"unscored frame includes a probability vector: {sid}/{end}")
        if kind == "calibration_rest_stable":
            rest_frames += 1
            rest_active_peak += int(parse_label(name)[1] != "neutral")
            rest_active_display += int(active is not None and parse_label(active)[1] != "neutral")
    trial_truth = [label for _, label in trials]
    trial_pred = [classes[int(np.argmax(np.mean(values, axis=0)))] for values in trials.values()]
    hand_truth = [parse_label(value)[1] for value in truth]
    hand_pred = [parse_label(value)[1] for value in pred]
    arm_truth = [parse_label(value)[0] for value in truth]
    arm_pred = [parse_label(value)[0] for value in pred]
    unscored = [{"trial_id": item[2], "label": item[3], "stable_samples": item[1]-item[0]}
                for item in formal if str(item[2]) not in {key[0] for key in trials}]
    checks = {
        "raw_emg_samples": raw_count, "raw_imu_samples": imu_count,
        "emitted_frames": len(rows), "dropped_frames": 0,
        "formal_stable_intervals": len(formal), "calibration_rest_stable_intervals": len(rest),
        "scored_formal_trials": len(trials), "formal_intervals_without_full_emitted_window": unscored,
        "stable_windows": {"n": len(truth), "accuracy": float(accuracy_score(truth, pred)),
                           "macro_f1": float(f1_score(truth, pred, labels=list(classes),
                                                       average="macro", zero_division=0)),
                           "true_support": dict(Counter(truth)),
                           "predicted_support": dict(Counter(pred))},
        "stable_hand_accuracy": float(accuracy_score(hand_truth, hand_pred)),
        "stable_arm_accuracy": float(accuracy_score(arm_truth, arm_pred)),
        "stable_hand_recall": {name: float(value) for name, value in zip(
            HANDS, recall_score(hand_truth, hand_pred, labels=list(HANDS),
                                average=None, zero_division=0))},
        "stable_arm_recall": {name: float(value) for name, value in zip(
            ARMS, recall_score(arm_truth, arm_pred, labels=list(ARMS),
                               average=None, zero_division=0))},
        "trial_mean_joint_probability": {
            "n": len(trials), "accuracy": float(accuracy_score(trial_truth, trial_pred)),
            "macro_f1": float(f1_score(trial_truth, trial_pred, labels=list(classes),
                                        average="macro", zero_division=0)),
            "true_support": dict(Counter(trial_truth)),
            "predicted_support": dict(Counter(trial_pred))},
        "rest_frames": rest_frames, "rest_active_hand_argmax_frames": rest_active_peak,
        "rest_active_hand_argmax_fraction": rest_active_peak/rest_frames,
        "rest_active_hand_display_frames": rest_active_display,
        "rest_active_hand_display_fraction": rest_active_display/rest_frames,
        "continuous_state_transitions_to_non_null": sum(events.values()),
        "state_transitions_by_scored_interval": dict(events),
        "unlabelled_frames": sum(row["interval_kind"] == "unlabelled" for row in rows),
        "all_frame_peak_counts": dict(peaks),
    }
    for key, value in checks.items():
        if isinstance(value, (dict, list)):
            if json.dumps(saved[key], sort_keys=True) != json.dumps(value, sort_keys=True):
                raise ValueError(f"saved {sid}/{key} differs from read-back")
        elif isinstance(value, float):
            if abs(float(saved[key]) - value) > 1e-12:
                raise ValueError(f"saved {sid}/{key} differs from read-back")
        elif saved[key] != value:
            raise ValueError(f"saved {sid}/{key} differs from read-back")
    return {"frames": len(rows), "formal_stable_windows": len(truth),
            "formal_trials": len(trials), "rest_frames": rest_frames,
            "non_null_transitions": sum(events.values())}


def run() -> dict:
    saved = json.loads(RESULT.read_text(encoding="utf-8"))
    model = json.loads(MODEL.read_text(encoding="utf-8"))
    classes = tuple(model["joint_classes"])
    if (saved["protocol_sha256"] != _sha(PROTOCOL)
            or saved["model_sha256"] != _sha(MODEL)
            or saved["runtime_source_sha256"] != _sha(RUNTIME)
            or saved["frame_csv_sha256"] != _sha(FRAMES)):
        raise ValueError("continuous replay source or result digest changed")
    for sid, digest in saved["source_hdf5_sha256"].items():
        if digest != _sha(SOURCE / f"2026-09-18_{sid}" / "session.h5"):
            raise ValueError(f"native recording changed: {sid}")
    with FRAMES.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != saved["frame_rows"]:
        raise ValueError("saved frame count changed")
    verified = {sid: _verify_session(sid, saved["sessions"][sid],
                                     [row for row in rows if row["session"] == sid], classes)
                for sid in ("S03", "S04")}
    audit = {"status": "all_saved_continuous_rows_replayed",
             "results_sha256": _sha(RESULT), "frames_sha256": _sha(FRAMES),
             "sessions": verified, "total_frames": len(rows),
             "scope": "Saved cue assignment, online state rule, rest summaries and trial probability averages; not an independent new recording or latency measurement."}
    (ROOT / "SONG_JOINT28_CONTINUOUS_VERIFICATION.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


if __name__ == "__main__":
    print(json.dumps(run(), indent=2), flush=True)
