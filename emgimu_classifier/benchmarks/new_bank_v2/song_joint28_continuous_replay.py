"""Fixed descriptive continuous replay of the already-exported Song 28-state model."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score

from benchmarks.song_real8_study import ARMS, HANDS, parse_label


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
SOURCE = Path("E:/qxy/emg_meta/emg_meta/data/Song")
BUNDLE = REPO / "collection/emg_meta/emg_meta/model_assets/song_joint28_window"
RUNTIME_FILE = REPO / "collection/emg_meta/emg_meta/emgforce/inference/song_joint28_local.py"
PROTOCOL_FILE = ROOT / "SONG_JOINT28_CONTINUOUS_PROTOCOL.json"
OUTPUT_JSON = ROOT / "SONG_JOINT28_CONTINUOUS_RESULTS.json"
OUTPUT_CSV = ROOT / "SONG_JOINT28_CONTINUOUS_FRAMES.csv"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _runtime_classes():
    spec = importlib.util.spec_from_file_location("song_joint28_continuous_runtime", RUNTIME_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SongJoint28WindowRuntime, module.SongJoint28Stream


def _intervals(records, labels: set[str] | None = None, *, trial_kind: str):
    selected = []
    for row in records:
        name = _text(row["label"])
        if (bool(row["valid"]) and _text(row["trial_kind"]) == trial_kind
                and _text(row["completion_status"]) == "completed"
                and (labels is None or name in labels)):
            start, end = int(row["stable_start_sample"]), int(row["stable_end_sample"])
            if start < 0 or end <= start:
                raise ValueError(f"invalid stable interval: {name}")
            selected.append((start, end, int(row["trial_id"]), name))
    return sorted(selected)


def _scores(truth: list[str], pred: list[str], labels: tuple[str, ...]) -> dict:
    if not truth:
        raise ValueError("no scored native 28-state stable intervals")
    return {"n": len(truth), "accuracy": float(accuracy_score(truth, pred)),
            "macro_f1": float(f1_score(truth, pred, labels=list(labels),
                                        average="macro", zero_division=0)),
            "true_support": dict(Counter(truth)), "predicted_support": dict(Counter(pred))}


def run(source: Path = SOURCE, bundle: Path = BUNDLE) -> dict:
    protocol = json.loads(PROTOCOL_FILE.read_text(encoding="utf-8"))
    if (protocol["sessions"] != ["S03", "S04"] or protocol["chunk_emg_samples"] != 25
            or protocol["event_rule"] !=
            "existing live worker: top joint class at probability >= 0.15 for three consecutive emitted frames; no target tuning"):
        raise ValueError("frozen continuous replay protocol changed")
    prior = json.loads((ROOT / "SONG_ARM_CAL_RESULTS.json").read_text(encoding="utf-8"))
    Window, Stream = _runtime_classes()
    model = Window(bundle)
    rows: list[dict] = []
    sessions = {}
    for sid in protocol["sessions"]:
        path = source / f"2026-09-18_{sid}" / "session.h5"
        if _sha(path) != prior["source_hdf5_sha256"][sid]:
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
            raise ValueError(f"invalid native stream shape or index order: {sid}")
        stream = Stream(model)
        emitted = []
        imu_offset = 0
        started = time.perf_counter()
        for left in range(0, len(raw), 25):
            right = min(left + 25, len(raw))
            gap, frames = stream.ingest_emg(raw[left:right], emg_indices[left:right])
            if gap:
                raise ValueError(f"unexpected EMG index gap: {sid}")
            emitted.extend(frames)
            # Recorded IMU indices use the controller's EMG packet-batch boundary.
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
        emitted_count = 0
        candidate = active = None
        candidate_count = 0
        by_label = Counter()
        for end, probability in emitted:
            emitted_count += 1
            peak = int(np.argmax(probability))
            name = model.joint_classes[peak]
            by_label[name] += 1
            threshold_name = name if probability[peak] >= .15 else None
            if threshold_name == candidate:
                candidate_count += 1
            else:
                candidate, candidate_count = threshold_name, 1
            if candidate_count >= 3 and threshold_name != active:
                active = threshold_name
                if active is not None:
                    event_count += 1
                    event_fired = True
                else:
                    event_fired = False
            else:
                event_fired = False
            begin = end - 49
            matching_formal = [interval for interval in formal
                               if interval[0] <= begin and end < interval[1]]
            matching_rest = [interval for interval in rest
                             if interval[0] <= begin and end < interval[1]]
            if len(matching_formal) + len(matching_rest) > 1:
                raise ValueError(f"overlapping scored intervals: {sid}/{end}")
            kind, trial_id, label = "unlabelled", "", ""
            if matching_formal:
                _, _, trial_id, label = matching_formal[0]
                kind = "formal_stable"
                frame_truth.append(label)
                frame_pred.append(name)
                trial_probability[(trial_id, label)].append(probability)
            elif matching_rest:
                _, _, trial_id, label = matching_rest[0]
                kind = "calibration_rest_stable"
                rest_frames += 1
                rest_active_argmax += int(parse_label(name)[1] != "neutral")
                rest_active_display += int(active is not None and parse_label(active)[1] != "neutral")
            if event_fired:
                event_intervals[kind] += 1
            rows.append({"session": sid, "emg_end_index": end,
                         "interval_kind": kind, "trial_id": trial_id, "cue_label": label,
                         "peak_label": name, "peak_probability": float(probability[peak]),
                         "display_label": active or "", "event_fired": int(event_fired),
                         "joint_probabilities": json.dumps(probability.tolist(), separators=(",", ":"))
                         if kind == "formal_stable" else ""})
        trial_truth = [label for _, label in trial_probability]
        trial_pred = [model.joint_classes[int(np.argmax(np.mean(values, axis=0)))]
                      for values in trial_probability.values()]
        scored_trial_ids = {trial_id for trial_id, _ in trial_probability}
        unscored_formal = [{"trial_id": trial_id, "label": label,
                            "stable_samples": end - start}
                           for start, end, trial_id, label in formal
                           if trial_id not in scored_trial_ids]
        hand_truth = [parse_label(value)[1] for value in frame_truth]
        hand_pred = [parse_label(value)[1] for value in frame_pred]
        arm_truth = [parse_label(value)[0] for value in frame_truth]
        arm_pred = [parse_label(value)[0] for value in frame_pred]
        sessions[sid] = {
            "raw_emg_samples": len(raw), "raw_imu_samples": len(imu_indices),
            "replay_cpu_seconds_not_live_latency": duration,
            "emitted_frames": emitted_count, "dropped_frames": stream.dropped_frames,
            "formal_stable_intervals": len(formal), "calibration_rest_stable_intervals": len(rest),
            "scored_formal_trials": len(trial_probability),
            "formal_intervals_without_full_emitted_window": unscored_formal,
            "stable_windows": _scores(frame_truth, frame_pred, model.joint_classes),
            "stable_hand_accuracy": float(accuracy_score(hand_truth, hand_pred)),
            "stable_arm_accuracy": float(accuracy_score(arm_truth, arm_pred)),
            "stable_hand_recall": {name: float(value) for name, value in zip(
                HANDS, recall_score(hand_truth, hand_pred, labels=list(HANDS),
                                    average=None, zero_division=0))},
            "stable_arm_recall": {name: float(value) for name, value in zip(
                ARMS, recall_score(arm_truth, arm_pred, labels=list(ARMS),
                                   average=None, zero_division=0))},
            "trial_mean_joint_probability": _scores(trial_truth, trial_pred, model.joint_classes),
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
        print(f"{sid}: {emitted_count} frames, {len(trial_probability)} scored trials, "
              f"{stream.dropped_frames} dropped; {duration:.1f}s CPU replay", flush=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    result = {"status": "exploratory_continuous_same_person_day_replay",
              "protocol_sha256": _sha(PROTOCOL_FILE),
              "model_sha256": model.sha256,
              "runtime_source_sha256": _sha(RUNTIME_FILE),
              "reference_results_sha256": _sha(ROOT / "SONG_ARM_CAL_RESULTS.json"),
              "source_hdf5_sha256": {sid: prior["source_hdf5_sha256"][sid]
                                      for sid in protocol["sessions"]},
              "frame_csv_sha256": _sha(OUTPUT_CSV), "frame_rows": len(rows),
              "sessions": sessions,
              "boundary": "Only pre-existing cue-labelled stable windows and explicit rest blocks are scored. No physiological onset, live latency, device sync or cross-person/day claim."}
    OUTPUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    run()
