"""Descriptive analysis of manually annotated live diagnostic captures."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import h5py


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _signal_profile(handle: h5py.File, sample_rate: int) -> dict:
    raw = handle["emg/raw"]
    count = len(raw)
    squared = np.zeros(8, dtype=np.float64)
    zeros = np.zeros(8, dtype=np.int64)
    saturation = np.zeros(8, dtype=np.int64)
    minimum = np.full(8, np.inf)
    maximum = np.full(8, -np.inf)
    flat_windows = np.zeros(8, dtype=np.int64)
    full_windows = 0
    chunk_size = max(sample_rate * 60, sample_rate)
    for start in range(0, count, chunk_size):
        values = raw[start:start + chunk_size].astype(np.float64)
        squared += np.square(values).sum(axis=0)
        zeros += np.count_nonzero(values == 0, axis=0)
        saturation += np.count_nonzero(np.abs(values) >= 8_300_000, axis=0)
        minimum = np.minimum(minimum, values.min(axis=0))
        maximum = np.maximum(maximum, values.max(axis=0))
        complete = len(values) // sample_rate
        if complete:
            windows = values[:complete * sample_rate].reshape(complete, sample_rate, 8)
            flat_windows += np.count_nonzero(np.ptp(windows, axis=1) == 0, axis=0)
            full_windows += complete
    received = handle["emg/received_ns"][:]
    valid_stamps = received[received >= 0]
    duration = ((int(valid_stamps[-1]) - int(valid_stamps[0])) / 1e9
                if len(valid_stamps) > 1 else 0.0)
    return {
        "raw_unit": "device integer values; no physical-unit conversion inferred",
        "rms_per_channel": (np.sqrt(squared / count).tolist() if count else None),
        "peak_to_peak_per_channel": ((maximum - minimum).tolist() if count else None),
        "zero_fraction_per_channel": ((zeros / count).tolist() if count else None),
        "near_adc_limit_fraction_per_channel": ((saturation / count).tolist() if count else None),
        "flat_one_second_windows_per_channel": flat_windows.tolist(),
        "complete_one_second_windows": full_windows,
        "received_rate_hz_approx": ((len(valid_stamps) - 1) / duration if duration > 0 else None),
    }


def analyze(directory: Path) -> dict:
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "emgforce_live_diagnostic_v1" or manifest.get("status") != "closed":
        raise ValueError("diagnostic capture must be complete before analysis")
    files = manifest.get("file_sha256", {})
    for name in ("signals.h5", "predictions.csv", "manual_annotations.csv"):
        if files.get(name) != _sha256(directory / name):
            raise ValueError(f"diagnostic capture file hash mismatch: {name}")
    labels = tuple(manifest["labels"])
    prediction_rows = _rows(directory / "predictions.csv")
    annotations = _rows(directory / "manual_annotations.csv")
    if (len(prediction_rows) != manifest["predictions"] or
            len(annotations) != manifest["manual_annotations"]):
        raise ValueError("diagnostic capture row count differs from manifest")
    with h5py.File(directory / "signals.h5", "r") as handle:
        raw_count = len(handle["emg/raw"])
        if (handle["emg/raw"].shape != (raw_count, 8) or
                len(handle["emg/sample_index"]) != raw_count or
                len(handle["emg/received_ns"]) != raw_count or
                raw_count != manifest["raw_samples"]):
            raise ValueError("diagnostic EMG arrays or counts are inconsistent")
        sample_indices = handle["emg/sample_index"][:]
        if len(sample_indices) > 1 and np.any(np.diff(sample_indices) <= 0):
            raise ValueError("diagnostic EMG sample indices do not increase")
        imu_count = len(handle["imu/gyro_rad_s"])
        if (handle["imu/gyro_rad_s"].shape != (imu_count, 3) or
                handle["imu/accel_m_s2"].shape != (imu_count, 3) or
                len(handle["imu/received_ns"]) != imu_count or
                imu_count != manifest["imu_samples"]):
            raise ValueError("diagnostic IMU arrays or counts are inconsistent")
        signal_profile = _signal_profile(handle, int(manifest["sample_rate_hz"]))
        imu_nonfinite_samples = 0
        for start in range(0, imu_count, 60_000):
            gyro = handle["imu/gyro_rad_s"][start:start + 60_000]
            accel = handle["imu/accel_m_s2"][start:start + 60_000]
            imu_nonfinite_samples += int(np.count_nonzero(
                ~np.isfinite(gyro).all(axis=1) | ~np.isfinite(accel).all(axis=1)))
    predictions = []
    for row in prediction_rows:
        values = np.asarray([float(row[f"p_{label}"]) for label in labels])
        if (values.shape != (len(labels),) or not np.isfinite(values).all() or
                row["peak_label"] != labels[int(np.argmax(values))]):
            raise ValueError("saved prediction schema or peak label is inconsistent")
        predictions.append((int(row["output_sample_index"]), row["peak_label"],
                            row["active_label"], values))
    outside_raw = int(sum(bool(not len(sample_indices) or index < sample_indices[0] or
                               index > sample_indices[-1]) for index, _, _, _ in predictions))
    intervals = []
    pending = None
    for row in annotations:
        index = int(row["latest_emg_sample_index"])
        action = row["action"]
        if action not in labels:
            raise ValueError("annotation label is absent from model labels")
        if row["event"] == "start":
            if pending is not None:
                raise ValueError("manual annotation intervals overlap")
            pending = (action, index)
        elif row["event"] == "end":
            if pending is None or pending[0] != action or index <= pending[1]:
                raise ValueError("manual annotation end lacks a matching later start")
            intervals.append((action, pending[1], index))
            pending = None
        else:
            raise ValueError("unknown manual annotation event")
    # An unfinished final annotation is preserved in the CSV, not silently scored.
    by_action = defaultdict(list)
    scored = []
    sample_rate_hz = int(manifest["sample_rate_hz"])
    active_labels = set(labels) - {"neutral"}
    for action, start, end in intervals:
        frames = [(index, peak, active, values) for index, peak, active, values in predictions
                  if start <= index <= end]
        pre_frames = [(index, active) for index, _, active, _ in predictions
                      if start - int(0.4 * sample_rate_hz) <= index < start]
        pre_active = sum(active in active_labels for _, active in pre_frames)
        if not frames:
            scored.append({"action": action, "start_sample_index": start,
                           "end_sample_index": end, "prediction_frames": 0,
                           "pre_start_frames": len(pre_frames),
                           "pre_start_active_frames": pre_active,
                           "pre_start_any_active": bool(pre_active) if pre_frames else None})
            continue
        target_column = labels.index(action)
        matching_display = [index for index, _, active, _ in frames if active == action]
        row = {
            "action": action, "start_sample_index": start, "end_sample_index": end,
            "prediction_frames": len(frames),
            "peak_agreement_fraction": float(np.mean([peak == action for _, peak, _, _ in frames])),
            "display_agreement_fraction": float(np.mean([active == action for _, _, active, _ in frames])),
            "mean_target_probability": float(np.mean([values[target_column] for _, _, _, values in frames])),
            "ever_display_match": bool(matching_display),
            "endpoint_display_match": frames[-1][2] == action,
            "first_display_match_after_start_seconds": (
                (matching_display[0] - start) / sample_rate_hz
                if matching_display and action != "neutral" else None),
            "pre_start_frames": len(pre_frames),
            "pre_start_active_frames": pre_active,
            "pre_start_any_active": bool(pre_active) if pre_frames else None,
        }
        by_action[action].append(row)
        scored.append(row)
    counts = Counter(peak for _, peak, _, _ in predictions)
    by_action_rows = [row for row in scored if row["prediction_frames"] > 0]
    summary = {
        "schema": "emgforce_live_diagnostic_analysis_v1",
        "capture_directory": str(directory.resolve()),
        "model_sha256": manifest["model_sha256"],
        "prediction_frames": len(predictions),
        "prediction_frames_outside_captured_raw": outside_raw,
        "raw_emg_samples_verified": raw_count,
        "raw_imu_samples_verified": imu_count,
        "imu_nonfinite_samples": imu_nonfinite_samples,
        "raw_signal_profile": signal_profile,
        "sample_index_gap_edges": int(np.count_nonzero(np.diff(sample_indices) > 1)),
        "reported_lost_packets": int(manifest["reported_lost_packets"]),
        "file_hashes_verified": True,
        "manual_intervals": len(intervals),
        "event_intervals_with_predictions": len(by_action_rows),
        "event_intervals_ever_display_match": sum(row["ever_display_match"] for row in by_action_rows),
        "event_intervals_endpoint_display_match": sum(row["endpoint_display_match"] for row in by_action_rows),
        "pre_start_intervals_with_frames": sum(row["pre_start_frames"] > 0 for row in scored),
        "pre_start_intervals_any_active": sum(row["pre_start_any_active"] is True for row in scored),
        "unfinished_manual_start": pending is not None,
        "peak_label_counts": {label: counts[label] for label in labels},
        "peak_label_fractions": {label: counts[label] / len(predictions) if predictions else None
                                 for label in labels},
        "annotated_action_summary": {
            action: {"intervals_with_predictions": len(rows),
                     "mean_peak_agreement": float(np.mean([row["peak_agreement_fraction"] for row in rows])),
                     "mean_display_agreement": float(np.mean([row["display_agreement_fraction"] for row in rows])),
                     "mean_target_probability": float(np.mean([row["mean_target_probability"] for row in rows])),
                     "intervals_ever_display_match": sum(row["ever_display_match"] for row in rows),
                     "intervals_endpoint_display_match": sum(row["endpoint_display_match"] for row in rows),
                     "median_first_display_match_after_start_seconds": (
                         float(np.median([row["first_display_match_after_start_seconds"]
                                          for row in rows if row["first_display_match_after_start_seconds"] is not None]))
                         if any(row["first_display_match_after_start_seconds"] is not None for row in rows)
                         else None)}
            for action, rows in by_action.items()},
        "intervals": scored,
        "scope": "Manual keypress intervals express intended actions, not measured physiological onset/offset; event matches are diagnostic agreement, not formal recognition accuracy. First-match seconds use nominal sample indices after a manual marker, not measured muscle or screen latency. Pre-start activity uses the final nominal 400 ms before the marker and may contain previous movement. Raw signal metrics are descriptive across movements, not a rest-calibration pass/fail. Packet loss can occur without a sample-index gap because indices count received samples. Predictions outside the captured raw range can occur when recording starts while inference is already running.",
    }
    (directory / "analysis.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                              encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.capture_directory), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
