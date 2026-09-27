"""Pair the original and signed Song models on one recorded diagnostic stream.

Manual keypresses are intended-action cues, not measured physiological labels.
The replay reconstructs a deterministic indexed stream; host packet arrival order
and filter state before capture start are not stored in the diagnostic format.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from .analyze_live_diagnostic import JOINT_ARMS, JOINT_HANDS, _joint28_parts, analyze
from .song_joint28_local import SongJoint28Stream, SongJoint28WindowRuntime


ASSETS = Path(__file__).resolve().parents[2] / "model_assets"
BASELINE = ASSETS / "song_joint28_window"
SIGNED = ASSETS / "song_joint28_signed"
MARGIN_SECONDS = 0.4


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _replay(model: SongJoint28WindowRuntime, raw: np.ndarray, emg_index: np.ndarray,
            accel: np.ndarray, gyro: np.ndarray,
            imu_index: np.ndarray) -> tuple[dict[int, np.ndarray], int]:
    stream = SongJoint28Stream(model)
    emitted: dict[int, np.ndarray] = {}
    imu_offset = 0
    for left in range(0, len(raw), 25):
        right = min(left + 25, len(raw))
        _, frames = stream.ingest_emg(raw[left:right], emg_index[left:right])
        for index, probability in frames:
            if index in emitted:
                raise ValueError("duplicate replay frame index")
            emitted[index] = probability
        watermark = int(emg_index[right - 1]) + 1
        last = int(np.searchsorted(imu_index, watermark, side="right"))
        for index, probability in stream.ingest_imu(
                accel[imu_offset:last], gyro[imu_offset:last], imu_index[imu_offset:last]):
            if index in emitted:
                raise ValueError("duplicate replay frame index")
            emitted[index] = probability
        imu_offset = last
    if imu_offset < len(imu_index):
        for index, probability in stream.ingest_imu(
                accel[imu_offset:], gyro[imu_offset:], imu_index[imu_offset:]):
            if index in emitted:
                raise ValueError("duplicate replay frame index")
            emitted[index] = probability
    return emitted, stream.dropped_frames


def _score(truth: list[str], predicted: list[str], *, labels: list[str]) -> dict:
    if not truth:
        return {"n": 0, "accuracy": None, "observed_label_macro_f1": None,
                "support": {}, "recall": {}}
    observed = sorted(set(truth))
    return {"n": len(truth), "accuracy": float(accuracy_score(truth, predicted)),
            "observed_label_macro_f1": float(f1_score(
                truth, predicted, labels=observed, average="macro", zero_division=0)),
            "support": dict(Counter(truth)),
            "recall": {name: float(sum(t == name and p == name for t, p in zip(truth, predicted))
                                   / sum(t == name for t in truth)) for name in observed},
            "label_space": labels}


def run(directory: Path, baseline_dir: Path = BASELINE, signed_dir: Path = SIGNED) -> dict:
    directory = Path(directory)
    # This verifies the capture's three hashed files and parses manual intervals.
    capture_analysis = analyze(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    base = SongJoint28WindowRuntime(baseline_dir)
    signed = SongJoint28WindowRuntime(signed_dir)
    expected_labels = {f"{arm}_{hand}" for arm in JOINT_ARMS for hand in JOINT_HANDS}
    if (manifest["schema"] != "emgforce_live_diagnostic_v2"
            or manifest["sample_rate_hz"] != 250
            or set(manifest["labels"]) != expected_labels
            or len(manifest["labels"]) != 28
            or manifest["model_sha256"] not in {base.sha256, signed.sha256}
            or set(base.joint_classes) != expected_labels
            or base.joint_classes != signed.joint_classes):
        raise ValueError("capture must use one of the two native Song 28-state models at 250 Hz")
    if capture_analysis["unfinished_manual_start"]:
        raise ValueError("finish or remove the unmatched manual action start before pairing")
    with h5py.File(directory / "signals.h5", "r") as handle:
        raw = handle["emg/raw"][:]
        emg_index = handle["emg/sample_index"][:]
        accel = handle["imu/accel_m_s2"][:]
        gyro = handle["imu/gyro_rad_s"][:]
        imu_index = handle["imu/emg_sample_index"][:]
    if (not len(raw) or len(accel) < 22 or np.any(imu_index < 0)
            or np.any(np.diff(imu_index) < 0)
            or not np.isfinite(accel).all() or not np.isfinite(gyro).all()):
        raise ValueError("capture lacks complete finite indexed EMG/IMU for paired replay")
    if not np.all(np.diff(emg_index) > 0):
        raise ValueError("capture EMG indices must strictly increase")
    interval_inputs = []
    first, last = int(emg_index[0]), int(emg_index[-1])
    for ordinal, row in enumerate(capture_analysis["intervals"], start=1):
        start, end = int(row["start_sample_index"]), int(row["end_sample_index"])
        if not first <= start < end <= last:
            raise ValueError("manual action interval lies outside captured EMG")
        interval_inputs.append((ordinal, row["action"], start, end))
    streams = {"baseline": _replay(base, raw, emg_index, accel, gyro, imu_index),
               "signed": _replay(signed, raw, emg_index, accel, gyro, imu_index)}
    frames = {name: value[0] for name, value in streams.items()}
    if set(frames["baseline"]) != set(frames["signed"]):
        raise ValueError("paired models did not emit the same indexed frame grid")
    if not frames["baseline"]:
        raise ValueError("capture produced no complete aligned model frame")
    margin = round(MARGIN_SECONDS * 250)
    assigned: dict[int, tuple[int, str]] = {}
    intervals = []
    for ordinal, label, start, end in interval_inputs:
        selected = [index for index in frames["baseline"]
                    if index - 49 >= start + margin and index < end - margin]
        for index in selected:
            if index in assigned:
                raise ValueError("manual stable-core intervals overlap")
            assigned[index] = (ordinal, label)
        intervals.append({"ordinal": ordinal, "cue_label": label, "start_sample_index": start,
                          "end_sample_index": end, "stable_core_margin_seconds_each_end": MARGIN_SECONDS,
                          "scored_frames": len(selected), "reason_if_unscored": (
                              None if selected else "no full aligned 50-sample window after both margins")})
    rows = []
    grouped: dict[int, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    by_model = {name: {"truth": [], "joint": [], "hand": [], "arm": []} for name in frames}
    for index in sorted(frames["baseline"]):
        annotation = assigned.get(index)
        row = {"emg_end_index": index,
               "manual_interval": annotation[0] if annotation else "",
               "cue_label": annotation[1] if annotation else ""}
        for name, probabilities in frames.items():
            p = probabilities[index]
            if p.shape != (28,) or not np.isfinite(p).all() or not np.isclose(p.sum(), 1, atol=1e-10):
                raise ValueError("replay produced invalid 28-state probabilities")
            label = base.joint_classes[int(np.argmax(p))]
            row[f"{name}_peak"] = label
            row[f"{name}_peak_probability"] = float(np.max(p))
            row[f"{name}_probabilities"] = (json.dumps(p.tolist(), separators=(",", ":"))
                                            if annotation else "")
            if annotation:
                grouped[annotation[0]][name].append(p)
                by_model[name]["truth"].append(annotation[1])
                by_model[name]["joint"].append(label)
                by_model[name]["arm"].append(_joint28_parts(label)[0])
                by_model[name]["hand"].append(_joint28_parts(label)[1])
        rows.append(row)
    frame_scores = {}
    trial_scores = {}
    decisions: dict[str, list[str]] = {name: [] for name in frames}
    trial_truth = []
    for interval in intervals:
        ordinal = interval["ordinal"]
        if not interval["scored_frames"]:
            continue
        trial_truth.append(interval["cue_label"])
        for name in frames:
            p = np.mean(grouped[ordinal][name], axis=0)
            decision = base.joint_classes[int(np.argmax(p))]
            decisions[name].append(decision)
            interval[f"{name}_mean_probability_decision"] = decision
            interval[f"{name}_cue_probability"] = float(p[base.joint_classes.index(interval["cue_label"])])
    for name, items in by_model.items():
        truth = items["truth"]
        frame_scores[name] = {
            "joint": _score(truth, items["joint"], labels=list(base.joint_classes)),
            "hand": _score([_joint28_parts(value)[1] for value in truth], items["hand"],
                           labels=list(JOINT_HANDS)),
            "arm": _score([_joint28_parts(value)[0] for value in truth], items["arm"],
                          labels=list(JOINT_ARMS)),
        }
        trial_scores[name] = _score(trial_truth, decisions[name], labels=list(base.joint_classes))
    corrected = sum(a != truth and b == truth for truth, a, b in
                    zip(trial_truth, decisions["baseline"], decisions["signed"]))
    new_errors = sum(a == truth and b != truth for truth, a, b in
                     zip(trial_truth, decisions["baseline"], decisions["signed"]))
    csv_path = directory / "paired_song28_frames.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    result = {"schema": "emgforce_song28_paired_diagnostic_v1",
              "capture_manifest_sha256": _sha(directory / "manifest.json"),
              "capture_file_sha256": manifest["file_sha256"],
              "model_sha256": {"baseline": base.sha256, "signed": signed.sha256},
              "recorded_model_sha256": manifest["model_sha256"],
              "replay_policy": "same indexed raw stream, 25-sample canonical chunks and IMU watermark; filters reset at capture start",
              "frame_csv_sha256": _sha(csv_path),
              "emitted_common_frames": len(rows),
              "dropped_frames": {name: value[1] for name, value in streams.items()},
              "manual_intervals": len(intervals),
              "scored_manual_stable_core_intervals": len(trial_truth),
              "intervals": intervals, "frame_scores": frame_scores,
              "trial_scores": trial_scores,
              "signed_corrected_baseline_trial_errors": corrected,
              "signed_new_trial_errors": new_errors,
              "scope": "Manual cue keypresses and 0.4-second endpoint margins are not measured muscle onset/offset. These are paired intended-action agreement scores, not formal device accuracy. Reconstructed packet ordering and reset filter state cannot prove equivalence to the original live display or hardware latency. Unlabelled frames have no error denominator."}
    (directory / "paired_song28_analysis.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    verify(directory, baseline_dir=baseline_dir, signed_dir=signed_dir)
    return result


def verify(directory: Path, baseline_dir: Path = BASELINE, signed_dir: Path = SIGNED) -> dict:
    """Recompute paired label metrics from the persisted frame and interval tables."""
    directory = Path(directory)
    path = directory / "paired_song28_analysis.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    baseline_model = SongJoint28WindowRuntime(baseline_dir)
    signed_model = SongJoint28WindowRuntime(signed_dir)
    model_classes = baseline_model.joint_classes
    if result.get("schema") != "emgforce_song28_paired_diagnostic_v1":
        raise ValueError("unknown paired diagnostic schema")
    if result["capture_manifest_sha256"] != _sha(directory / "manifest.json"):
        raise ValueError("paired diagnostic manifest hash mismatch")
    if result["frame_csv_sha256"] != _sha(directory / "paired_song28_frames.csv"):
        raise ValueError("paired diagnostic frame hash mismatch")
    for name, digest in result["capture_file_sha256"].items():
        if _sha(directory / name) != digest:
            raise ValueError(f"paired diagnostic source hash mismatch: {name}")
    if result["model_sha256"] != {
            "baseline": baseline_model.sha256,
            "signed": signed_model.sha256}:
        raise ValueError("paired diagnostic model hash mismatch")
    with (directory / "paired_song28_frames.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != result["emitted_common_frames"]:
        raise ValueError("paired diagnostic frame count mismatch")
    seen = set()
    by_model = {name: {"truth": [], "joint": [], "hand": [], "arm": []}
                for name in ("baseline", "signed")}
    by_interval: dict[int, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        index = int(row["emg_end_index"])
        if index in seen:
            raise ValueError("paired diagnostic duplicate frame index")
        seen.add(index)
        for name in ("baseline", "signed"):
            if row[f"{name}_peak"] not in model_classes:
                raise ValueError("paired diagnostic unknown peak label")
            if row["manual_interval"]:
                p = np.asarray(json.loads(row[f"{name}_probabilities"]), dtype=np.float64)
                if (p.shape != (28,) or not np.isfinite(p).all()
                        or not np.isclose(p.sum(), 1, atol=1e-10)
                        or row[f"{name}_peak"] != model_classes[int(np.argmax(p))]
                        or not np.isclose(float(row[f"{name}_peak_probability"]), float(p.max()), atol=1e-12)):
                    raise ValueError("paired diagnostic frame probability mismatch")
            elif row[f"{name}_probabilities"]:
                raise ValueError("unlabelled paired frame unexpectedly stores probabilities")
        if row["manual_interval"]:
            ordinal = int(row["manual_interval"])
            truth = row["cue_label"]
            for name in ("baseline", "signed"):
                peak = row[f"{name}_peak"]
                by_model[name]["truth"].append(truth)
                by_model[name]["joint"].append(peak)
                by_model[name]["hand"].append(_joint28_parts(peak)[1])
                by_model[name]["arm"].append(_joint28_parts(peak)[0])
                by_interval[ordinal][name].append(row)
        elif row["cue_label"]:
            raise ValueError("paired diagnostic cue lacks interval")
    trial_truth = []
    decisions: dict[str, list[str]] = {name: [] for name in by_model}
    expected_ordinals = {item["ordinal"] for item in result["intervals"]}
    if set(by_interval) - expected_ordinals:
        raise ValueError("paired diagnostic unknown interval")
    for item in result["intervals"]:
        ordinal = item["ordinal"]
        grouped = by_interval.get(ordinal, {})
        observed_count = len(grouped.get("baseline", []))
        if (observed_count != item["scored_frames"]
                or observed_count != len(grouped.get("signed", []))
                or any(row["cue_label"] != item["cue_label"] for row in grouped.get("baseline", []))):
            raise ValueError("paired diagnostic interval population mismatch")
        if not observed_count:
            continue
        trial_truth.append(item["cue_label"])
        for name in by_model:
            p = np.mean([np.asarray(json.loads(row[f"{name}_probabilities"]))
                         for row in grouped[name]], axis=0)
            decision = model_classes[int(np.argmax(p))]
            if (decision != item[f"{name}_mean_probability_decision"]
                    or not np.isclose(float(item[f"{name}_cue_probability"]),
                                      float(p[model_classes.index(item["cue_label"])]),
                                      atol=1e-12)):
                raise ValueError("paired diagnostic trial probability mismatch")
            decisions[name].append(decision)
    labels = list(model_classes)
    for name, values in by_model.items():
        truth = values["truth"]
        expected_frame = {
            "joint": _score(truth, values["joint"], labels=labels),
            "hand": _score([_joint28_parts(value)[1] for value in truth], values["hand"],
                           labels=list(JOINT_HANDS)),
            "arm": _score([_joint28_parts(value)[0] for value in truth], values["arm"],
                          labels=list(JOINT_ARMS)),
        }
        if expected_frame != result["frame_scores"][name]:
            raise ValueError(f"paired diagnostic frame metric mismatch: {name}")
        if _score(trial_truth, decisions[name], labels=labels) != result["trial_scores"][name]:
            raise ValueError(f"paired diagnostic trial metric mismatch: {name}")
    corrected = sum(a != truth and b == truth for truth, a, b in
                    zip(trial_truth, decisions["baseline"], decisions["signed"]))
    new_errors = sum(a == truth and b != truth for truth, a, b in
                     zip(trial_truth, decisions["baseline"], decisions["signed"]))
    if (corrected != result["signed_corrected_baseline_trial_errors"]
            or new_errors != result["signed_new_trial_errors"]
            or len(trial_truth) != result["scored_manual_stable_core_intervals"]):
        raise ValueError("paired diagnostic trial comparison mismatch")
    audit = {"status": "verified", "analysis_sha256": _sha(path),
             "frame_rows": len(rows), "scored_intervals": len(trial_truth),
             "boundary": "Read-back verifies saved cue agreement, not physiological truth or live latency."}
    (directory / "paired_song28_verification.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_directory", type=Path)
    args = parser.parse_args()
    result = run(args.capture_directory)
    print(json.dumps({key: result[key] for key in (
        "emitted_common_frames", "scored_manual_stable_core_intervals",
        "signed_corrected_baseline_trial_errors", "signed_new_trial_errors")},
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
