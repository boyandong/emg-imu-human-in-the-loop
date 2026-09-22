from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from emgforce.collection_protocol import (
    FORMAL_PROTOCOL_NAME, FORMAL_SESSION_SPLITS, QUALITY_THRESHOLD_VERSION,
    SESSION_MANIFEST_FILENAME, TIMESTAMP_SOURCE,
)
from emgforce.config import EMG_CHANNELS, IMU_SAMPLING_RATE, SAMPLING_RATE


REQUIRED_METADATA = (
    "participant_id", "session_id", "dominant_hand", "recorded_arm",
    "donning_id", "channel_1_orientation", "anatomical_distance_mm",
    "strap_scale", "strap_tightness", "skin_condition", "fatigue_before",
    "fatigue_after", "reference_photo_name", "reference_photo_sha256",
    "operator_id", "protocol_version", "protocol_file_sha256",
)
MIN_STABLE_EMG_SAMPLES = round(SAMPLING_RATE * 0.2)


def _text(value: Any) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8").rstrip("\x00")
    return str(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate_session_readiness(
        hdf5_path: Path, output_path: Path | None = None) -> dict[str, Any]:
    """Audit a closed acquisition and atomically write its collection gate."""
    hdf5_path = Path(hdf5_path)
    output_path = (Path(output_path) if output_path is not None
                   else hdf5_path.with_name(SESSION_MANIFEST_FILENAME))
    problems: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {}

    try:
        with h5py.File(hdf5_path, "r") as handle:
            meta = handle["meta"].attrs
            schema_ok = _text(meta.get("schema_version", "")) == "3.0"
            checks["hdf5_v3"] = schema_ok
            if not schema_ok:
                problems.append("不是正式 HDF5 v3")

            protocol_name = _text(meta.get("protocol_name", ""))
            protocol_version = _text(meta.get("protocol_version", ""))
            protocol_ok = protocol_name == FORMAL_PROTOCOL_NAME and protocol_version == FORMAL_PROTOCOL_NAME
            checks["formal_protocol"] = protocol_ok
            if not protocol_ok:
                problems.append(f"协议必须为 {FORMAL_PROTOCOL_NAME}")
            try:
                protocol = json.loads(_text(meta.get("protocol_json", "{}")))
            except json.JSONDecodeError:
                protocol = {}
                problems.append("protocol_json 无法解析")
            protocol_hash_ok = bool(meta.get("protocol_file_sha256", "")) and (
                not protocol.get("source_sha256")
                or _text(meta.get("protocol_file_sha256", "")) == str(protocol["source_sha256"])
            )
            checks["protocol_hash_recorded"] = protocol_hash_ok
            if not protocol_hash_ok:
                problems.append("协议文件 SHA-256 缺失或与保存配置不一致")

            nominal_emg = float(meta.get("emg_nominal_rate_hz", 0))
            nominal_imu = float(meta.get("imu_nominal_rate_hz", 0))
            measured_emg = float(meta.get("measured_emg_rate_hz", 0))
            measured_imu = float(meta.get("measured_imu_rate_hz", 0))
            checks["sampling_rates"] = {
                "emg_nominal_hz": nominal_emg, "emg_measured_hz": measured_emg,
                "imu_nominal_hz": nominal_imu, "imu_measured_hz": measured_imu,
            }
            if not np.isclose(nominal_emg, SAMPLING_RATE):
                problems.append(f"EMG 标称率应为 {SAMPLING_RATE} Hz")
            if not np.isclose(nominal_imu, IMU_SAMPLING_RATE):
                problems.append(f"IMU 标称率应为 {IMU_SAMPLING_RATE} Hz")
            if measured_emg and abs(measured_emg - SAMPLING_RATE) / SAMPLING_RATE > 0.10:
                problems.append(f"EMG 实测率异常：{measured_emg:.2f} Hz")
            if measured_imu and abs(measured_imu - IMU_SAMPLING_RATE) / IMU_SAMPLING_RATE > 0.15:
                problems.append(f"IMU 实测率异常：{measured_imu:.2f} Hz")
            if handle["streams/emg/raw"].shape[1] != EMG_CHANNELS:
                problems.append(f"EMG 通道数不是 {EMG_CHANNELS}")
            if _text(meta.get("timestamp_source", "")) != TIMESTAMP_SOURCE:
                problems.append(f"timestamp_source 必须明确为 {TIMESTAMP_SOURCE}")

            missing_meta = [name for name in REQUIRED_METADATA
                            if name not in meta or _text(meta.get(name, "")).strip() in {"", "-1"}]
            checks["metadata_complete"] = not missing_meta
            checks["missing_metadata"] = missing_meta
            if missing_meta:
                problems.append("结构化元数据缺失：" + ", ".join(missing_meta))

            session_id = _text(meta.get("session_id", "")).upper()
            expected_split = FORMAL_SESSION_SPLITS.get(session_id)
            split_ok = expected_split is not None and _text(meta.get("dataset_split", "")) == expected_split
            checks["session_split"] = {"session": session_id, "expected": expected_split,
                                       "actual": _text(meta.get("dataset_split", "")),
                                       "passed": split_ok}
            if not split_ok:
                problems.append("Session 与 train/validation/test 划分不一致")
            if session_id == "S04" and not bool(meta.get("model_frozen_confirmed", False)):
                problems.append("S04 未确认模型、阈值和校准算法已冻结")

            trials = handle["trials"][:]
            names = set(trials.dtype.names or ())
            required_trial_fields = {
                "trial_kind", "block_index", "attempt", "rerecord_of_trial_id",
                "event_uid", "stable_start_sample", "stable_end_sample",
                "release_prompt_sample", "completion_status", "discard_reason",
            }
            if not required_trial_fields.issubset(names):
                problems.append("trial 审计字段不完整")
                formal_rows = []
            else:
                formal_rows = [row for row in trials if _text(row["trial_kind"]) == "formal"]

            expected_counts = {
                label: int(protocol.get("trials_per_label", {}).get(
                    label, protocol.get("trials_per_class", 0)))
                for label in protocol.get("labels", [])
            }
            valid_counts = Counter(
                _text(row["label"]) for row in formal_rows if bool(row["valid"]))
            checks["valid_trial_counts"] = dict(sorted(valid_counts.items()))
            missing_counts = {
                label: expected - valid_counts.get(label, 0)
                for label, expected in expected_counts.items()
                if valid_counts.get(label, 0) < expected
            }
            if len(expected_counts) != 28:
                problems.append("协议未声明完整 28 个组合")
            if "still_open_hand" not in expected_counts:
                problems.append("28 组合中缺少主动 Open")
            if missing_counts:
                problems.append("有效 trial 数不足：" + json.dumps(missing_counts, ensure_ascii=False))
            incomplete_trials = [int(row["trial_id"]) for row in formal_rows
                                 if bool(row["valid"])
                                 and _text(row["completion_status"]) != "completed"]
            unstable_intervals = [int(row["trial_id"]) for row in formal_rows
                                  if bool(row["valid"])
                                  and (int(row["stable_start_sample"]) < 0
                                       or int(row["stable_end_sample"])
                                       <= int(row["stable_start_sample"]))]
            checks["trial_execution"] = {
                "incomplete_trials": incomplete_trials,
                "invalid_stable_intervals": unstable_intervals,
            }
            if incomplete_trials:
                problems.append("存在未确认按要求完成的有效 trial")
            if unstable_intervals:
                problems.append("存在没有保守稳定标签区间的有效 trial")

            emg_indices = np.asarray(handle["streams/emg/sample_index"][:], dtype=np.int64)
            indices_ordered = bool(len(emg_indices) and np.all(np.diff(emg_indices) > 0))
            invalid_recorded_intervals: list[int] = []
            overlapping_intervals: list[int] = []
            previous_end = -1
            for row in sorted((row for row in formal_rows if bool(row["valid"])),
                              key=lambda item: int(item["stable_start_sample"])):
                trial_id = int(row["trial_id"])
                start = int(row["stable_start_sample"])
                end = int(row["stable_end_sample"])
                if start < 0 or end <= start:
                    continue
                if start < previous_end:
                    overlapping_intervals.append(trial_id)
                previous_end = max(previous_end, end)
                within_trial = (int(row["trial_start_sample"]) <= start < end
                                <= int(row["trial_end_sample"]))
                if indices_ordered:
                    first = int(np.searchsorted(emg_indices, start, side="left"))
                    last = int(np.searchsorted(emg_indices, end, side="left"))
                    recorded = emg_indices[first:last]
                    gaps = np.flatnonzero(np.diff(recorded) != 1)
                    longest_run = int(max(np.diff(np.r_[-1, gaps, len(recorded) - 1]),
                                          default=0))
                    within_recording = (start >= int(emg_indices[0])
                                        and end <= int(emg_indices[-1]) + 1)
                else:
                    longest_run = 0
                    within_recording = False
                if (not within_trial or not within_recording
                        or longest_run < MIN_STABLE_EMG_SAMPLES):
                    invalid_recorded_intervals.append(trial_id)
            checks["stable_interval_data"] = {
                "emg_samples": len(emg_indices),
                "sample_indices_strictly_increasing": indices_ordered,
                "outside_trial_or_recording_or_under_200ms": invalid_recorded_intervals,
                "overlapping_trial_ids": overlapping_intervals,
            }
            if not indices_ordered:
                problems.append("EMG 样本序号为空或不严格递增")
            if invalid_recorded_intervals:
                problems.append("有效 trial 的稳定区间没有至少 200 ms 的独立录制 EMG 数据")
            if overlapping_intervals:
                problems.append("有效 trial 的稳定区间相互重叠")

            offset_counts: dict[str, Counter[int]] = defaultdict(Counter)
            for row in formal_rows:
                if bool(row["valid"]):
                    offset_counts[_text(row["label"])][int(row["relative_onset_offset_ms"])] += 1
            offset_ok = True
            for label, expected in expected_counts.items():
                counts = offset_counts[label]
                values = [counts.get(offset, 0) for offset in (-200, 0, 200)]
                if expected >= 3 and (min(values) == 0 or max(values) - min(values) > 1):
                    offset_ok = False
            checks["onset_offset_counts"] = {
                label: {str(offset): count for offset, count in sorted(counts.items())}
                for label, counts in sorted(offset_counts.items())
            }
            if not offset_ok:
                problems.append("-200/0/+200 ms onset 条件缺失或不平衡")

            cues_by_trial: dict[int, list[Any]] = defaultdict(list)
            for cue in handle["cue_events"][:]:
                cues_by_trial[int(cue["trial_id"])].append(cue)
            missing_cue_trials: list[int] = []
            bad_actual_offsets: list[int] = []
            missing_release_trials: list[int] = []
            clock_errors: list[float] = []
            sample_errors: list[float] = []
            for row in formal_rows:
                if not bool(row["valid"]):
                    continue
                trial_id = int(row["trial_id"])
                cues = cues_by_trial[trial_id]
                arm_cue = next((cue for cue in cues if _text(cue["name"]).startswith("arm:")), None)
                hand_cue = next((cue for cue in cues if _text(cue["name"]).startswith("hand:")
                                 and _text(cue["name"]) != "hand:release"), None)
                if arm_cue is None or hand_cue is None:
                    missing_cue_trials.append(trial_id)
                    continue
                planned = int(row["relative_onset_offset_ms"])
                clock_offset = (int(hand_cue["emitted_monotonic_ns"])
                                - int(arm_cue["emitted_monotonic_ns"])) / 1e6
                sample_offset = (int(hand_cue["sample_index"])
                                 - int(arm_cue["sample_index"])) * 1000.0 / SAMPLING_RATE
                clock_errors.append(abs(clock_offset - planned))
                sample_errors.append(abs(sample_offset - planned))
                if abs(clock_offset - planned) > 80 or abs(sample_offset - planned) > 80:
                    bad_actual_offsets.append(trial_id)
                if _text(row["label"]).endswith("_open_hand") and not any(
                        _text(cue["name"]) == "hand:release" for cue in cues):
                    missing_release_trials.append(trial_id)
            checks["actual_cue_offsets"] = {
                "missing_cue_trials": missing_cue_trials,
                "outside_tolerance_trials": bad_actual_offsets,
                "missing_open_release_trials": missing_release_trials,
                "max_clock_error_ms": max(clock_errors, default=0.0),
                "max_sample_error_ms": max(sample_errors, default=0.0),
                "tolerance_ms": 80,
            }
            if missing_cue_trials:
                problems.append("存在未保存完整 arm/hand 实际提示时间的 trial")
            if bad_actual_offsets:
                problems.append("实际提示 offset 与计划值偏差超过 80 ms")
            if missing_release_trials:
                problems.append("Open trial 缺少独立 Release 提示记录")

            calibration_expected = [str(block.get("name"))
                                    for block in protocol.get("calibration_blocks", [])]
            calibration_rows = handle.get("calibration_blocks")
            calibration_valid = Counter()
            if calibration_rows is not None:
                calibration_valid.update(
                    _text(row["label"]) for row in calibration_rows[:] if bool(row["valid"]))
            missing_calibration = [name for name in calibration_expected
                                   if calibration_valid.get(name, 0) < 1]
            checks["calibration_complete"] = bool(calibration_expected) and not missing_calibration
            checks["missing_calibration_blocks"] = missing_calibration
            if not calibration_expected or missing_calibration:
                problems.append("Session 校准块不完整")

            try:
                quality = json.loads(_text(meta.get("quality_report_json", "{}")))
            except json.JSONDecodeError:
                quality = {}
            quality_ok = (bool(quality.get("passed")) and
                          quality.get("threshold_version") == QUALITY_THRESHOLD_VERSION)
            checks["pre_session_quality"] = quality
            if not quality_ok:
                problems.append("采集前质量门控未通过正式版本阈值")

            audit = handle["packet_audit"][:]
            total = max(1, len(audit) + int(np.sum(audit["lost_before"])))
            packet_metrics = {
                "lost_ratio": float(np.sum(audit["lost_before"]) / total),
                "duplicate_ratio": float(np.sum(audit["duplicate"]) / total),
                "out_of_order_ratio": float(np.sum(audit["out_of_order"]) / total),
            }
            checks["packet_quality"] = packet_metrics
            if packet_metrics["lost_ratio"] > 0.02:
                problems.append("丢包比例超过 2%")
            if packet_metrics["duplicate_ratio"] > 0.01:
                problems.append("重复包比例超过 1%")
            if packet_metrics["out_of_order_ratio"] > 0.01:
                problems.append("逆序包比例超过 1%")

            invalid_unexplained = [int(row["trial_id"]) for row in trials
                                   if not bool(row["valid"])
                                   and not _text(row["discard_reason"]).strip()]
            checks["invalid_trials"] = int(sum(not bool(row["valid"]) for row in trials))
            if invalid_unexplained:
                problems.append("存在未说明的废弃 trial：" + str(invalid_unexplained))

            events = handle["events"][:]
            event_types = Counter(_text(row["event_type"]) for row in events)
            block_count = max((int(row["block_index"]) for row in formal_rows), default=0)
            expected_breaks = max(0, block_count - 1)
            checks["block_breaks"] = {
                "blocks": block_count,
                "required_breaks": expected_breaks,
                "recorded_breaks": event_types.get("BLOCK_BREAK_START", 0),
                "fatigue_reports": event_types.get("FATIGUE_REPORT", 0),
            }
            if event_types.get("BLOCK_BREAK_START", 0) < expected_breaks:
                problems.append("block 强制休息记录不完整")
            if event_types.get("FATIGUE_REPORT", 0) < expected_breaks:
                problems.append("block 疲劳/不适记录不完整")

            continuous_alerts = event_types.get("QUALITY_ALERT", 0)
            checks["continuous_quality_alerts"] = continuous_alerts
            if continuous_alerts:
                warnings.append(f"采集中记录了 {continuous_alerts} 次质量异常；相关 trial 已标记 invalid")
    except Exception as exc:
        problems.append(f"HDF5 无法重新打开或完整性检查失败：{exc}")

    payload: dict[str, Any] = {
        "manifest_version": "session_collection_readiness_v1",
        "status": "passed" if not problems else "failed",
        "hdf5_file": hdf5_path.name,
        "hdf5_sha256": _sha256(hdf5_path) if hdf5_path.exists() else "",
        "protocol_required": FORMAL_PROTOCOL_NAME,
        "checks": checks,
        "problems": problems,
        "warnings": warnings,
    }
    # A UI session may finish while another page is refreshing its data view.
    # Reassert the destination directory before the atomic sibling write so a
    # successfully closed recording always receives its readiness report.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the sibling name short enough for Windows installations where the
    # project path is already close to the legacy MAX_PATH boundary.
    temporary = output_path.with_name(f".readiness.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output_path)
    return payload
