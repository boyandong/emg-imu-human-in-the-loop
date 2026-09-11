from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .signal import polyphase_resample
from .state import Direction, Gesture, Phase


SCHEMA_VERSION = "3.0"
TARGET_RATE_HZ = 200.0
ARM_LABELS = {
    "still": Direction.NONE,
    "forward": Direction.FORWARD,
    "backward": Direction.BACKWARD,
    "left": Direction.LEFT,
    "right": Direction.RIGHT,
    "up": Direction.UP,
    "down": Direction.DOWN,
}
HAND_LABELS = {
    "neutral": Gesture.NEUTRAL,
    "index_pinch": Gesture.PINCH,
    "fist": Gesture.FIST,
    "open_hand": Gesture.OPEN,
}


class Hdf5V3Error(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Hdf5V3Policy:
    target_rate_hz: float = TARGET_RATE_HZ
    arm_onset_guard_ms: float = 120.0
    arm_active_duration_ms: float = 500.0
    hand_settle_ms: float = 150.0
    max_imu_gap_ms: float = 30.0
    max_missing_ratio: float = 0.10
    min_valid_emg_channels: int = 6


@dataclass(frozen=True, slots=True)
class Hdf5V3Result:
    output_root: Path
    session_count: int
    trial_count: int
    warnings: tuple[str, ...]


def _h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - dependency error is actionable
        raise RuntimeError("h5py is required for HDF5 3.0 import") from exc
    return h5py


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _session_number(value: str) -> str:
    match = re.search(r"(\d{1,3})$", value.strip())
    if not match:
        raise Hdf5V3Error(f"session_id {value!r} has no numeric suffix")
    number = int(match.group(1))
    if number not in (1, 2, 3, 4):
        raise Hdf5V3Error(f"formal session must be S01-S04; got {value!r}")
    return str(number)


def _parse_combination(label: str) -> tuple[Direction, Gesture]:
    for arm in ARM_LABELS:
        prefix = arm + "_"
        if label.startswith(prefix):
            hand = label[len(prefix):]
            if hand in HAND_LABELS:
                return ARM_LABELS[arm], HAND_LABELS[hand]
    raise Hdf5V3Error(f"unsupported 28-state label: {label!r}")


def _quality_metadata(attrs: Any) -> tuple[bool, np.ndarray, dict[str, object]]:
    raw = _text(attrs.get("quality_report_json", "")).strip()
    if not raw:
        return False, np.zeros(8), {"passed": False, "reasons": ["missing_quality_report"]}
    try:
        report = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise Hdf5V3Error(f"quality_report_json is invalid: {exc}") from exc
    if not isinstance(report, dict):
        raise Hdf5V3Error("quality_report_json must contain an object")
    passed = bool(report.get("passed", False))
    quality = np.ones(8, dtype=np.float64)
    rules = (
        ("saturation_ratio", lambda value: value <= 0.0),
        ("zero_ratio", lambda value: value <= 0.98),
        ("mains_50hz_ratio", lambda value: value <= 0.35),
    )
    for name, predicate in rules:
        if name not in report:
            continue
        values = np.asarray(report[name], dtype=np.float64).reshape(-1)
        if values.shape != (8,) or not np.isfinite(values).all():
            raise Hdf5V3Error(f"quality report {name} must contain 8 finite values")
        quality *= np.asarray([1.0 if predicate(float(value)) else 0.0 for value in values])
    return passed, quality, report


def _require(handle: Any, name: str) -> Any:
    if name not in handle:
        raise Hdf5V3Error(f"missing HDF5 object: {name}")
    return handle[name]


def _strict_time(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.int64).reshape(-1)
    if len(result) < 2 or np.any(result <= 0) or np.any(np.diff(result) <= 0):
        raise Hdf5V3Error(f"{name} must be positive and strictly increasing")
    return result


def _validate_stream(
    values: np.ndarray, times: np.ndarray, sequence: np.ndarray, channels: int, name: str,
) -> None:
    if values.ndim != 2 or values.shape[1] != channels or len(values) != len(times):
        raise Hdf5V3Error(f"{name} has inconsistent shapes")
    if len(sequence) != len(values):
        raise Hdf5V3Error(f"{name}/packet_seq length mismatch")
    if not np.isfinite(values).all():
        raise Hdf5V3Error(f"{name} contains NaN or Inf")


def _interpolate_imu(
    target_ns: np.ndarray,
    source_ns: np.ndarray,
    accel: np.ndarray,
    gyro: np.ndarray,
    max_gap_ms: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target = target_ns.astype(np.float64)
    source = source_ns.astype(np.float64)
    aligned_accel = np.column_stack([
        np.interp(target, source, accel[:, axis]) for axis in range(3)
    ])
    aligned_gyro = np.column_stack([
        np.interp(target, source, gyro[:, axis]) for axis in range(3)
    ])
    positions = np.searchsorted(source, target)
    left = np.clip(positions - 1, 0, len(source) - 1)
    right = np.clip(positions, 0, len(source) - 1)
    nearest_gap = np.minimum(np.abs(target - source[left]), np.abs(target - source[right]))
    valid = (
        (target >= source[0]) & (target <= source[-1])
        & (nearest_gap <= max_gap_ms * 1_000_000.0)
    )
    interpolated = nearest_gap > 250_000.0
    return aligned_accel, aligned_gyro, valid, interpolated


def _audit_bad_mask(
    target_ns: np.ndarray,
    audit: np.ndarray,
    packet_type: int,
    rate_hz: float,
) -> np.ndarray:
    bad = np.zeros(len(target_ns), dtype=bool)
    if not len(audit):
        return bad
    period_ns = 1e9 / rate_hz
    for row in audit:
        if int(row["packet_type"]) != packet_type:
            continue
        lost = max(int(row["lost_before"]), 0)
        duplicate = bool(row["duplicate"])
        out_of_order = bool(row["out_of_order"])
        if not (lost or duplicate or out_of_order):
            continue
        received = float(row["pc_received_ns"])
        start = received - max(lost, 1) * period_ns - period_ns / 2
        end = received + period_ns / 2
        bad |= (target_ns >= start) & (target_ns <= end)
    return bad


def _time_for_sample(sample_index: int, indices: np.ndarray, times: np.ndarray) -> int:
    position = int(np.searchsorted(indices, sample_index))
    if position < len(indices) and int(indices[position]) == sample_index:
        return int(times[position])
    period = float(np.median(np.diff(times)))
    return int(round(times[0] + (sample_index - int(indices[0])) * period))


def _cue_indices(cues: np.ndarray, trial_id: int) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in cues:
        if int(row["trial_id"]) == trial_id:
            result[_text(row["name"])] = int(row["sample_index"])
    return result


def inspect_hdf5_v3(path: str | Path) -> dict[str, object]:
    source = Path(path)
    errors: list[str] = []
    warnings: list[str] = []
    summary: dict[str, object] = {"path": str(source)}
    try:
        h5py = _h5py()
        with h5py.File(source, "r") as handle:
            meta = _require(handle, "meta")
            schema = _text(meta.attrs.get("schema_version", ""))
            if schema != SCHEMA_VERSION:
                raise Hdf5V3Error(f"schema_version must be 3.0; got {schema!r}")
            emg = np.asarray(_require(handle, "streams/emg/raw"))
            emg_time = _strict_time(
                np.asarray(_require(handle, "streams/emg/sample_time_ns")),
                "streams/emg/sample_time_ns",
            )
            emg_seq = np.asarray(_require(handle, "streams/emg/packet_seq")).reshape(-1)
            emg_index = np.asarray(
                _require(handle, "streams/emg/sample_index"), dtype=np.int64,
            ).reshape(-1)
            accel = np.asarray(_require(handle, "streams/imu/accel"), dtype=np.float64)
            gyro = np.asarray(_require(handle, "streams/imu/gyro"), dtype=np.float64)
            imu_time = _strict_time(
                np.asarray(_require(handle, "streams/imu/sample_time_ns")),
                "streams/imu/sample_time_ns",
            )
            imu_seq = np.asarray(_require(handle, "streams/imu/packet_seq")).reshape(-1)
            _validate_stream(emg, emg_time, emg_seq, 8, "streams/emg")
            if len(emg_index) != len(emg) or np.any(np.diff(emg_index) != 1):
                raise Hdf5V3Error("streams/emg/sample_index must be contiguous and match EMG")
            _validate_stream(accel, imu_time, imu_seq, 3, "streams/imu/accel")
            if gyro.shape != accel.shape or not np.isfinite(gyro).all():
                raise Hdf5V3Error("streams/imu/gyro has inconsistent shape or values")
            trials = np.asarray(_require(handle, "trials"))
            cues = np.asarray(_require(handle, "cue_events"))
            audit = np.asarray(_require(handle, "packet_audit"))
            required_trial = {
                "trial_id", "label", "trial_start_sample", "prompt_start_sample",
                "prompt_end_sample", "trial_end_sample", "valid", "relative_onset_offset_ms",
            }
            if trials.dtype.names is None or not required_trial.issubset(trials.dtype.names):
                raise Hdf5V3Error("trials table is missing v3 fields")
            required_cue = {"name", "sample_index", "trial_id"}
            if cues.dtype.names is None or not required_cue.issubset(cues.dtype.names):
                raise Hdf5V3Error("cue_events table is missing v3 fields")
            cue_semantics = _text(handle["cue_events"].attrs.get("semantics", ""))
            if cue_semantics != "ui_action_cue_not_movement_onset":
                raise Hdf5V3Error(
                    "cue_events semantics must explicitly say ui_action_cue_not_movement_onset"
                )
            required_audit = {
                "packet_type", "pc_received_ns", "lost_before", "duplicate", "out_of_order",
            }
            if audit.dtype.names is None or not required_audit.issubset(audit.dtype.names):
                raise Hdf5V3Error("packet_audit table is missing v3 fields")
            participant = _text(meta.attrs.get("participant_id", "")).strip()
            session = _text(meta.attrs.get("session_id", "")).strip()
            if not participant or not session:
                raise Hdf5V3Error("participant_id and session_id are required")
            session_number = _session_number(session)
            gate, channels, _ = _quality_metadata(meta.attrs)
            valid_trials = [row for row in trials if bool(row["valid"])]
            seen_trial_ids: set[int] = set()
            previous_end = -1
            for row in valid_trials:
                trial_id = int(row["trial_id"])
                if trial_id in seen_trial_ids:
                    errors.append(f"duplicate trial_id: {trial_id}")
                seen_trial_ids.add(trial_id)
                bounds = [
                    int(row[name]) for name in (
                        "trial_start_sample", "prompt_start_sample",
                        "prompt_end_sample", "trial_end_sample",
                    )
                ]
                if not (0 <= bounds[0] <= bounds[1] < bounds[2] <= bounds[3] <= len(emg)):
                    errors.append(f"trial {trial_id} has invalid boundaries {bounds}")
                if bounds[0] < previous_end:
                    errors.append(f"trial {trial_id} overlaps the previous trial")
                previous_end = max(previous_end, bounds[3])
                declared_offset = int(row["relative_onset_offset_ms"])
                if declared_offset not in (-200, 0, 200):
                    errors.append(f"trial {trial_id} has invalid onset offset {declared_offset}")
                direction, gesture = _parse_combination(_text(row["label"]))
                expected = {f"arm:{direction.name.lower() if direction != Direction.NONE else 'still'}"}
                hand_name = next(name for name, value in HAND_LABELS.items() if value == gesture)
                expected.add(f"hand:{hand_name}")
                cue_map = _cue_indices(cues, trial_id)
                names = set(cue_map)
                if not expected.issubset(names):
                    errors.append(
                        f"trial {trial_id} missing cues {sorted(expected - names)}"
                    )
                else:
                    arm_name = next(name for name in expected if name.startswith("arm:"))
                    hand_name_full = next(name for name in expected if name.startswith("hand:"))
                    observed_offset = (
                        cue_map[hand_name_full] - cue_map[arm_name]
                    ) * 1000.0 / float(meta.attrs.get("emg_nominal_rate_hz", 250.0))
                    if abs(observed_offset - declared_offset) > 25.0:
                        warnings.append(
                            f"trial {trial_id} cue offset {observed_offset:.1f} ms "
                            f"differs from declared {declared_offset} ms"
                        )
            emg_rate = float(meta.attrs.get("emg_nominal_rate_hz", meta.attrs.get("sampling_rate", 0)))
            imu_rate = float(meta.attrs.get("imu_nominal_rate_hz", 0))
            if not np.isclose(emg_rate, 250.0, rtol=0.05):
                errors.append(f"nominal EMG rate must be about 250 Hz; got {emg_rate}")
            if not np.isclose(imu_rate, 112.0, rtol=0.05):
                errors.append(f"nominal IMU rate must be about 112 Hz; got {imu_rate}")
            measured_emg = float(meta.attrs.get("measured_emg_rate_hz", 0.0))
            measured_imu = float(meta.attrs.get("measured_imu_rate_hz", 0.0))
            if measured_emg > 0 and not np.isclose(measured_emg, emg_rate, rtol=0.10):
                warnings.append(
                    f"measured EMG rate {measured_emg:.2f} differs from nominal {emg_rate:.2f}"
                )
            if measured_imu > 0 and not np.isclose(measured_imu, imu_rate, rtol=0.10):
                warnings.append(
                    f"measured IMU rate {measured_imu:.2f} differs from nominal {imu_rate:.2f}"
                )
            if not gate:
                warnings.append("quality gate did not pass; trials will be retained but excluded from training")
            if int(np.count_nonzero(channels >= 0.5)) < 6:
                warnings.append("fewer than six EMG channels passed the quality report")
            summary.update({
                "schema_version": schema,
                "participant_id": participant,
                "source_session_id": session,
                "session_id": session_number,
                "emg_samples": len(emg),
                "imu_samples": len(accel),
                "valid_trials": len(valid_trials),
                "quality_gate_pass": gate,
                "valid_emg_channels": int(np.count_nonzero(channels >= 0.5)),
                "nominal_emg_rate_hz": emg_rate,
                "nominal_imu_rate_hz": imu_rate,
            })
    except Exception as exc:
        if isinstance(exc, Hdf5V3Error):
            errors.append(str(exc))
        else:
            errors.append(f"cannot read HDF5 3.0 file: {exc}")
    return {
        **summary,
        "status": "ok" if not errors else "error",
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
    }


def _convert_file(source: Path, stage: Path, policy: Hdf5V3Policy) -> tuple[dict[str, object], list[str]]:
    check = inspect_hdf5_v3(source)
    if check["status"] != "ok":
        raise Hdf5V3Error("; ".join(check["errors"]))
    warnings = list(check["warnings"])
    h5py = _h5py()
    with h5py.File(source, "r") as handle:
        attrs = handle["meta"].attrs
        participant = _text(attrs["participant_id"]).strip()
        source_session = _text(attrs["session_id"]).strip()
        session = _session_number(source_session)
        session_origin_ns = int(np.asarray(handle["streams/emg/sample_time_ns"])[0])
        emg = np.asarray(handle["streams/emg/raw"], dtype=np.float64)
        emg_indices = np.asarray(handle["streams/emg/sample_index"], dtype=np.int64)
        emg_ns = _strict_time(
            np.asarray(handle["streams/emg/sample_time_ns"]), "streams/emg/sample_time_ns",
        )
        accel = np.asarray(handle["streams/imu/accel"], dtype=np.float64)
        gyro = np.asarray(handle["streams/imu/gyro"], dtype=np.float64)
        imu_ns = _strict_time(
            np.asarray(handle["streams/imu/sample_time_ns"]), "streams/imu/sample_time_ns",
        )
        trials = np.asarray(handle["trials"])
        cues = np.asarray(handle["cue_events"])
        audit = np.asarray(handle["packet_audit"])
        emg_rate = float(attrs.get("emg_nominal_rate_hz", attrs.get("sampling_rate", 250.0)))
        imu_rate = float(attrs.get("imu_nominal_rate_hz", 112.0))
        gate_pass, channel_quality, quality_report = _quality_metadata(attrs)
        wearing_keys = (
            "dominant_hand", "donning_notes", "tested_arm", "channel1_orientation", "anatomical_marker",
            "strap_setting", "stabilization_sec", "physical_condition", "dataset_split",
            "protocol_name", "experiment_name", "software_version", "start_datetime",
            "measured_emg_rate_hz",
            "measured_imu_rate_hz", "lost_frames", "duplicate_frames", "out_of_order_frames",
        )
        wearing = {key: _text(attrs[key]) for key in wearing_keys if key in attrs}

        output_dir = stage / f"session_{session}"
        output_dir.mkdir(parents=True, exist_ok=True)
        trial_count = 0
        stable_samples = 0
        excluded_trials = 0
        for row in trials:
            if not bool(row["valid"]):
                excluded_trials += 1
                continue
            numeric_trial_id = int(row["trial_id"])
            source_label = _text(row["label"])
            direction_label, gesture_label = _parse_combination(source_label)
            start_index = int(row["trial_start_sample"])
            end_index = int(row["trial_end_sample"])
            prompt_start_index = int(row["prompt_start_sample"])
            prompt_end_index = int(row["prompt_end_sample"])
            start = int(np.searchsorted(emg_indices, start_index))
            end = int(np.searchsorted(emg_indices, end_index))
            if end - start < 3:
                raise Hdf5V3Error(f"trial {numeric_trial_id} is too short")
            source_emg = emg[start:end]
            source_emg_ns = emg_ns[start:end]
            target_emg = polyphase_resample(
                source_emg, emg_rate, policy.target_rate_hz,
            ).astype(np.float32)
            target_ns = source_emg_ns[0] + np.rint(
                np.arange(len(target_emg)) * 1e9 / policy.target_rate_hz
            ).astype(np.int64)
            source_imu_trial_ns = imu_ns[
                (imu_ns >= source_emg_ns[0]) & (imu_ns <= source_emg_ns[-1])
            ]
            aligned_accel, aligned_gyro, imu_valid, interpolated = _interpolate_imu(
                target_ns, imu_ns, accel, gyro, policy.max_imu_gap_ms,
            )
            emg_transport_bad = _audit_bad_mask(target_ns, audit, 1, emg_rate)
            imu_transport_bad = _audit_bad_mask(target_ns, audit, 2, imu_rate)
            imu_valid &= ~imu_transport_bad
            missing_mask = emg_transport_bad | ~imu_valid
            timestamp_valid = np.ones(len(target_ns), dtype=bool)
            target_ms = (target_ns - session_origin_ns).astype(np.float64) / 1e6

            cue_map = _cue_indices(cues, numeric_trial_id)
            arm_name = "still" if direction_label == Direction.NONE else direction_label.name.lower()
            hand_name = next(name for name, value in HAND_LABELS.items() if value == gesture_label)
            arm_cue_index = cue_map[f"arm:{arm_name}"]
            hand_cue_index = cue_map[f"hand:{hand_name}"]
            arm_cue_ns = _time_for_sample(arm_cue_index, emg_indices, emg_ns)
            hand_cue_ns = _time_for_sample(hand_cue_index, emg_indices, emg_ns)
            prompt_start_ns = _time_for_sample(prompt_start_index, emg_indices, emg_ns)
            prompt_end_ns = _time_for_sample(prompt_end_index, emg_indices, emg_ns)
            prompt = (target_ns >= prompt_start_ns) & (target_ns < prompt_end_ns)

            direction = np.full(len(target_ns), int(Direction.UNKNOWN), dtype=np.int16)
            gesture = np.full(len(target_ns), int(Gesture.UNKNOWN), dtype=np.int16)
            arm_phase = np.full(len(target_ns), int(Phase.TRANSITION), dtype=np.int16)
            hand_phase = np.full(len(target_ns), int(Phase.TRANSITION), dtype=np.int16)
            before_prompt = target_ns < prompt_start_ns
            direction[before_prompt] = int(Direction.NONE)
            gesture[before_prompt] = int(Gesture.NEUTRAL)
            arm_phase[before_prompt] = int(Phase.IDLE)
            hand_phase[before_prompt] = int(Phase.IDLE)

            if direction_label == Direction.NONE:
                direction[prompt] = int(Direction.NONE)
                arm_phase[prompt] = int(Phase.HOLD)
            else:
                arm_settled_ns = arm_cue_ns + round(policy.arm_onset_guard_ms * 1e6)
                arm_active_end_ns = min(
                    prompt_end_ns,
                    arm_settled_ns + round(policy.arm_active_duration_ms * 1e6),
                )
                arm_onset = prompt & (target_ns >= arm_cue_ns) & (target_ns < arm_settled_ns)
                arm_active = prompt & (target_ns >= arm_settled_ns) & (target_ns < arm_active_end_ns)
                arm_phase[arm_onset] = int(Phase.ONSET)
                arm_phase[arm_active] = int(Phase.ACTIVE)
                direction[arm_active] = int(direction_label)

            if gesture_label == Gesture.NEUTRAL:
                gesture[prompt] = int(Gesture.NEUTRAL)
                hand_phase[prompt] = int(Phase.HOLD)
            else:
                hand_settled_ns = hand_cue_ns + round(policy.hand_settle_ms * 1e6)
                hand_onset = prompt & (target_ns >= hand_cue_ns) & (target_ns < hand_settled_ns)
                hand_hold = prompt & (target_ns >= hand_settled_ns)
                hand_phase[hand_onset] = int(Phase.ONSET)
                hand_phase[hand_hold] = int(Phase.HOLD)
                gesture[hand_hold] = int(gesture_label)
            after_prompt = target_ns >= prompt_end_ns
            arm_phase[after_prompt] = int(Phase.RELEASE)
            hand_phase[after_prompt] = int(Phase.RELEASE)

            emg_quality = (~emg_transport_bad).astype(np.float64) * float(np.mean(channel_quality))
            imu_quality = imu_valid.astype(np.float64)
            window_quality = np.minimum(emg_quality, imu_quality)
            if not gate_pass:
                window_quality.fill(0.0)
            stable_mask = (
                prompt & (direction != int(Direction.UNKNOWN))
                & (gesture != int(Gesture.UNKNOWN))
                & timestamp_valid & ~missing_mask
                & (window_quality >= 0.8)
                & gate_pass
                & (np.count_nonzero(channel_quality >= 0.5) >= policy.min_valid_emg_channels)
            )
            stable_samples += int(np.count_nonzero(stable_mask))
            trial_id = f"hdf5v3-{participant}-{source_session}-t{numeric_trial_id:04d}"
            metadata = {
                "source_format": "HDF5 3.0",
                "source_label": source_label,
                "label_semantics": "cue_derived_conservative_not_biological_onset",
                "relative_onset_offset_ms": int(row["relative_onset_offset_ms"]),
                "stage_id": int(row["stage_id"]),
                "donning_id": int(row["donning_id"]),
                "source_emg_rate_hz": emg_rate,
                "source_imu_rate_hz": imu_rate,
                "target_rate_hz": policy.target_rate_hz,
                "wearing": wearing,
                "quality_report": quality_report,
            }
            np.savez_compressed(
                output_dir / f"{trial_id}.npz",
                timestamp_ms=target_ms,
                emg=target_emg,
                accel=aligned_accel.astype(np.float32),
                gyro=aligned_gyro.astype(np.float32),
                direction=direction,
                gesture=gesture,
                arm_phase_label=arm_phase,
                hand_phase_label=hand_phase,
                stable_mask=stable_mask,
                window_quality=window_quality.astype(np.float32),
                emg_quality=emg_quality.astype(np.float32),
                imu_quality=imu_quality.astype(np.float32),
                missing_mask=missing_mask,
                channel_quality=channel_quality.astype(np.float32),
                timestamp_valid=timestamp_valid,
                imu_valid=imu_valid,
                interpolated_imu=interpolated,
                quality_gate_pass=np.asarray(gate_pass, dtype=np.bool_),
                session_id=np.asarray(session),
                source_session_id=np.asarray(source_session),
                stage_id=np.asarray(int(row["stage_id"]), dtype=np.int16),
                donning_id=np.asarray(int(row["donning_id"]), dtype=np.int16),
                relative_onset_offset_ms=np.asarray(
                    int(row["relative_onset_offset_ms"]), dtype=np.int16,
                ),
                trial_id=np.asarray(trial_id),
                session_date=np.asarray(_text(attrs.get("start_datetime", ""))[:10]),
                cue_ms=np.asarray((prompt_start_ns - session_origin_ns) / 1e6),
                arm_onset_ms=np.asarray((arm_cue_ns - session_origin_ns) / 1e6),
                hand_onset_ms=np.asarray((hand_cue_ns - session_origin_ns) / 1e6),
                arm_end_ms=np.asarray((prompt_end_ns - session_origin_ns) / 1e6),
                hand_end_ms=np.asarray((prompt_end_ns - session_origin_ns) / 1e6),
                source_emg_timestamp_ms=(source_emg_ns - session_origin_ns) / 1e6,
                source_imu_timestamp_ms=(source_imu_trial_ns - session_origin_ns) / 1e6,
                metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
            )
            trial_count += 1

        return ({
            "source": str(source),
            "source_sha256": _sha256(source),
            "schema_version": SCHEMA_VERSION,
            "participant_id": participant,
            "source_session_id": source_session,
            "session_id": session,
            "trial_count": trial_count,
            "invalid_trials_excluded": excluded_trials,
            "stable_samples": stable_samples,
            "quality_gate_pass": gate_pass,
            "valid_emg_channels": int(np.count_nonzero(channel_quality >= 0.5)),
            "nominal_emg_rate_hz": emg_rate,
            "nominal_imu_rate_hz": imu_rate,
            "measured_emg_rate_hz": float(attrs.get("measured_emg_rate_hz", 0.0)),
            "measured_imu_rate_hz": float(attrs.get("measured_imu_rate_hz", 0.0)),
            "wearing": wearing,
            "quality_report": quality_report,
        }, warnings)


def adapt_hdf5_v3(
    source_root: str | Path,
    output_root: str | Path,
    *,
    policy: Hdf5V3Policy | None = None,
) -> Hdf5V3Result:
    source = Path(source_root)
    output = Path(output_root)
    if output.exists():
        raise Hdf5V3Error(f"output already exists; refusing to overwrite: {output}")
    files = [source] if source.is_file() else sorted(source.rglob("*.h5"))
    if not files:
        raise Hdf5V3Error(f"no .h5 files found under {source}")
    policy = policy or Hdf5V3Policy()
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    records: list[dict[str, object]] = []
    warnings: list[str] = []
    try:
        seen_sessions: set[tuple[str, str]] = set()
        for path in files:
            record, file_warnings = _convert_file(path, stage, policy)
            key = (str(record["participant_id"]), str(record["session_id"]))
            if key in seen_sessions:
                raise Hdf5V3Error(f"duplicate participant/session source: {key}")
            seen_sessions.add(key)
            records.append(record)
            warnings.extend(f"{path.name}: {warning}" for warning in file_warnings)
        manifest = {
            "format": "emgimu-self-collected-hdf5-v3-adapter",
            "adapter_version": "1.0.0",
            "target_rate_hz": policy.target_rate_hz,
            "policy": asdict(policy),
            "label_source": "UI cues with conservative margins; not biological onset",
            "calibration_status": "separate guided 60-second calibration still required",
            "sessions": records,
            "warnings": sorted(set(warnings)),
        }
        (stage / "HDF5_V3_IMPORT.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        (stage / ".gitignore").write_text("session_*/\n", encoding="utf-8")
        stage.replace(output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return Hdf5V3Result(
        output, len(records), sum(int(row["trial_count"]) for row in records),
        tuple(sorted(set(warnings))),
    )
