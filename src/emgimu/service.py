from __future__ import annotations

import socket
import csv
import json
import os
import time
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TextIO

import numpy as np

from .osc import OscPublisher, decode_message
from .runtime import HumanStateEstimator
from .state import Direction, Gesture, HumanState, QualityFlag


RAW_OSC_ADDRESS = "/emgimu/raw"
RAW_OSC_V2_ADDRESS = "/emgimu/raw/v2"


@dataclass(frozen=True, slots=True)
class RawSample:
    timestamp_ms: int
    emg: np.ndarray
    accel: np.ndarray
    gyro: np.ndarray
    sample_quality: float = 1.0
    missing: bool = False
    timestamp_valid: bool = True
    imu_valid: bool = True
    interpolated_imu: bool = False
    quality_gate_pass: bool = True
    duplicate_packet: bool = False
    out_of_order_packet: bool = False
    channel_quality: np.ndarray | None = None


def parse_raw_message(packet: bytes, *, address: str | None = None) -> RawSample | None:
    actual_address, args = decode_message(packet)
    accepted = {RAW_OSC_ADDRESS, RAW_OSC_V2_ADDRESS} if address is None else {address}
    if actual_address not in accepted:
        return None
    if actual_address == RAW_OSC_ADDRESS:
        if len(args) != 15:
            raise ValueError(
                f"{RAW_OSC_ADDRESS} expects timestamp + 8 EMG + 3 accel + 3 gyro values"
            )
        return RawSample(
            int(args[0]), np.asarray(args[1:9], dtype=np.float64),
            np.asarray(args[9:12], dtype=np.float64),
            np.asarray(args[12:15], dtype=np.float64),
        )
    if len(args) != 31:
        raise ValueError(
            f"{RAW_OSC_V2_ADDRESS} expects 31 values including transport and channel quality"
        )
    return RawSample(
        int(args[0]), np.asarray(args[1:9], dtype=np.float64),
        np.asarray(args[9:12], dtype=np.float64),
        np.asarray(args[12:15], dtype=np.float64),
        sample_quality=float(args[15]), missing=bool(args[16]),
        timestamp_valid=bool(args[17]), imu_valid=bool(args[18]),
        interpolated_imu=bool(args[19]), quality_gate_pass=bool(args[20]),
        duplicate_packet=bool(args[21]), out_of_order_packet=bool(args[22]),
        channel_quality=np.asarray(args[23:31], dtype=np.float64),
    )


class RawOscServer:
    def __init__(
        self,
        callback: Callable[[RawSample], None],
        host: str = "127.0.0.1",
        port: int = 9100,
        error_callback: Callable[[Exception], None] | None = None,
    ) -> None:
        self.callback = callback
        self.host = host
        self.port = int(port)
        self.last_error: Exception | None = None
        self.error_callback = error_callback

    def run_forever(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.bind((self.host, self.port))
            self.port = int(sock.getsockname()[1])
            while True:
                packet, _ = sock.recvfrom(8192)
                try:
                    sample = parse_raw_message(packet)
                except (ValueError, TypeError) as exc:
                    self.last_error = exc
                    if self.error_callback is not None:
                        self.error_callback(exc)
                    continue
                if sample is not None:
                    self.callback(sample)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    """Durably replace a small JSON audit artifact in the destination directory."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


class RuntimeHealthMonitor:
    """Bounded in-memory counters with an atomic, machine-readable health snapshot."""

    FORMAT_VERSION = 1

    def __init__(
        self,
        path: str | Path,
        *,
        run_id: str,
        write_interval_seconds: float = 1.0,
        latency_window: int = 2048,
    ) -> None:
        if write_interval_seconds <= 0 or latency_window < 1:
            raise ValueError("health monitor intervals must be positive")
        self.path = Path(path)
        self.run_id = str(run_id)
        self.started_at_utc = _utc_now()
        self.started_monotonic = time.monotonic()
        self.write_interval_seconds = float(write_interval_seconds)
        self.last_write_monotonic = 0.0
        self.samples_received = 0
        self.states_emitted = 0
        self.parse_errors = 0
        self.processing_errors = 0
        self.last_input_timestamp_ms: int | None = None
        self.last_state_timestamp_ms: int | None = None
        self.direction_unknown = 0
        self.gesture_unknown = 0
        self.quality_flag_counts: Counter[str] = Counter()
        self.processing_latency_ms: deque[float] = deque(maxlen=int(latency_window))
        self.last_error: dict[str, str] | None = None
        self.write(force=True)

    def observe_sample(self, sample: RawSample) -> None:
        self.samples_received += 1
        self.last_input_timestamp_ms = int(sample.timestamp_ms)

    def observe_state(self, state: HumanState, processing_latency_ms: float) -> None:
        self.states_emitted += 1
        self.last_state_timestamp_ms = int(state.timestamp_ms)
        self.processing_latency_ms.append(max(0.0, float(processing_latency_ms)))
        self.direction_unknown += int(state.direction == Direction.UNKNOWN)
        self.gesture_unknown += int(state.gesture == Gesture.UNKNOWN)
        for flag in QualityFlag:
            if flag != QualityFlag.NONE and state.signal_quality.flags & flag:
                self.quality_flag_counts[flag.name.lower()] += 1
        self.write()

    def observe_error(self, error: Exception, *, parse: bool = False) -> None:
        if parse:
            self.parse_errors += 1
        else:
            self.processing_errors += 1
        self.last_error = {
            "type": type(error).__name__,
            "message": str(error),
            "observed_at_utc": _utc_now(),
        }
        self.write(force=True)

    def snapshot(self, *, status: str = "running") -> dict[str, Any]:
        elapsed = max(time.monotonic() - self.started_monotonic, 1e-9)
        latencies = np.asarray(self.processing_latency_ms, dtype=np.float64)
        p50 = float(np.percentile(latencies, 50)) if len(latencies) else None
        p95 = float(np.percentile(latencies, 95)) if len(latencies) else None
        output_rate = self.states_emitted / elapsed
        return {
            "format_version": self.FORMAT_VERSION,
            "run_id": self.run_id,
            "status": status,
            "started_at_utc": self.started_at_utc,
            "updated_at_utc": _utc_now(),
            "uptime_seconds": elapsed,
            "samples_received": self.samples_received,
            "states_emitted": self.states_emitted,
            "output_rate_hz": output_rate,
            "last_input_timestamp_ms": self.last_input_timestamp_ms,
            "last_state_timestamp_ms": self.last_state_timestamp_ms,
            "direction_unknown_rate": (
                self.direction_unknown / self.states_emitted if self.states_emitted else None
            ),
            "gesture_unknown_rate": (
                self.gesture_unknown / self.states_emitted if self.states_emitted else None
            ),
            "processing_latency_ms": {
                "window_samples": len(latencies),
                "p50": p50,
                "p95": p95,
            },
            "parse_errors": self.parse_errors,
            "processing_errors": self.processing_errors,
            "quality_flag_counts": dict(sorted(self.quality_flag_counts.items())),
            "last_error": self.last_error,
        }

    def write(self, *, force: bool = False, status: str = "running") -> None:
        now = time.monotonic()
        if not force and now - self.last_write_monotonic < self.write_interval_seconds:
            return
        write_json_atomic(self.path, self.snapshot(status=status))
        self.last_write_monotonic = now

    def close(self, *, status: str = "stopped") -> None:
        self.write(force=True, status=status)


class LiveClassifierService:
    def __init__(
        self,
        estimator: HumanStateEstimator,
        publisher: OscPublisher,
        logger: "StateCsvLogger | None" = None,
        health: RuntimeHealthMonitor | None = None,
    ) -> None:
        self.estimator = estimator
        self.publisher = publisher
        self.logger = logger
        self.health = health

    def accept(self, sample: RawSample) -> HumanState | None:
        started = time.perf_counter()
        if self.health is not None:
            self.health.observe_sample(sample)
        try:
            state = self.estimator.push_sample(
                sample.timestamp_ms, sample.emg, sample.accel, sample.gyro,
                interpolated_imu=sample.interpolated_imu,
                sample_quality=sample.sample_quality,
                missing=sample.missing,
                channel_quality=sample.channel_quality,
                timestamp_valid=sample.timestamp_valid,
                imu_valid=sample.imu_valid,
                quality_gate_pass=sample.quality_gate_pass,
                duplicate_packet=sample.duplicate_packet,
                out_of_order_packet=sample.out_of_order_packet,
            )
            if state is not None:
                self.publisher.publish(state)
                if self.logger is not None:
                    self.logger.write(state, self.estimator.current_onset_lag_ms)
                if self.health is not None:
                    self.health.observe_state(state, (time.perf_counter() - started) * 1000.0)
            return state
        except Exception as exc:
            if self.health is not None:
                self.health.observe_error(exc)
            raise


class StateCsvLogger:
    """Append runtime observations used to fit the shadow consistency model."""

    FIELDS = (
        "run_id", "state_sequence", "recorded_at_utc", "timestamp_ms",
        "direction", "gesture", "arm_phase", "hand_phase",
        "activation", "motion", "onset_lag_ms", "q_direction", "q_gesture",
        "emg_quality", "imu_quality",
        "quality_flags",
        "direction_margin", "gesture_margin", "direction_drift", "gesture_drift",
    )

    def __init__(
        self,
        path: str | Path,
        *,
        run_id: str = "unspecified",
        fsync_every: int = 25,
    ) -> None:
        if fsync_every < 1:
            raise ValueError("fsync_every must be positive")
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        existed = output.exists() and output.stat().st_size > 0
        if existed:
            with output.open("r", encoding="utf-8", newline="") as existing:
                header = next(csv.reader(existing), [])
            if tuple(header) != self.FIELDS:
                raise ValueError(
                    "runtime CSV schema differs from the current model diagnostics; "
                    "use a new log file"
                )
        self._handle: TextIO = output.open("a", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._handle, fieldnames=self.FIELDS)
        self.run_id = str(run_id)
        self._state_sequence = 0
        self._fsync_every = int(fsync_every)
        self._pending_fsync = 0
        if not existed:
            self._writer.writeheader()
            self._handle.flush()

    def write(self, state: HumanState, onset_lag_ms: float | None) -> None:
        self._state_sequence += 1
        self._writer.writerow({
            "run_id": self.run_id,
            "state_sequence": self._state_sequence,
            "recorded_at_utc": _utc_now(),
            "timestamp_ms": state.timestamp_ms,
            "direction": int(state.direction),
            "gesture": int(state.gesture),
            "arm_phase": int(state.phase.arm),
            "hand_phase": int(state.phase.hand),
            "activation": state.activation,
            "motion": state.motion_intensity,
            "onset_lag_ms": "" if onset_lag_ms is None else onset_lag_ms,
            "q_direction": state.confidence.direction,
            "q_gesture": state.confidence.gesture,
            "emg_quality": state.signal_quality.emg,
            "imu_quality": state.signal_quality.imu,
            "quality_flags": int(state.signal_quality.flags),
            "direction_margin": state.direction_margin,
            "gesture_margin": state.gesture_margin,
            "direction_drift": state.direction_drift,
            "gesture_drift": state.gesture_drift,
        })
        self._handle.flush()
        self._pending_fsync += 1
        if self._pending_fsync >= self._fsync_every:
            os.fsync(self._handle.fileno())
            self._pending_fsync = 0

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.flush()
            os.fsync(self._handle.fileno())
        self._handle.close()

    def __enter__(self) -> "StateCsvLogger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
