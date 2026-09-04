from __future__ import annotations

from dataclasses import dataclass
import socket
import struct
import time

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal, Slot

from .config import EMG_CHANNELS


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class EmgEffortEstimator:
    """Personal EMG envelope calibration and continuous 0..1 effort estimate."""

    def __init__(self, channels: int = EMG_CHANNELS, window_samples: int = 200,
                 comfortable_target: float = 0.85) -> None:
        self.channels = int(channels)
        self.window_samples = int(window_samples)
        self.comfortable_target = _clamp01(comfortable_target)
        if not 0.5 <= self.comfortable_target < 1.0:
            raise ValueError("7/10 握拳目标值必须在 0.5（含）到 1.0（不含）之间")
        self._buffer = np.empty((0, self.channels), dtype=np.float64)
        self._rest_levels: list[float] = []
        self._active_levels: list[float] = []
        self.baseline: float | None = None
        self.comfortable_active: float | None = None
        self.activation = 0.0

    @property
    def calibrated(self) -> bool:
        return self.baseline is not None and self.comfortable_active is not None

    def begin_calibration(self) -> None:
        self._rest_levels.clear()
        self._active_levels.clear()
        self.baseline = None
        self.comfortable_active = None
        self.activation = 0.0

    def ingest(self, raw: np.ndarray, collect: str | None = None) -> float:
        values = np.asarray(raw, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.channels:
            raise ValueError(f"EMG 数据必须为 N×{self.channels}")
        if not len(values):
            return self.activation

        self._buffer = np.concatenate((self._buffer, values), axis=0)[-self.window_samples :]
        if len(self._buffer) < max(16, self.window_samples // 4):
            return self.activation

        level = self._envelope_level(self._buffer)
        if collect == "rest":
            self._rest_levels.append(level)
        elif collect == "fist":
            self._active_levels.append(level)

        target = self.normalize(level) if self.calibrated else 0.0
        # A relaxed hand should fade naturally; onset remains responsive.
        alpha = 0.34 if target >= self.activation else 0.10
        self.activation += alpha * (target - self.activation)
        self.activation = _clamp01(self.activation)
        return self.activation

    @staticmethod
    def _envelope_level(values: np.ndarray) -> float:
        centered = values - np.median(values, axis=0, keepdims=True)
        channel_rms = np.sqrt(np.mean(np.square(centered), axis=0))
        return float(np.percentile(channel_rms, 75.0))

    def finish_calibration(self) -> tuple[float, float]:
        if len(self._rest_levels) < 3 or len(self._active_levels) < 3:
            raise ValueError("收到的标定数据不足，请保持设备连接后重试")
        # The median rejects the brief activation burst that can occur while
        # the fingers are transitioning into a naturally open resting pose.
        baseline = float(np.median(self._rest_levels))
        active = float(np.percentile(self._active_levels, 65.0))
        required_gap = max(1.0, baseline * 0.25)
        if not np.isfinite(baseline + active) or active - baseline < required_gap:
            raise ValueError("握拳信号与放松基线过于接近，请检查佩戴并以主观 7/10 力度重试")
        self.baseline = baseline
        self.comfortable_active = active
        self.activation = 0.0
        return baseline, active

    def normalize(self, level: float) -> float:
        if not self.calibrated:
            return 0.0
        assert self.baseline is not None and self.comfortable_active is not None
        comfortable_span = self.comfortable_active - self.baseline
        full_span = comfortable_span / self.comfortable_target
        return _clamp01((float(level) - self.baseline) / full_span)


class ImuMotionEstimator:
    """Removes gravity and turns linear acceleration/rotation into 0..1 motion."""

    def __init__(self) -> None:
        self.gravity: np.ndarray | None = None
        self.motion = 0.0

    def ingest(self, gyro: np.ndarray, accel: np.ndarray) -> float:
        gyro_values = np.asarray(gyro, dtype=np.float64)
        accel_values = np.asarray(accel, dtype=np.float64)
        if gyro_values.ndim != 2 or accel_values.ndim != 2:
            raise ValueError("IMU 数据必须为二维数组")
        if gyro_values.shape != accel_values.shape or gyro_values.shape[1] != 3:
            raise ValueError("陀螺仪和加速度数据必须同为 N×3")
        if not len(gyro_values):
            return self.motion

        if self.gravity is None:
            self.gravity = accel_values[0].copy()
        linear_norms: list[float] = []
        for row in accel_values:
            self.gravity += 0.035 * (row - self.gravity)
            linear_norms.append(float(np.linalg.norm(row - self.gravity)))
        linear = float(np.sqrt(np.mean(np.square(linear_norms))))
        rotation = float(np.sqrt(np.mean(np.sum(np.square(gyro_values), axis=1))))
        linear_score = _clamp01((linear - 0.08) / 2.8)
        rotation_score = _clamp01((rotation - 0.03) / 2.6)
        target = _clamp01(0.58 * linear_score + 0.42 * rotation_score)
        alpha = 0.38 if target >= self.motion else 0.13
        self.motion += alpha * (target - self.motion)
        self.motion = _clamp01(self.motion)
        return self.motion


@dataclass(frozen=True, slots=True)
class CalibrationPhase:
    kind: str
    duration: float
    instruction: str


REST_SETTLE_SECONDS = 1.0


CALIBRATION_PHASES = (
    CalibrationPhase("rest", 4.0, "自然松手，手指松散张开，不要用力撑开；前 1 秒等待稳定"),
    CalibrationPhase("fist", 2.6, "第 1 次：手指完全合拢，以主观 7/10 力度稳定握拳"),
    CalibrationPhase("release", 1.6, "自然松手，不要用力撑开"),
    CalibrationPhase("fist", 2.6, "第 2 次：主观 7/10 稳定握拳，不颤抖、不疼痛"),
    CalibrationPhase("release", 1.6, "自然松手，不要用力撑开"),
    CalibrationPhase("fist", 2.6, "第 3 次：重复相同的主观 7/10 握拳力度"),
    CalibrationPhase("release", 1.6, "自然松手并保持放松，不要用力撑开"),
)


def _osc_string(value: str) -> bytes:
    data = value.encode("utf-8") + b"\x00"
    return data + b"\x00" * ((4 - len(data) % 4) % 4)


def encode_music_state(
    timestamp_ms: int,
    gesture: int,
    confidence: float,
    motion: float,
    effort: float,
) -> bytes:
    return (
        _osc_string("/emgimu/state")
        + _osc_string(",hifff")
        + struct.pack(">qifff", int(timestamp_ms), int(gesture), float(confidence),
                      _clamp01(motion), _clamp01(effort))
    )


class MusicControlBridge(QObject):
    """Qt bridge from acquisition streams to the local generative music engine."""

    values_changed = Signal(float, float)
    calibration_changed = Signal(str, int, bool)
    output_changed = Signal(str)

    def __init__(self, parent: QObject | None = None, host: str = "127.0.0.1",
                 port: int = 9000) -> None:
        super().__init__(parent)
        self.effort = EmgEffortEstimator()
        self.motion = ImuMotionEstimator()
        self.target = (host, int(port))
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.connected = False
        self._calibrating = False
        self._phase_index = 0
        self._phase_started = 0.0
        self._gesture = 0
        self._confidence = 1.0
        self._gesture_updated = 0.0
        self._last_emg = 0.0
        self._last_motion = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    @Slot(bool)
    def set_connected(self, connected: bool) -> None:
        self.connected = bool(connected)
        if not connected and self._calibrating:
            self._calibrating = False
            self.calibration_changed.emit("设备已断开，力度标定已取消", 0, False)
        self.output_changed.emit(
            "等待个人力度标定" if connected and not self.effort.calibrated
            else "设备未连接" if not connected
            else "正在向音乐引擎发送控制"
        )

    @Slot()
    def start_calibration(self) -> None:
        if not self.connected:
            self.calibration_changed.emit("请先连接手环", 0, False)
            return
        self.effort.begin_calibration()
        self._calibrating = True
        self._phase_index = 0
        self._phase_started = time.monotonic()
        self.calibration_changed.emit(CALIBRATION_PHASES[0].instruction, 0, False)
        self.output_changed.emit("力度标定进行中")

    @Slot(object, object)
    def ingest_emg(self, raw: np.ndarray, _index: np.ndarray) -> None:
        collect: str | None = None
        if self._calibrating:
            kind = CALIBRATION_PHASES[self._phase_index].kind
            elapsed = time.monotonic() - self._phase_started
            if kind == "fist":
                collect = "fist"
            elif kind == "rest" and elapsed >= REST_SETTLE_SECONDS:
                collect = "rest"
        self._last_emg = self.effort.ingest(raw, collect)
        self.values_changed.emit(self._last_emg, self._last_motion)

    @Slot(object, object, object)
    def ingest_imu(self, gyro: np.ndarray, accel: np.ndarray, _timestamps: np.ndarray) -> None:
        self._last_motion = self.motion.ingest(gyro, accel)
        self.values_changed.emit(self._last_emg, self._last_motion)

    @Slot(int, float)
    def set_gesture(self, gesture: int, confidence: float) -> None:
        self._gesture = int(gesture) if 0 <= int(gesture) <= 8 else 0
        self._confidence = _clamp01(confidence)
        self._gesture_updated = time.monotonic()

    def _tick(self) -> None:
        now = time.monotonic()
        if self._calibrating:
            phase = CALIBRATION_PHASES[self._phase_index]
            elapsed = now - self._phase_started
            total_duration = sum(item.duration for item in CALIBRATION_PHASES)
            completed = sum(item.duration for item in CALIBRATION_PHASES[:self._phase_index])
            progress = round(100.0 * min(total_duration, completed + elapsed) / total_duration)
            self.calibration_changed.emit(phase.instruction, progress, False)
            if elapsed >= phase.duration:
                self._phase_index += 1
                if self._phase_index >= len(CALIBRATION_PHASES):
                    self._finish_calibration()
                else:
                    self._phase_started = now
                    next_phase = CALIBRATION_PHASES[self._phase_index]
                    self.calibration_changed.emit(next_phase.instruction, progress, False)

        if not self.connected or not self.effort.calibrated:
            return
        if now - self._gesture_updated > 0.45:
            self._gesture = 0
            self._confidence = 1.0
        packet = encode_music_state(
            int(time.time() * 1000), self._gesture, self._confidence,
            self._last_motion, self._last_emg,
        )
        try:
            self.socket.sendto(packet, self.target)
        except OSError as exc:
            self.output_changed.emit(f"音乐控制发送失败：{exc}")

    def _finish_calibration(self) -> None:
        self._calibrating = False
        try:
            baseline, active = self.effort.finish_calibration()
        except ValueError as exc:
            self.calibration_changed.emit(str(exc), 0, False)
            self.output_changed.emit("力度标定失败，请重试")
            return
        self.calibration_changed.emit(
            f"标定完成：放松 {baseline:.0f}，7/10 握拳 {active:.0f}（映射为 0.85）", 100, True)
        self.output_changed.emit("正在向音乐引擎发送控制 · 127.0.0.1:9000")

    def close(self) -> None:
        self._timer.stop()
        self.socket.close()
