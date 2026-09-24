from __future__ import annotations

import logging
import time
from dataclasses import asdict

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from .config import IMU_SAMPLING_RATE, SAMPLING_RATE
from .protocol import Packet, ParserStats
from .sample_clock import SampleIndexClock
from .storage.hdf5_recorder import Hdf5Recorder
from .quality.monitor import SignalQualityMonitor


LOGGER = logging.getLogger(__name__)


class AcquisitionController(QObject):
    """Fan-out point for decoded frames: lossless recorder queue and droppable GUI."""

    emg_display_ready = Signal(object, object)
    emg_inference_ready = Signal(object, object, object)
    imu_display_ready = Signal(object, object, object)
    packet_loss = Signal(int, int, int)
    statistics_ready = Signal(object)
    recording_error = Signal(str)
    quality_alert = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.recorder: Hdf5Recorder | None = None
        self.clock = SampleIndexClock()
        self._recording = False
        self._emg_time_origin_ns: int | None = None
        self._imu_time_origin_ns: int | None = None
        self._imu_session_index = 0
        self._last_stats = ParserStats()
        self._stats_baseline = ParserStats()
        self._quality_window = np.empty((0, 8), dtype=np.int32)
        self._last_quality_alert_ns = 0

    @property
    def current_sample_index(self) -> int:
        """Next EMG row index at the current session boundary."""
        return self.clock.session_index

    def begin_recording(self, recorder: Hdf5Recorder) -> None:
        self.recorder = recorder
        self.clock.begin_session()
        self._emg_time_origin_ns = None
        self._imu_time_origin_ns = None
        self._imu_session_index = 0
        self._stats_baseline = ParserStats(**asdict(self._last_stats))
        self._quality_window = np.empty((0, 8), dtype=np.int32)
        self._last_quality_alert_ns = 0
        self._recording = True

    def recording_summary(self) -> dict[str, float | int]:
        emg_count = self.clock.session_index
        imu_count = self._imu_session_index
        return {
            "measured_emg_rate_hz": self._effective_rate(emg_count, self._emg_time_origin_ns),
            "measured_imu_rate_hz": self._effective_rate(imu_count, self._imu_time_origin_ns),
            "received_frames": max(0, self._last_stats.frames - self._stats_baseline.frames),
            "lost_frames": max(0, self._last_stats.lost_frames - self._stats_baseline.lost_frames),
            "duplicate_frames": max(0, self._last_stats.duplicate_frames - self._stats_baseline.duplicate_frames),
            "out_of_order_frames": max(0, self._last_stats.out_of_order_frames - self._stats_baseline.out_of_order_frames),
        }

    @staticmethod
    def _effective_rate(count: int, origin_ns: int | None) -> float:
        if count < 2 or origin_ns is None:
            return 0.0
        elapsed = (time.monotonic_ns() - origin_ns) / 1e9
        return float((count - 1) / elapsed) if elapsed > 0 else 0.0

    def end_recording(self) -> None:
        self._recording = False
        self.recorder = None

    def reset_trial_quality_window(self) -> None:
        self._quality_window = np.empty((0, 8), dtype=np.int32)
        self._last_quality_alert_ns = 0

    @Slot(object)
    def ingest_packets(self, packets: list[Packet]) -> None:
        if self._recording and self.recorder is not None and packets:
            audit_dtype = np.dtype([
                ("packet_type", "u1"), ("packet_seq", "u1"),
                ("pc_received_ns", "i8"), ("lost_before", "i4"),
                ("duplicate", "?"), ("out_of_order", "?"),
            ])
            audit = np.asarray([
                (1 if p.packet_type == "EMG" else 2, p.sequence, p.received_ns,
                 p.lost_before, p.duplicate, p.out_of_order) for p in packets
            ], dtype=audit_dtype)
            try:
                self.recorder.enqueue_packet_audit(audit)
            except Exception as exc:
                self._recording = False
                self.recording_error.emit(str(exc))
        emg_packets = [p for p in packets if p.packet_type == "EMG"
                       and p.emg_uv is not None and not p.duplicate and not p.out_of_order]
        imu_packets = [p for p in packets if p.packet_type == "IMU"
                       and p.gyro_rad_s is not None and not p.duplicate and not p.out_of_order]

        for packet in packets:
            if packet.lost_before:
                self.packet_loss.emit(packet.lost_before,
                                      int(packet.previous_sequence), packet.sequence)

        if emg_packets:
            raw = np.asarray([p.emg_uv for p in emg_packets], dtype=np.int32)
            global_index, session_index = self.clock.allocate(len(raw))
            received_ns = np.asarray([p.received_ns for p in emg_packets], dtype=np.int64)
            if self._emg_time_origin_ns is None:
                self._emg_time_origin_ns = int(received_ns[0])
            sample_time_ns = self._emg_time_origin_ns + np.rint(
                session_index * (1e9 / SAMPLING_RATE)).astype(np.int64)
            if self._recording and self.recorder is not None:
                seq = np.asarray([p.sequence for p in emg_packets], dtype=np.uint8)
                try:
                    self.recorder.enqueue_emg(raw, session_index, seq,
                                              received_ns, sample_time_ns)
                except Exception as exc:
                    self._recording = False
                    self.recording_error.emit(str(exc))
            self.emg_display_ready.emit(raw, global_index)
            self.emg_inference_ready.emit(raw, global_index, received_ns)
            if self._recording:
                self._quality_window = np.concatenate((self._quality_window, raw), axis=0)[-2 * SAMPLING_RATE:]
                if len(self._quality_window) >= 2 * SAMPLING_RATE:
                    reasons = SignalQualityMonitor().continuous_reasons(self._quality_window)
                    now = time.monotonic_ns()
                    if reasons and now - self._last_quality_alert_ns >= 2_000_000_000:
                        self._last_quality_alert_ns = now
                        self.quality_alert.emit(",".join(reasons))

        if imu_packets:
            gyro = np.asarray([p.gyro_rad_s for p in imu_packets], dtype=np.float32)
            accel = np.asarray([p.accel_m_s2 for p in imu_packets], dtype=np.float32)
            received_ns = np.asarray([p.received_ns for p in imu_packets], dtype=np.int64)
            imu_index = np.arange(self._imu_session_index,
                                  self._imu_session_index + len(imu_packets), dtype=np.int64)
            if self._imu_time_origin_ns is None:
                self._imu_time_origin_ns = int(received_ns[0])
            sample_time_ns = self._imu_time_origin_ns + np.rint(
                imu_index * (1e9 / IMU_SAMPLING_RATE)).astype(np.int64)
            self._imu_session_index += len(imu_packets)
            if self._recording and self.recorder is not None:
                count = len(imu_packets)
                try:
                    self.recorder.enqueue_imu(
                        gyro, accel,
                        received_ns,
                        np.asarray([p.sequence for p in imu_packets], dtype=np.uint8),
                        np.full(count, self.current_sample_index, dtype=np.int64),
                        sample_time_ns,
                    )
                except Exception as exc:
                    self._recording = False
                    self.recording_error.emit(str(exc))
            self.imu_display_ready.emit(
                gyro,
                accel,
                np.asarray([p.received_ns for p in imu_packets], dtype=np.int64),
            )

    @Slot(object)
    def update_statistics(self, stats: ParserStats) -> None:
        self._last_stats = stats
        self.statistics_ready.emit(stats)
