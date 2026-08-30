from __future__ import annotations

import logging

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from .protocol import Packet, ParserStats
from .sample_clock import SampleIndexClock
from .storage.hdf5_recorder import Hdf5Recorder


LOGGER = logging.getLogger(__name__)


class AcquisitionController(QObject):
    """Fan-out point for decoded frames: lossless recorder queue and droppable GUI."""

    emg_display_ready = Signal(object, object)
    imu_display_ready = Signal(object, object, object)
    packet_loss = Signal(int, int, int)
    statistics_ready = Signal(object)
    recording_error = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.recorder: Hdf5Recorder | None = None
        self.clock = SampleIndexClock()
        self._recording = False

    @property
    def current_sample_index(self) -> int:
        """Next EMG row index at the current session boundary."""
        return self.clock.session_index

    def begin_recording(self, recorder: Hdf5Recorder) -> None:
        self.recorder = recorder
        self.clock.begin_session()
        self._recording = True

    def end_recording(self) -> None:
        self._recording = False
        self.recorder = None

    @Slot(object)
    def ingest_packets(self, packets: list[Packet]) -> None:
        emg_packets = [p for p in packets if p.packet_type == "EMG"
                       and p.emg_uv is not None and not p.duplicate]
        imu_packets = [p for p in packets if p.packet_type == "IMU"
                       and p.gyro_rad_s is not None and not p.duplicate]

        for packet in packets:
            if packet.lost_before:
                self.packet_loss.emit(packet.lost_before,
                                      int(packet.previous_sequence), packet.sequence)

        if emg_packets:
            raw = np.asarray([p.emg_uv for p in emg_packets], dtype=np.int32)
            global_index, session_index = self.clock.allocate(len(raw))
            if self._recording and self.recorder is not None:
                seq = np.asarray([p.sequence for p in emg_packets], dtype=np.uint8)
                try:
                    self.recorder.enqueue_emg(raw, session_index, seq)
                except Exception as exc:
                    self._recording = False
                    self.recording_error.emit(str(exc))
            self.emg_display_ready.emit(raw, global_index)

        if imu_packets:
            gyro = np.asarray([p.gyro_rad_s for p in imu_packets], dtype=np.float32)
            accel = np.asarray([p.accel_m_s2 for p in imu_packets], dtype=np.float32)
            if self._recording and self.recorder is not None:
                count = len(imu_packets)
                try:
                    self.recorder.enqueue_imu(
                        gyro, accel,
                        np.asarray([p.received_ns for p in imu_packets], dtype=np.int64),
                        np.asarray([p.sequence for p in imu_packets], dtype=np.uint8),
                        np.full(count, self.current_sample_index, dtype=np.int64),
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
        self.statistics_ready.emit(stats)
