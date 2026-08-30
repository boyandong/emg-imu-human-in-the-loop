from __future__ import annotations

import math
import random
import threading
import time
from dataclasses import dataclass

import serial
from PySide6.QtCore import QThread, Signal

from .config import BAUDRATE, SAMPLING_RATE, SERIAL_TIMEOUT_SEC
from .protocol import FrameParser, Packet


@dataclass(frozen=True, slots=True)
class AcquisitionConfig:
    port: str
    baudrate: int = BAUDRATE
    simulated: bool = False
    emg_sample_rate: float = SAMPLING_RATE


class AcquisitionWorker(QThread):
    """Serial decoding only. It never touches GUI state, HDF5, or prompts."""

    packets_ready = Signal(object)
    connected = Signal(str)
    disconnected = Signal(str)
    error = Signal(str)
    stats_ready = Signal(object)

    def __init__(self, config: AcquisitionConfig, parent=None) -> None:
        super().__init__(parent)
        self.config = config
        self.parser = FrameParser()
        self._stop_event = threading.Event()
        self._serial: serial.Serial | None = None
        self._sim_sequence = 0

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        self._stop_event.clear()
        self.parser.reset()
        try:
            if self.config.simulated:
                self.connected.emit("模拟设备")
                self._run_simulator()
            else:
                self._serial = serial.Serial(
                    port=self.config.port, baudrate=self.config.baudrate,
                    bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE, timeout=SERIAL_TIMEOUT_SEC,
                    write_timeout=0.5, rtscts=False, dsrdtr=False, xonxoff=False,
                )
                self.connected.emit(self.config.port)
                self._run_serial()
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            if self._serial is not None:
                try:
                    self._serial.close()
                except Exception:
                    pass
                self._serial = None
            self.disconnected.emit("设备已断开")

    def _run_serial(self) -> None:
        assert self._serial is not None
        last_stats = time.monotonic()
        while not self._stop_event.is_set():
            waiting = self._serial.in_waiting
            data = self._serial.read(min(max(waiting, 1), 16384))
            if data:
                packets = self.parser.feed(data, time.monotonic_ns())
                if packets:
                    self.packets_ready.emit(packets)
            now = time.monotonic()
            if now - last_stats >= 0.5:
                self.stats_ready.emit(self.parser.stats)
                last_stats = now

    def _run_simulator(self) -> None:
        sample = 0
        next_tick = time.perf_counter()
        last_stats = time.monotonic()
        while not self._stop_event.is_set():
            batch: list[Packet] = []
            for _ in range(40):
                t = sample / self.config.emg_sample_rate
                emg = tuple(int(
                    150 * math.sin(2 * math.pi * (18 + channel * 3) * t)
                    + random.gauss(0, 24)
                    + (600 * math.sin(2 * math.pi * 85 * t) if int(t) % 4 == 2 else 0)
                ) for channel in range(8))
                batch.extend(self.parser.feed(self._make_emg_frame(emg), time.monotonic_ns()))
                sample += 1
                if sample % 20 == 0:
                    batch.extend(self.parser.feed(self._make_imu_frame(t), time.monotonic_ns()))
            self.packets_ready.emit(batch)
            now = time.monotonic()
            if now - last_stats >= 0.5:
                self.stats_ready.emit(self.parser.stats)
                last_stats = now
            next_tick += 0.02
            self._stop_event.wait(max(0.0, next_tick - time.perf_counter()))
            if next_tick < time.perf_counter() - 0.02:
                next_tick = time.perf_counter()

    def _header(self, packet_type: int) -> bytearray:
        frame = bytearray(b"\xD2\xD2\xD2")
        frame.extend((packet_type, self._sim_sequence))
        self._sim_sequence = (self._sim_sequence + 1) & 0xFF
        return frame

    def _make_emg_frame(self, values: tuple[int, ...]) -> bytes:
        frame = self._header(0xAA)
        for value in values:
            frame.extend((value % (1 << 24)).to_bytes(3, "big"))
        return bytes(frame)

    def _make_imu_frame(self, t: float) -> bytes:
        frame = self._header(0xBB)
        payload = bytearray(24)
        physical = (0.5 * math.sin(t), 0.4 * math.cos(t * 0.7),
                    0.25 * math.sin(t * 1.3), 0.2 * math.sin(t),
                    0.3 * math.cos(t), 9.81 + 0.1 * math.sin(t * 2))
        scales = (0.0012,) * 3 + (0.0005978,) * 3
        for index, (value, scale) in enumerate(zip(physical, scales)):
            raw = max(-32768, min(32767, round(value / scale)))
            payload[2 + index * 2:4 + index * 2] = raw.to_bytes(2, "big", signed=True)
        frame.extend(payload)
        return bytes(frame)
