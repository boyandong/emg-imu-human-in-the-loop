from __future__ import annotations

from dataclasses import dataclass, replace
from time import monotonic_ns
from typing import Literal


SYNC = b"\xD2\xD2\xD2"
FRAME_SIZE = 29
PAYLOAD_SIZE = 24
EMG_TYPE = 0xAA
IMU_TYPE = 0xBB


def signed24_be(data: bytes) -> int:
    """Decode one signed 24-bit big-endian integer."""
    if len(data) != 3:
        raise ValueError("signed24_be requires exactly 3 bytes")
    value = int.from_bytes(data, byteorder="big", signed=False)
    return value - (1 << 24) if value & (1 << 23) else value


@dataclass(frozen=True, slots=True)
class Packet:
    packet_type: Literal["EMG", "IMU"]
    sequence: int
    received_ns: int
    raw: bytes
    emg_uv: tuple[int, ...] | None = None
    gyro_rad_s: tuple[float, ...] | None = None
    accel_m_s2: tuple[float, ...] | None = None
    previous_sequence: int | None = None
    lost_before: int = 0
    duplicate: bool = False


@dataclass(slots=True)
class ParserStats:
    received_bytes: int = 0
    frames: int = 0
    emg_frames: int = 0
    imu_frames: int = 0
    lost_frames: int = 0
    duplicate_frames: int = 0
    discarded_bytes: int = 0


class FrameParser:
    """Streaming parser for the WAVELETECH 29-byte AA/BB protocol."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._last_sequence: int | None = None
        self.stats = ParserStats()

    def reset(self) -> None:
        self._buffer.clear()
        self._last_sequence = None
        self.stats = ParserStats()

    def feed(self, data: bytes, received_ns: int | None = None) -> list[Packet]:
        if not data:
            return []
        self.stats.received_bytes += len(data)
        self._buffer.extend(data)
        packets: list[Packet] = []
        stamp = received_ns if received_ns is not None else monotonic_ns()

        while True:
            sync_index = self._buffer.find(SYNC)
            if sync_index < 0:
                keep = min(len(self._buffer), len(SYNC) - 1)
                discarded = len(self._buffer) - keep
                if discarded:
                    del self._buffer[:discarded]
                    self.stats.discarded_bytes += discarded
                break
            if sync_index:
                del self._buffer[:sync_index]
                self.stats.discarded_bytes += sync_index
            if len(self._buffer) < FRAME_SIZE:
                break
            if self._buffer[3] not in (EMG_TYPE, IMU_TYPE):
                del self._buffer[0]
                self.stats.discarded_bytes += 1
                continue

            raw = bytes(self._buffer[:FRAME_SIZE])
            del self._buffer[:FRAME_SIZE]
            packet = self._decode(raw, stamp)
            previous, lost, duplicate = self._update_sequence(packet.sequence)
            packet = replace(packet, previous_sequence=previous,
                             lost_before=lost, duplicate=duplicate)
            self.stats.frames += 1
            if packet.packet_type == "EMG":
                self.stats.emg_frames += 1
            else:
                self.stats.imu_frames += 1
            packets.append(packet)
        return packets

    def _update_sequence(self, sequence: int) -> tuple[int | None, int, bool]:
        if self._last_sequence is None:
            self._last_sequence = sequence
            return None, 0, False
        previous = self._last_sequence
        delta = (sequence - self._last_sequence) & 0xFF
        if delta == 0:
            self.stats.duplicate_frames += 1
            return previous, 0, True
        elif delta <= 128:
            lost = delta - 1
            self.stats.lost_frames += lost
            self._last_sequence = sequence
            return previous, lost, False
        else:
            # Late/out-of-order packets do not move the continuity reference.
            self.stats.duplicate_frames += 1
            return previous, 0, True

    @staticmethod
    def _decode(raw: bytes, received_ns: int) -> Packet:
        sequence = raw[4]
        payload = raw[5:]
        if raw[3] == EMG_TYPE:
            channels = tuple(
                signed24_be(payload[offset : offset + 3])
                for offset in range(0, PAYLOAD_SIZE, 3)
            )
            return Packet("EMG", sequence, received_ns, raw, emg_uv=channels)

        # The first two BB payload bytes are reserved according to the manual.
        values = tuple(
            int.from_bytes(payload[offset : offset + 2], "big", signed=True)
            for offset in range(2, 14, 2)
        )
        gyro = tuple(value * 0.0012 for value in values[:3])
        accel = tuple(value * 0.0005978 for value in values[3:])
        return Packet(
            "IMU",
            sequence,
            received_ns,
            raw,
            gyro_rad_s=gyro,
            accel_m_s2=accel,
        )
