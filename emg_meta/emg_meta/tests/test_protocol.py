from __future__ import annotations

from emgforce.protocol import FrameParser, signed24_be


def encode_signed24(value: int) -> bytes:
    if value < 0:
        value += 1 << 24
    return value.to_bytes(3, "big")


def emg_frame(sequence: int, values: tuple[int, ...]) -> bytes:
    return b"\xD2\xD2\xD2\xAA" + bytes([sequence]) + b"".join(
        encode_signed24(value) for value in values
    )


def imu_frame(sequence: int) -> bytes:
    payload = bytearray(24)
    raw_values = (1000, -1000, 250, 100, -200, 300)
    for index, value in enumerate(raw_values):
        payload[2 + index * 2 : 4 + index * 2] = value.to_bytes(2, "big", signed=True)
    return b"\xD2\xD2\xD2\xBB" + bytes([sequence]) + bytes(payload)


def test_signed24_boundaries() -> None:
    assert signed24_be(b"\x00\x00\x00") == 0
    assert signed24_be(b"\x7f\xff\xff") == 8_388_607
    assert signed24_be(b"\x80\x00\x00") == -8_388_608
    assert signed24_be(b"\xff\xff\xff") == -1


def test_streaming_parser_handles_noise_and_split_frames() -> None:
    parser = FrameParser()
    first = emg_frame(254, (1, -2, 3, -4, 5, -6, 7, -8))
    second = imu_frame(255)
    third = emg_frame(0, (11, 12, 13, 14, 15, 16, 17, 18))

    assert parser.feed(b"noise" + first[:8]) == []
    packets = parser.feed(first[8:] + second + third[:10])
    packets += parser.feed(third[10:])

    assert [packet.packet_type for packet in packets] == ["EMG", "IMU", "EMG"]
    assert packets[0].emg_uv == (1, -2, 3, -4, 5, -6, 7, -8)
    assert packets[1].gyro_rad_s == (1.2, -1.2, 0.3)
    assert packets[2].sequence == 0
    assert parser.stats.frames == 3
    assert parser.stats.received_bytes == 92
    assert parser.stats.lost_frames == 0
    assert parser.stats.discarded_bytes == 5


def test_sequence_gap_is_counted() -> None:
    parser = FrameParser()
    parser.feed(emg_frame(10, (0,) * 8))
    parser.feed(emg_frame(13, (0,) * 8))
    assert parser.stats.lost_frames == 2


def test_invalid_type_resynchronizes() -> None:
    parser = FrameParser()
    invalid = b"\xD2\xD2\xD2\xCC" + bytes(25)
    packets = parser.feed(invalid + emg_frame(1, (9,) * 8))
    assert len(packets) == 1
    assert packets[0].emg_uv == (9,) * 8
    assert parser.stats.discarded_bytes >= 29
