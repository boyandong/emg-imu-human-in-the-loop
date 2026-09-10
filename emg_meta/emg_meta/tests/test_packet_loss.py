from __future__ import annotations

from emgforce.protocol import FrameParser


def _signed24(value: int) -> bytes:
    return (value % (1 << 24)).to_bytes(3, "big")


def emg_frame(sequence: int, values: tuple[int, ...]) -> bytes:
    return b"\xD2\xD2\xD2\xAA" + bytes([sequence]) + b"".join(map(_signed24, values))


def imu_frame(sequence: int) -> bytes:
    return b"\xD2\xD2\xD2\xBB" + bytes([sequence]) + bytes(24)


def test_packet_loss_annotation_contains_exact_sequence_gap() -> None:
    parser = FrameParser()
    parser.feed(emg_frame(250, (0,) * 8))
    packet = parser.feed(imu_frame(253))[0]
    assert packet.previous_sequence == 250
    assert packet.lost_before == 2
    assert not packet.duplicate


def test_duplicate_is_annotated_and_rollover_is_not_loss() -> None:
    parser = FrameParser()
    parser.feed(emg_frame(255, (0,) * 8))
    rollover = parser.feed(emg_frame(0, (0,) * 8))[0]
    duplicate = parser.feed(emg_frame(0, (0,) * 8))[0]
    assert rollover.lost_before == 0
    assert duplicate.duplicate


def test_late_packet_is_counted_separately_from_duplicate() -> None:
    parser = FrameParser()
    parser.feed(emg_frame(10, (0,) * 8))
    parser.feed(emg_frame(12, (0,) * 8))
    late = parser.feed(emg_frame(11, (0,) * 8))[0]
    assert late.out_of_order and not late.duplicate
    assert parser.stats.out_of_order_frames == 1
