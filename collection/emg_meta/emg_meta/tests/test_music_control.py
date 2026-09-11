from __future__ import annotations

import struct

import numpy as np

from emgforce.music_control import (
    CALIBRATION_PHASES,
    EmgEffortEstimator,
    ImuMotionEstimator,
    REST_SETTLE_SECONDS,
    encode_music_state,
)


def test_calibration_instructions_define_repeatable_effort() -> None:
    fist_prompts = [phase.instruction for phase in CALIBRATION_PHASES if phase.kind == "fist"]
    assert len(fist_prompts) == 3
    assert all("7/10" in prompt for prompt in fist_prompts)
    assert [phase.duration for phase in CALIBRATION_PHASES] == [
        4.0, 2.6, 1.6, 2.6, 1.6, 2.6, 1.6,
    ]
    assert round(sum(phase.duration for phase in CALIBRATION_PHASES), 1) == 16.6
    assert REST_SETTLE_SECONDS == 1.0
    assert "不要用力撑开" in CALIBRATION_PHASES[0].instruction


def test_personal_effort_calibration_maps_rest_and_comfortable_fist() -> None:
    rng = np.random.default_rng(42)
    estimator = EmgEffortEstimator(window_samples=80)
    estimator.begin_calibration()
    for _ in range(12):
        estimator.ingest(rng.normal(0, 8, size=(80, 8)), "rest")
    for _ in range(12):
        estimator.ingest(rng.normal(0, 120, size=(80, 8)), "fist")
    baseline, active = estimator.finish_calibration()

    assert active > baseline * 5
    assert estimator.normalize(baseline) == 0.0
    assert 0.84 < estimator.normalize(active) < 0.86
    assert 0.38 < estimator.normalize((baseline + active) / 2) < 0.47
    above_comfortable = baseline + (active - baseline) / 0.85
    assert estimator.normalize(above_comfortable) == 1.0


def test_brief_opening_burst_does_not_raise_rest_baseline() -> None:
    rng = np.random.default_rng(7)
    estimator = EmgEffortEstimator(window_samples=80)
    estimator.begin_calibration()
    for _ in range(4):
        estimator.ingest(rng.normal(0, 250, size=(80, 8)), "rest")
    for _ in range(16):
        estimator.ingest(rng.normal(0, 8, size=(80, 8)), "rest")
    for _ in range(12):
        estimator.ingest(rng.normal(0, 120, size=(80, 8)), "fist")
    baseline, _ = estimator.finish_calibration()
    assert baseline < 15


def test_imu_motion_is_low_when_still_and_rises_during_motion() -> None:
    estimator = ImuMotionEstimator()
    still_gyro = np.zeros((40, 3), dtype=np.float32)
    still_accel = np.tile([0.0, 0.0, 9.81], (40, 1)).astype(np.float32)
    for _ in range(5):
        still = estimator.ingest(still_gyro, still_accel)
    moving = estimator.ingest(
        np.tile([2.0, 1.0, 0.8], (40, 1)),
        still_accel + np.tile([2.5, 0.8, 0.4], (40, 1)),
    )
    assert still < 0.05
    assert moving > still + 0.15
    assert 0.0 <= moving <= 1.0


def test_music_state_packet_uses_expected_osc_signature() -> None:
    packet = encode_music_state(1234567890123, 7, 0.9, 0.4, 0.8)
    assert packet.startswith(b"/emgimu/state\x00")
    assert b",hifff\x00" in packet
    assert struct.pack(">q", 1234567890123) in packet
