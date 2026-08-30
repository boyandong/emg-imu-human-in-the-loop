from __future__ import annotations

import math

import numpy as np

from emgforce.filters import EmgDisplayFilterBank
from emgforce.processing.alignment_review import review_activity_envelopes


SAMPLE_RATE = 2000.0


def response_rms(frequency: float, duration: float = 4.0) -> float:
    samples = np.arange(int(SAMPLE_RATE * duration)) / SAMPLE_RATE
    source = np.sin(2 * math.pi * frequency * samples)
    bank = EmgDisplayFilterBank(channels=1, sample_rate=SAMPLE_RATE)
    output = np.asarray([bank.process((value,))[0] for value in source])
    discard = int(SAMPLE_RATE * 1.5)
    source_rms = float(np.sqrt(np.mean(source[discard:] ** 2)))
    output_rms = float(np.sqrt(np.mean(output[discard:] ** 2)))
    return output_rms / source_rms


def test_dc_offset_is_removed() -> None:
    bank = EmgDisplayFilterBank(channels=1, sample_rate=SAMPLE_RATE)
    output = [bank.process((250.0,))[0] for _ in range(8000)]
    assert abs(output[-1]) < 0.01


def test_20_hz_bandpass_rejects_5_hz() -> None:
    assert response_rms(5.0) < 0.08


def test_notches_reject_50_and_100_hz() -> None:
    assert response_rms(50.0) < 0.08
    assert response_rms(100.0) < 0.08


def test_midband_is_preserved() -> None:
    # Avoid the deliberately notched 50 Hz harmonics in the live display.
    assert response_rms(213.0) > 0.85


def test_upper_band_rolls_off_before_nyquist() -> None:
    response_800 = response_rms(800.0)
    response_950 = response_rms(950.0)
    assert response_800 > 0.65
    assert response_950 < response_800 * 0.35


def test_filter_bank_requires_eight_values_by_default() -> None:
    bank = EmgDisplayFilterBank()
    try:
        bank.process((1.0, 2.0))
    except ValueError as exc:
        assert "expected 8 channels" in str(exc)
    else:
        raise AssertionError("expected a channel-count error")


def test_alignment_review_hides_mains_sine_and_keeps_emg_burst() -> None:
    time = np.arange(int(SAMPLE_RATE * 1.3)) / SAMPLE_RATE
    mains = 5000.0 * np.sin(2 * math.pi * 50.0 * time)
    burst = np.zeros_like(time)
    active = (time >= 0.65) & (time < 0.82)
    burst[active] = 700.0 * np.sin(2 * math.pi * 213.0 * time[active])
    raw = np.repeat((mains + burst)[:, None], 8, axis=1)

    channels, combined = review_activity_envelopes(raw, SAMPLE_RATE)

    assert channels.shape == raw.shape
    assert float(np.median(combined[active])) > 20 * float(np.median(combined[:800]))
