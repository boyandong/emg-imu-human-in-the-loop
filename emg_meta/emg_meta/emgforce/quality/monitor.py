from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class ChannelQuality:
    rms: float
    peak_to_peak: float
    saturated: bool
    status: str


class SignalQualityMonitor:
    """Lightweight contact/QC metrics only; never segments or validates gestures."""

    def __init__(self, saturation_level: int = 8_300_000) -> None:
        self.saturation_level = saturation_level

    def calculate(self, samples: np.ndarray) -> list[ChannelQuality]:
        data = np.asarray(samples, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != 8:
            raise ValueError("Signal QC 需要 [N, 8] EMG")
        if not len(data):
            return [ChannelQuality(0, 0, False, "NO DATA") for _ in range(8)]
        rms = np.sqrt(np.mean(np.square(data), axis=0))
        ptp = np.ptp(data, axis=0)
        saturated = np.any(np.abs(data) >= self.saturation_level, axis=0)
        return [ChannelQuality(float(rms[i]), float(ptp[i]), bool(saturated[i]),
                               "WARNING" if saturated[i] or ptp[i] == 0 else "GOOD")
                for i in range(8)]

    def report(self, samples: np.ndarray, sample_rate: float = 250.0) -> dict[str, Any]:
        """Return an auditable pre-session rest-signal report."""
        data = np.asarray(samples, dtype=np.float64)
        channels = self.calculate(data)
        if not len(data):
            return {"passed": False, "grade": "BAD", "reasons": ["no_data"]}
        median = np.median(data, axis=0)
        mad = np.median(np.abs(data - median), axis=0)
        p95 = np.percentile(np.abs(data - median), 95, axis=0)
        saturation_ratio = np.mean(np.abs(data) >= self.saturation_level, axis=0)
        zero_ratio = np.mean(data == 0, axis=0)
        centered = data - np.mean(data, axis=0)
        spectrum = np.abs(np.fft.rfft(centered, axis=0)) ** 2
        frequencies = np.fft.rfftfreq(len(data), 1.0 / sample_rate)
        band = (frequencies >= 45) & (frequencies <= 55)
        mains_ratio = spectrum[band].sum(axis=0) / np.maximum(spectrum.sum(axis=0), 1e-12)
        correlations = np.corrcoef(centered, rowvar=False) if len(data) > 2 else np.eye(8)
        adjacent = np.asarray([correlations[i, i + 1] for i in range(7)])
        reasons: list[str] = []
        if np.any(saturation_ratio > 0): reasons.append("clipping")
        if np.any(zero_ratio > 0.98) or np.any(np.ptp(data, axis=0) == 0): reasons.append("flatline")
        if np.any(mains_ratio > 0.35): reasons.append("50hz_interference")
        if np.any(np.abs(adjacent) > 0.995): reasons.append("adjacent_channel_correlation")
        passed = not reasons and all(item.status == "GOOD" for item in channels)
        return {
            "passed": passed, "grade": "GOOD" if passed else "BAD", "reasons": reasons,
            "duration_sec": len(data) / sample_rate, "sample_rate_hz": sample_rate,
            "channel_mad": mad.tolist(), "channel_p95": p95.tolist(),
            "saturation_ratio": saturation_ratio.tolist(), "zero_ratio": zero_ratio.tolist(),
            "mains_50hz_ratio": mains_ratio.tolist(),
            "adjacent_channel_correlation": adjacent.tolist(),
        }
