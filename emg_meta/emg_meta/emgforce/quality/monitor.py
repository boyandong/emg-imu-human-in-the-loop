from __future__ import annotations

from dataclasses import dataclass

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
