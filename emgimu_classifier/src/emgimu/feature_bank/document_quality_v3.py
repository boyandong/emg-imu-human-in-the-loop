"""Opt-in F9a–f observations with explicit availability and exact denominators.

These are observations, not a gesture predictor or an approved Unknown gate.
The older quality family and its saved results remain unchanged.
"""

from __future__ import annotations

import numpy as np

from .core import FeatureBatch, FeatureFamily


EPS = 1e-10


def _covariance(x: np.ndarray) -> np.ndarray:
    centered = x - x.mean(axis=1, keepdims=True)
    return np.einsum("ntc,ntd->ncd", centered, centered) / max(x.shape[1] - 1, 1)


def _longest_strict_flatline(x: np.ndarray, tolerance: np.ndarray) -> np.ndarray:
    edges = np.abs(np.diff(x, axis=1)) < tolerance[None, None, :]
    current = np.zeros((len(x), x.shape[2]), dtype=np.int64)
    longest = current.copy()
    for step in range(edges.shape[1]):
        current = np.where(edges[:, step], current + 1, 0)
        longest = np.maximum(longest, current)
    return longest / x.shape[1]


class DocumentQualityObservationsV3(FeatureFamily):
    family_id = "F9_document_observations_v3"

    def __init__(self, *, adc_range: tuple[float, float] | None = None,
                 line_frequency_hz: int | None = None, pre_highpass_available: bool = False,
                 ring_order: tuple[int, ...] | None = None,
                 zero_threshold: float | None = None, flat_threshold: float | None = None,
                 low_frequency_hz: float = 10.0) -> None:
        if adc_range is not None and (len(adc_range) != 2 or not np.isfinite(adc_range).all()
                                      or adc_range[0] >= adc_range[1]):
            raise ValueError("ADC range must be measured metadata, min < max")
        if line_frequency_hz not in (None, 50, 60):
            raise ValueError("Known line frequency must be 50 or 60 Hz")
        for value in (zero_threshold, flat_threshold):
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError("Explicit quantization threshold must be positive")
        if not np.isfinite(low_frequency_hz) or not 0 < low_frequency_hz < 20:
            raise ValueError("Low-frequency bound must lie strictly below the 20 Hz valid band")
        self.adc_range = adc_range
        self.line_frequency_hz = line_frequency_hz
        self.pre_highpass_available = bool(pre_highpass_available)
        self.ring_order = ring_order
        self.zero_threshold = zero_threshold
        self.flat_threshold = flat_threshold
        self.low_frequency_hz = float(low_frequency_hz)

    def fit(self, batch: FeatureBatch, labels=None) -> "DocumentQualityObservationsV3":
        x = np.asarray(batch.emg, dtype=np.float64)
        if x.ndim != 3 or not len(x) or x.shape[1] < 2 or not np.isfinite(x).all():
            raise ValueError("finite source EMG windows with at least two samples required")
        channels = x.shape[2]
        if self.ring_order is not None and (len(self.ring_order) != channels
                                            or set(self.ring_order) != set(range(channels))):
            raise ValueError("ring order must be an explicit permutation of all channels")
        self.channels_ = channels
        self.samples_ = x.shape[1]
        self.rate_ = batch.sample_rate_hz
        if self.low_frequency_hz >= self.rate_ / 2:
            raise ValueError("Low-frequency bound must lie below Nyquist")
        source_edge = np.median(np.abs(np.diff(x, axis=1)), axis=(0, 1))
        self.zero_threshold_ = np.full(channels, self.zero_threshold, dtype=float) if self.zero_threshold is not None else np.maximum(source_edge * 1e-4, EPS)
        self.flat_threshold_ = np.full(channels, self.flat_threshold, dtype=float) if self.flat_threshold is not None else np.maximum(source_edge * 1e-4, EPS)
        activation = np.sqrt(np.mean(x * x, axis=1))
        self.reference_median_ = np.median(activation, axis=0)
        self.reference_mad_ = np.median(np.abs(activation - self.reference_median_), axis=0)
        self.reference_covariance_ = _covariance(x).mean(axis=0)
        self.reference_neighbor_correlation_ = self._neighbor_correlation(x).mean(axis=0) if self.ring_order is not None else None
        frequency = np.fft.rfftfreq(self.samples_, d=1 / self.rate_)
        if self.line_frequency_hz is None:
            self.line_mask_ = self.neighbor_mask_ = None
        else:
            line = np.abs(frequency - self.line_frequency_hz) <= 1
            neighbor = (((frequency >= self.line_frequency_hz - 6) & (frequency <= self.line_frequency_hz - 3))
                        | ((frequency >= self.line_frequency_hz + 3) & (frequency <= self.line_frequency_hz + 6)))
            self.line_mask_, self.neighbor_mask_ = (line, neighbor) if (self.line_frequency_hz < self.rate_ / 2
                                                                         and line.any() and neighbor.any()) else (None, None)
        self.low_mask_ = frequency <= self.low_frequency_hz
        self.valid_mask_ = (frequency >= 20) & (frequency <= min(450, self.rate_ * .475))
        self.low_available_ = bool(self.pre_highpass_available and self.low_mask_.any() and self.valid_mask_.any())
        self.fitted_ = True
        return self

    def _neighbor_correlation(self, x: np.ndarray) -> np.ndarray:
        centered = x - x.mean(axis=1, keepdims=True)
        dot = np.einsum("ntc,ntd->ncd", centered, centered)
        norm = np.sqrt(np.maximum(np.diagonal(dot, axis1=1, axis2=2), 0))
        corr = dot / (norm[:, :, None] * norm[:, None, :] + EPS)
        order = self.ring_order
        values = np.empty((len(x), x.shape[2]))
        for position, channel in enumerate(order):
            values[:, channel] = (corr[:, channel, order[(position - 1) % len(order)]]
                                  + corr[:, channel, order[(position + 1) % len(order)]]) / 2
        return values

    def transform(self, batch: FeatureBatch) -> np.ndarray:
        self._check()
        x = np.asarray(batch.emg, dtype=np.float64)
        if (x.shape[1:] != (self.samples_, self.channels_) or batch.sample_rate_hz != self.rate_
                or not np.isfinite(x).all()):
            raise ValueError("quality observation sensor/window contract mismatch")
        zero = np.mean(np.abs(x) < self.zero_threshold_[None, None, :], axis=1)
        variance = np.var(x, axis=1)
        flat = _longest_strict_flatline(x, self.flat_threshold_)
        clip = np.zeros_like(zero)
        if self.adc_range is not None:
            low, high = self.adc_range
            tolerance = max((high - low) * 1e-6, EPS)
            clip = np.mean((x <= low + tolerance) | (x >= high - tolerance), axis=1)
        line = np.zeros_like(zero)
        low = np.zeros_like(zero)
        if self.line_mask_ is not None or self.low_available_:
            centered = (x - x.mean(axis=1, keepdims=True)) * np.hanning(self.samples_)[None, :, None]
            power = np.abs(np.fft.rfft(centered, axis=1)) ** 2
            if self.line_mask_ is not None:
                line = power[:, self.line_mask_].mean(axis=1) / (power[:, self.neighbor_mask_].mean(axis=1) + EPS)
            if self.low_available_:
                low = power[:, self.low_mask_].sum(axis=1) / (power[:, self.valid_mask_].sum(axis=1) + EPS)
        activation = np.sqrt(np.mean(x * x, axis=1))
        amplitude_z = (activation - self.reference_median_) / (1.4826 * self.reference_mad_ + EPS)
        neighbor = (self._neighbor_correlation(x) - self.reference_neighbor_correlation_
                    if self.ring_order is not None else np.zeros_like(zero))
        covariance_distance = np.linalg.norm(_covariance(x) - self.reference_covariance_[None], axis=(1, 2))
        availability = np.broadcast_to(np.array((self.adc_range is not None, self.line_mask_ is not None,
                                                   self.low_available_, self.ring_order is not None), dtype=float),
                                       (len(x), 4))
        result = np.concatenate((zero, variance, flat, clip, line, low, amplitude_z, neighbor,
                                 covariance_distance[:, None], availability), axis=1)
        if not np.isfinite(result).all():
            raise ValueError("quality observations must be finite")
        return result.astype(np.float32)

    @property
    def feature_names(self) -> tuple[str, ...]:
        self._check()
        metrics = ("zero_fraction", "variance", "longest_flatline_ratio", "clip_fraction",
                   "line_noise_ratio", "low_frequency_ratio", "amplitude_z", "neighbor_correlation_shift")
        return (tuple(f"F9v3.{metric}.ch{channel + 1}" for metric in metrics for channel in range(self.channels_))
                + ("F9v3.covariance_distance", "F9v3.available.adc", "F9v3.available.line",
                   "F9v3.available.low_frequency", "F9v3.available.ring"))
