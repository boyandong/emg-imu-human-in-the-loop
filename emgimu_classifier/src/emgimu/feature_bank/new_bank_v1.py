"""New eight-channel EMG feature bank with explicit, independently defined math.

This module does not call historical/reference feature transforms. All families
produce one feature row per window and keep only source-fit input metadata.
"""
from __future__ import annotations

import numpy as np

from .core import FeatureBatch, FeatureFamily, FeatureRegistry


EPS = 1e-12


class _EightChannelFamily(FeatureFamily):
    def __init__(self) -> None:
        self.fitted_ = False
        self.sample_rate_hz_: float | None = None
        self.samples_: int | None = None
        self._names: tuple[str, ...] = ()

    def fit(self, batch: FeatureBatch, labels: np.ndarray | None = None):
        if batch.channels != 8:
            raise ValueError("new bank v1 requires eight ordered EMG channels")
        self.sample_rate_hz_ = float(batch.sample_rate_hz)
        self.samples_ = int(batch.emg.shape[1])
        self._fit_metadata()
        self.fitted_ = True
        return self

    def _fit_metadata(self) -> None:
        raise NotImplementedError

    def _validate(self, batch: FeatureBatch) -> None:
        self._check()
        if batch.channels != 8 or batch.emg.shape[1] != self.samples_ or batch.sample_rate_hz != self.sample_rate_hz_:
            raise ValueError("channel count, window samples or sample rate differ from source fit")

    @property
    def feature_names(self) -> tuple[str, ...]:
        self._check()
        return self._names


class ScalePatternV1(_EightChannelFamily):
    """Channel RMS divided by the root mean square of channel RMS values."""
    family_id = "new_v1_scale_pattern"

    def _fit_metadata(self) -> None:
        self._names = tuple(f"scale_pattern.ch{index + 1}" for index in range(8))

    def transform(self, batch: FeatureBatch) -> np.ndarray:
        self._validate(batch)
        x = np.asarray(batch.emg, dtype=np.float64)
        channel_rms = np.sqrt(np.mean(x * x, axis=1))
        global_rms = np.sqrt(np.mean(channel_rms * channel_rms, axis=1, keepdims=True))
        return (channel_rms / np.maximum(global_rms, EPS)).astype(np.float32)


def _moving_rms(x: np.ndarray, width: int) -> np.ndarray:
    left = (width - 1) // 2
    right = width - 1 - left
    padded = np.pad(np.square(x, dtype=np.float64), ((0, 0), (left, right), (0, 0)), mode="edge")
    cumulative = np.concatenate((np.zeros((x.shape[0], 1, x.shape[2])), np.cumsum(padded, axis=1)), axis=1)
    return np.sqrt(np.maximum((cumulative[:, width:] - cumulative[:, :-width]) / width, 0.0))


def _envelope_correlation(x: np.ndarray, width: int) -> np.ndarray:
    envelope = _moving_rms(x, width)
    centered = envelope - envelope.mean(axis=1, keepdims=True)
    numerator = np.einsum("ntc,ntd->ncd", centered, centered)
    norm = np.sqrt(np.maximum(np.einsum("ntc,ntc->nc", centered, centered), 0.0))
    denominator = norm[:, :, None] * norm[:, None, :]
    correlation = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > EPS)
    return np.clip((correlation + correlation.transpose(0, 2, 1)) / 2.0, -1.0, 1.0)


class _EnvelopeFamily(_EightChannelFamily):
    envelope_ms = 25.0

    def _fit_metadata(self) -> None:
        self.envelope_samples_ = max(1, int(round(self.sample_rate_hz_ * self.envelope_ms / 1000.0)))
        self._set_names()

    def _set_names(self) -> None:
        raise NotImplementedError

    def _correlation(self, batch: FeatureBatch) -> np.ndarray:
        self._validate(batch)
        return _envelope_correlation(np.asarray(batch.emg, dtype=np.float64), self.envelope_samples_)


class RingLagV1(_EnvelopeFamily):
    """Mean/std channel-envelope correlation at each circular electrode lag."""
    family_id = "new_v1_ring_lag"

    def _set_names(self) -> None:
        self._names = tuple(f"ring_lag.lag{lag}.{stat}" for lag in range(1, 5)
                            for stat in ("mean", "std"))

    def transform(self, batch: FeatureBatch) -> np.ndarray:
        correlation = self._correlation(batch)
        pieces = []
        for lag in range(1, 5):
            values = np.stack([correlation[:, channel, (channel + lag) % 8] for channel in range(8)], axis=1)
            pieces.extend((values.mean(axis=1), values.std(axis=1)))
        return np.stack(pieces, axis=1).astype(np.float32)


class CorrelationSpectrumV1(_EnvelopeFamily):
    """Sorted, sum-normalized envelope-correlation eigenvalues."""
    family_id = "new_v1_correlation_spectrum"

    def _set_names(self) -> None:
        self._names = tuple(f"correlation_spectrum.eigen{index + 1}" for index in range(8))

    def transform(self, batch: FeatureBatch) -> np.ndarray:
        eigenvalues = np.maximum(np.linalg.eigvalsh(self._correlation(batch))[:, ::-1], 0.0)
        return (eigenvalues / np.maximum(eigenvalues.sum(axis=1, keepdims=True), EPS)).astype(np.float32)


class FrequencyDirectionV1(_EightChannelFamily):
    """Four bandwise L2-normalized channel-power direction vectors."""
    family_id = "new_v1_frequency_direction"

    def _fit_metadata(self) -> None:
        nyquist = self.sample_rate_hz_ / 2.0
        low = min(20.0, 0.1 * nyquist)
        high = min(450.0, 0.95 * nyquist)
        if high <= low:
            raise ValueError("sample rate cannot support frequency bands")
        edges = np.linspace(low, high, 5)
        frequencies = np.fft.rfftfreq(self.samples_, d=1.0 / self.sample_rate_hz_)
        self.band_indices_ = tuple(np.flatnonzero((frequencies >= a) &
            ((frequencies <= b) if index == 3 else (frequencies < b)))
            for index, (a, b) in enumerate(zip(edges[:-1], edges[1:])))
        if any(len(indices) == 0 for indices in self.band_indices_):
            raise ValueError("window has an empty frequency band")
        self.band_edges_hz_ = tuple(float(value) for value in edges)
        self._names = tuple(f"frequency_direction.band{band + 1}.ch{channel + 1}"
                            for band in range(4) for channel in range(8))

    def transform(self, batch: FeatureBatch) -> np.ndarray:
        self._validate(batch)
        x = np.asarray(batch.emg, dtype=np.float64)
        taper = np.hanning(self.samples_)
        spectrum = np.fft.rfft((x - x.mean(axis=1, keepdims=True)) * taper[None, :, None], axis=1)
        power = np.abs(spectrum) ** 2
        pieces = []
        for indices in self.band_indices_:
            energy = power[:, indices, :].sum(axis=1)
            pieces.append(energy / np.maximum(np.linalg.norm(energy, axis=1, keepdims=True), EPS))
        return np.concatenate(pieces, axis=1).astype(np.float32)


NEW_BANK_V1 = {"scale_pattern": ScalePatternV1, "ring_lag": RingLagV1,
               "correlation_spectrum": CorrelationSpectrumV1,
               "frequency_direction": FrequencyDirectionV1}


def new_bank_v1_registry() -> FeatureRegistry:
    """Opt-in registry; leaves all existing production/default families unchanged."""
    registry = FeatureRegistry()
    for family in NEW_BANK_V1.values():
        registry.register(family.family_id, family)
    return registry
