"""Opt-in F5 window trajectory features with the goal document's raw ratios.

This new formula implementation is separate from the frozen native UniBo G5.
"""
from __future__ import annotations

import numpy as np

from .core import FeatureBatch, FeatureFamily


EPS = 1e-10


def _rms_envelope(x: np.ndarray, width: int) -> np.ndarray:
    values = np.asarray(x, dtype=np.float64) ** 2
    width = min(int(width), values.shape[1])
    left = (width - 1) // 2
    right = width - 1 - left
    padded = np.pad(values, ((0, 0), (left, right), (0, 0)), mode='edge')
    integral = np.pad(np.cumsum(padded, axis=1), ((0, 0), (1, 0), (0, 0)))
    return np.sqrt(np.maximum((integral[:, width:] - integral[:, :-width]) / width, 0.))


class DocumentTemporalFormV3(FeatureFamily):
    family_id = 'F5_document_temporal_v3'

    def __init__(self, envelope_ms: float = 25.) -> None:
        self.envelope_ms = float(envelope_ms)

    def fit(self, batch: FeatureBatch, labels=None) -> 'DocumentTemporalFormV3':
        if not np.isfinite(self.envelope_ms) or self.envelope_ms <= 0:
            raise ValueError('positive source-fixed envelope duration required')
        self.channels_ = batch.channels
        self.samples_ = batch.emg.shape[1]
        self.rate_ = batch.sample_rate_hz
        self.width_ = max(1, round(self.rate_ * self.envelope_ms / 1000.))
        self.fitted_ = True
        return self

    def transform(self, batch: FeatureBatch) -> np.ndarray:
        self._check()
        if (batch.channels != self.channels_ or batch.emg.shape[1] != self.samples_
                or batch.sample_rate_hz != self.rate_):
            raise ValueError('F5 sensor/window contract differs from source')
        x = np.asarray(batch.emg, dtype=np.float64)
        midpoint = self.samples_ // 2
        early = np.sqrt(np.mean(x[:, :midpoint] ** 2, axis=1))
        late = np.sqrt(np.mean(x[:, midpoint:] ** 2, axis=1))
        envelope = _rms_envelope(x, self.width_)
        time = np.arange(self.samples_, dtype=np.float64) / self.rate_
        centered_time = time - time.mean()
        slope = np.einsum('t,ntc->nc', centered_time, envelope) / np.sum(centered_time ** 2)
        peak_time = np.argmax(envelope, axis=1) / self.samples_
        q = envelope / (envelope.sum(axis=1, keepdims=True) + EPS)
        entropy = -np.sum(q * np.log(q + EPS), axis=1)
        spatial = envelope / (envelope.sum(axis=2, keepdims=True) + EPS)
        velocity = np.linalg.norm(np.diff(spatial, axis=1), axis=2).mean(axis=1, keepdims=True)
        output = np.concatenate((early, late, early / (late + EPS), late - early,
                                 slope, peak_time, entropy, velocity), axis=1)
        if not np.isfinite(output).all():
            raise ValueError('F5 temporal features must remain finite')
        return output.astype(np.float32)

    @property
    def feature_names(self) -> tuple[str, ...]:
        self._check()
        metrics = ('early_rms', 'late_rms', 'early_over_late', 'late_minus_early',
                   'envelope_slope_per_second', 'peak_index_over_T', 'temporal_entropy')
        return (tuple(f'F5v3.{metric}.ch{channel + 1}' for metric in metrics
                      for channel in range(self.channels_))
                + ('F5v3.spatial_map_velocity',))
