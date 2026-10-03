"""Versioned F3b channel-correlation eigenvalue spectrum.

This is a new formula implementation, not a recovery of historical CES code.
The envelope width is fixed in physical time before reading held-out windows.
"""
from __future__ import annotations

import numpy as np

from .core import FeatureBatch, FeatureFamily
EPS = 1e-10


def _edge_envelope(x: np.ndarray, width: int) -> np.ndarray:
    """Moving rectified mean without zero-padding artificial edge transients."""
    rectified = np.abs(np.asarray(x, dtype=np.float64))
    width = min(width, rectified.shape[1])
    left = (width - 1) // 2
    right = width - 1 - left
    padded = np.pad(rectified, ((0, 0), (left, right), (0, 0)), mode="edge")
    cumulative = np.pad(np.cumsum(padded, axis=1), ((0, 0), (1, 0), (0, 0)))
    return (cumulative[:, width:] - cumulative[:, :-width]) / width


class DocumentCesFamilyV3(FeatureFamily):
    family_id = "F3b_document_ces_v3"

    def __init__(self, envelope_ms: float = 25.) -> None:
        self.envelope_ms = float(envelope_ms)

    def fit(self, batch: FeatureBatch, labels=None) -> "DocumentCesFamilyV3":
        if batch.channels < 2 or not np.isfinite(self.envelope_ms) or self.envelope_ms <= 0:
            raise ValueError("CES requires >=2 channels and positive fixed envelope width")
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
            raise ValueError("CES window sensor contract differs from source")
        envelope = _edge_envelope(batch.emg, self.width_)
        centered = envelope - envelope.mean(axis=1, keepdims=True)
        gram = np.einsum("ntc,ntd->ncd", centered, centered)
        scale = np.sqrt(np.maximum(np.diagonal(gram, axis1=1, axis2=2), 0.))
        correlation = gram / np.maximum(scale[:, :, None] * scale[:, None, :], EPS)
        values = np.linalg.eigvalsh(correlation)[:, ::-1]
        values = np.maximum(values, 0.)  # round-off can make a PSD eigenvalue slightly negative
        return (values / (values.sum(axis=1, keepdims=True) + EPS)).astype(np.float32)

    @property
    def feature_names(self) -> tuple[str, ...]:
        self._check()
        return tuple(f"F3b.ces.p{index + 1}" for index in range(self.channels_))
