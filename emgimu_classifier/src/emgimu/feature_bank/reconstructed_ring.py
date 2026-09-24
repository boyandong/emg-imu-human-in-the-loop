"""Separately testable ring blocks reconstructed from the current specification.

These wrappers reuse today's RingGeometryFamily signal path. They do not assert
identity with the unavailable historical RLCS or CES implementations.
"""
from __future__ import annotations

import numpy as np

from .core import FeatureBatch
from .families import RingGeometryFamily


class _RingBlock:
    family_id = "reconstructed_ring_block"

    def __init__(self, envelope_ms: float = 25.0) -> None:
        self.base = RingGeometryFamily(envelope_ms=envelope_ms)
        self.channels_: int | None = None

    def fit(self, batch: FeatureBatch, labels: np.ndarray | None = None):
        self.base.fit(batch, labels)
        self.channels_ = batch.channels
        return self

    def _bounds(self) -> tuple[int, int]:
        if self.channels_ is None:
            raise ValueError("ring block must be fit on source data first")
        lags = self.channels_ // 2
        return self._block_bounds(lags, self.channels_)

    @staticmethod
    def _block_bounds(lags: int, channels: int) -> tuple[int, int]:
        raise NotImplementedError

    def transform(self, batch: FeatureBatch) -> np.ndarray:
        start, end = self._bounds()
        return self.base.transform(batch)[:, start:end]

    def fit_transform(self, batch: FeatureBatch, labels: np.ndarray | None = None) -> np.ndarray:
        return self.fit(batch, labels).transform(batch)

    @property
    def feature_names(self) -> tuple[str, ...]:
        start, end = self._bounds()
        return self.base.feature_names[start:end]


class ReconstructedRlcs(_RingBlock):
    """Circular-lag envelope-correlation mean/std; ring origin invariant."""
    family_id = "RLCS_reconstructed_v1"

    @staticmethod
    def _block_bounds(lags: int, channels: int) -> tuple[int, int]:
        return 0, 2 * lags


class ReconstructedCes(_RingBlock):
    """Sorted normalized envelope-correlation eigenvalues; permutation invariant."""
    family_id = "CES_reconstructed_v1"

    @staticmethod
    def _block_bounds(lags: int, channels: int) -> tuple[int, int]:
        return 2 * lags, 2 * lags + channels
