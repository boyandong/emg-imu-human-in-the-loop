"""Versioned F2a/F2c/F3c candidates following the goal document's centered F2a.

V2's uncentered arms and saved probabilities retain their original identity.
F2b remains deliberately uncentered, as specified separately in its section.
"""
from __future__ import annotations

import numpy as np

from .core import FeatureBatch, FeatureRegistry
from .families import EPS as MATRIX_EIGEN_FLOOR, _sym_log, _sym_power, _vech
from .new_bank_v1 import EPS, _EightChannelFamily


# The shared matrix-log/inverse-root helpers floor eigenvalues at 1e-10.
# A smaller ridge would make a zero-variance source window map away from its
# own fitted reference, violating the tangent origin invariant.
SPD_RIDGE = max(EPS, MATRIX_EIGEN_FLOOR)


def centered_f2a_matrix(windows: np.ndarray, shrinkage: float) -> np.ndarray:
    """Centered sample covariance, fixed isotropic shrinkage, trace+epsilon."""
    x = np.asarray(windows, dtype=np.float64)
    if x.ndim != 3 or x.shape[1] < 2 or x.shape[2] != 8:
        raise ValueError("F2a requires [windows, >=2 samples, 8 channels]")
    centered = x - x.mean(axis=1, keepdims=True)
    covariance = np.einsum("ntc,ntd->ncd", centered, centered) / (x.shape[1] - 1)
    trace = np.trace(covariance, axis1=1, axis2=2)
    shrunk = ((1 - shrinkage) * covariance
              + (shrinkage * trace / 8)[:, None, None] * np.eye(8))
    return shrunk / (np.trace(shrunk, axis1=1, axis2=2)[:, None, None] + EPS)


class SpecTraceCovarianceV3(_EightChannelFamily):
    """F2a's centered trace-normalized upper triangle, with Frobenius scaling."""
    family_id = "new_v3_spec_trace_covariance"

    def __init__(self, shrinkage: float = .05) -> None:
        super().__init__()
        if not np.isfinite(shrinkage) or not 0 <= shrinkage < 1:
            raise ValueError("shrinkage must be fixed in [0,1)")
        self.shrinkage = float(shrinkage)

    def _fit_metadata(self) -> None:
        self._names = tuple(f"spec_f2a.ch{i+1}.ch{j+1}"
                            for i, j in zip(*np.triu_indices(8)))

    def transform(self, batch: FeatureBatch) -> np.ndarray:
        self._validate(batch)
        matrix = centered_f2a_matrix(batch.emg, self.shrinkage)
        return _vech(matrix).astype(np.float32)


class SpecSpdTangentV3(SpecTraceCovarianceV3):
    """F2c source-only log-Euclidean tangent map of centered F2a matrices."""
    family_id = "new_v3_spec_spd_tangent"

    def _fit_metadata(self) -> None:
        self._names = tuple(f"spec_f2c.ch{i+1}.ch{j+1}"
                            for i, j in zip(*np.triu_indices(8)))

    def fit(self, batch: FeatureBatch, labels: np.ndarray | None = None):
        super().fit(batch)
        matrix = centered_f2a_matrix(batch.emg, self.shrinkage) + SPD_RIDGE * np.eye(8)
        mean_log = np.mean(np.stack([_sym_log(item) for item in matrix]), axis=0)
        values, vectors = np.linalg.eigh((mean_log + mean_log.T) * .5)
        self.reference_ = (vectors * np.exp(values)) @ vectors.T
        return self

    def transform(self, batch: FeatureBatch) -> np.ndarray:
        self._validate(batch)
        inverse_root = _sym_power(self.reference_, -.5)
        matrix = centered_f2a_matrix(batch.emg, self.shrinkage) + SPD_RIDGE * np.eye(8)
        tangent = np.stack([_sym_log(inverse_root @ item @ inverse_root)
                            for item in matrix])
        return _vech(tangent).astype(np.float32)


class SpecRingRelativeCovarianceV3(SpecTraceCovarianceV3):
    """F3c lag summaries of the same centered F2a matrix."""
    family_id = "new_v3_spec_ring_relative_covariance"

    def __init__(self, *, ring_topology: bool = False, shrinkage: float = .05,
                 min_temporal_samples: int = 100) -> None:
        super().__init__(shrinkage)
        if min_temporal_samples < 6:
            raise ValueError("temporal summaries require at least six samples")
        self.ring_topology = bool(ring_topology)
        self.min_temporal_samples = int(min_temporal_samples)

    def fit(self, batch: FeatureBatch, labels: np.ndarray | None = None):
        if not self.ring_topology:
            raise ValueError("physical circular channel order must be asserted")
        return super().fit(batch, labels)

    def _fit_metadata(self) -> None:
        self.with_temporal_ = self.samples_ >= self.min_temporal_samples
        metrics = ("mean", "median", "std", "q25", "q75")
        if self.with_temporal_:
            metrics += ("half_mean_abs_delta",)
        self._names = tuple(f"spec_f3c.lag{lag}.{metric}"
                            for lag in range(1, 5) for metric in metrics)

    @staticmethod
    def _lag(matrix: np.ndarray, lag: int) -> np.ndarray:
        indices = np.arange(8)
        return matrix[:, indices, (indices + lag) % 8]

    def transform(self, batch: FeatureBatch) -> np.ndarray:
        self._validate(batch)
        x = np.asarray(batch.emg, dtype=np.float64)
        matrix = centered_f2a_matrix(x, self.shrinkage)
        if self.with_temporal_:
            half = x.shape[1] // 2
            early = centered_f2a_matrix(x[:, :half], self.shrinkage)
            late = centered_f2a_matrix(x[:, half:], self.shrinkage)
        columns = []
        for lag in range(1, 5):
            values = self._lag(matrix, lag)
            columns.extend((values.mean(axis=1), np.median(values, axis=1),
                            values.std(axis=1), np.quantile(values, .25, axis=1),
                            np.quantile(values, .75, axis=1)))
            if self.with_temporal_:
                columns.append(np.abs(self._lag(late, lag).mean(axis=1)
                                      - self._lag(early, lag).mean(axis=1)))
        return np.stack(columns, axis=1).astype(np.float32)


def spec_spatial_v3_registry() -> FeatureRegistry:
    registry = FeatureRegistry()
    for family in (SpecTraceCovarianceV3, SpecSpdTangentV3,
                   SpecRingRelativeCovarianceV3):
        registry.register(family.family_id, family)
    return registry
