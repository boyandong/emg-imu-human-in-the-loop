"""New source-fitted F0/F2a/F3c blocks for the independent eight-channel bank.

These do not call the historical or reference feature implementations. The
unchanged v1 families can be registered beside them for staged experiments.
"""
from __future__ import annotations

import numpy as np

from .core import FeatureBatch, FeatureRegistry
from .families import _sym_log, _sym_power, _vech
from .new_bank_v1 import EPS, NEW_BANK_V1, _EightChannelFamily


def _normalized_covariance(windows: np.ndarray, shrinkage: float) -> np.ndarray:
    x = np.asarray(windows, dtype=np.float64)
    centered = x - x.mean(axis=1, keepdims=True)
    covariance = np.einsum("ntc,ntd->ncd", centered, centered) / (x.shape[1] - 1)
    trace = np.trace(covariance, axis1=1, axis2=2)
    covariance = ((1.0 - shrinkage) * covariance
                  + (shrinkage * trace / x.shape[2])[:, None, None] * np.eye(x.shape[2]))
    return covariance / np.maximum(np.trace(covariance, axis1=1, axis2=2)[:, None, None], EPS)


class RestNoiseDetailV2(_EightChannelFamily):
    """Six local metrics with thresholds fitted only from source Rest windows."""
    family_id = "new_v2_rest_noise_detail"

    def __init__(self, rest_label: int | str = 0) -> None:
        super().__init__()
        self.rest_label = rest_label

    def _fit_metadata(self) -> None:
        self._names = tuple(f"rest_detail.{metric}.ch{channel + 1}"
                            for metric in ("rms", "mav", "wl", "zc", "ssc", "wamp")
                            for channel in range(8))

    def fit(self, batch: FeatureBatch, labels: np.ndarray | None = None):
        values = np.asarray(labels) if labels is not None else None
        if values is None or values.shape != (batch.windows,):
            raise ValueError("aligned source labels are required for Rest-noise fitting")
        rest = values == self.rest_label
        if not rest.any():
            raise ValueError("source Rest windows are required for noise thresholds")
        super().fit(batch)
        differences = np.abs(np.diff(np.asarray(batch.emg)[rest].astype(np.float64), axis=1))
        flattened = differences.reshape(-1, 8)
        median = np.median(flattened, axis=0)
        mad = np.median(np.abs(flattened - median), axis=0)
        self.thresholds_ = np.maximum(median + 3.0 * 1.4826 * mad, EPS)
        self.rest_windows_ = int(rest.sum())
        return self

    def transform(self, batch: FeatureBatch) -> np.ndarray:
        self._validate(batch)
        x = np.asarray(batch.emg, dtype=np.float64)
        difference = np.diff(x, axis=1)
        threshold = self.thresholds_[None, None, :]
        rms = np.sqrt(np.mean(x * x, axis=1))
        mav = np.mean(np.abs(x), axis=1)
        wl = np.sum(np.abs(difference), axis=1)
        zc = np.sum((x[:, :-1] * x[:, 1:] < 0) & (np.abs(difference) > threshold), axis=1)
        left, right = x[:, 1:-1] - x[:, :-2], x[:, 1:-1] - x[:, 2:]
        ssc = np.sum((left * right > 0) &
                     (np.maximum(np.abs(left), np.abs(right)) > threshold), axis=1)
        wamp = np.sum(np.abs(difference) > threshold, axis=1)
        return np.concatenate((rms, mav, wl, zc, ssc, wamp), axis=1).astype(np.float32)


class TraceCovarianceV2(_EightChannelFamily):
    """Centered, fixed-shrinkage, trace-normalized covariance upper triangle."""
    family_id = "new_v2_trace_covariance"

    def __init__(self, shrinkage: float = 0.05) -> None:
        super().__init__()
        if not np.isfinite(shrinkage) or not 0 <= shrinkage < 1:
            raise ValueError("shrinkage must be fixed in [0,1)")
        self.shrinkage = float(shrinkage)

    def _fit_metadata(self) -> None:
        self._names = tuple(f"trace_cov.ch{i + 1}.ch{j + 1}" for i, j in zip(*np.triu_indices(8)))

    def transform(self, batch: FeatureBatch) -> np.ndarray:
        self._validate(batch)
        covariance = _normalized_covariance(batch.emg, self.shrinkage)
        i, j = np.triu_indices(8)
        vector = covariance[:, i, j].copy()
        vector[:, i != j] *= np.sqrt(2.0)
        return vector.astype(np.float32)


class DocumentTraceCovarianceV2(_EightChannelFamily):
    """Document F2a: uncentered XX^T, trace normalization and fixed shrinkage.

    This is a distinct opt-in view from the already screened centered F2a.
    It retains channel DC/offset energy, so neither is silently substituted
    for the other in saved experiments.
    """
    family_id = "new_v2_document_trace_covariance"

    def __init__(self, shrinkage: float = 0.05) -> None:
        super().__init__()
        if not np.isfinite(shrinkage) or not 0 <= shrinkage < 1:
            raise ValueError("shrinkage must be fixed in [0,1)")
        self.shrinkage = float(shrinkage)

    def _fit_metadata(self) -> None:
        self._names = tuple(f"document_trace.ch{i + 1}.ch{j + 1}"
                            for i, j in zip(*np.triu_indices(8)))

    def transform(self, batch: FeatureBatch) -> np.ndarray:
        self._validate(batch)
        x = np.asarray(batch.emg, dtype=np.float64)
        second_moment = np.einsum("ntc,ntd->ncd", x, x) / x.shape[1]
        trace = np.trace(second_moment, axis1=1, axis2=2)
        shrunk = ((1.0 - self.shrinkage) * second_moment
                  + (self.shrinkage * trace / 8.0)[:, None, None] * np.eye(8))
        normalized = shrunk / np.maximum(np.trace(shrunk, axis1=1, axis2=2)[:, None, None], EPS)
        i, j = np.triu_indices(8)
        vector = normalized[:, i, j].copy()
        vector[:, i != j] *= np.sqrt(2.0)
        return vector.astype(np.float32)


def _document_spd_moment(windows: np.ndarray, shrinkage: float) -> np.ndarray:
    """F2a's uncentered matrix with an absolute ridge for the SPD log map."""
    x = np.asarray(windows, dtype=np.float64)
    moment = np.einsum("ntc,ntd->ncd", x, x) / x.shape[1]
    trace = np.trace(moment, axis1=1, axis2=2)
    shrunk = ((1.0 - shrinkage) * moment
              + (shrinkage * trace / 8.0)[:, None, None] * np.eye(8))
    shrunk += EPS * np.eye(8)[None, :, :]
    return shrunk / np.trace(shrunk, axis1=1, axis2=2)[:, None, None]


class DocumentSpdTangentV2(_EightChannelFamily):
    """F2c tangent map using the document-exact uncentered F2a matrix."""
    family_id = "new_v2_document_spd_tangent"

    def __init__(self, shrinkage: float = .05) -> None:
        super().__init__()
        if not np.isfinite(shrinkage) or not 0 <= shrinkage < 1:
            raise ValueError("shrinkage must be fixed in [0,1)")
        self.shrinkage = float(shrinkage)

    def _fit_metadata(self) -> None:
        self._names = tuple(f"document_spd.ch{i + 1}.ch{j + 1}"
                            for i, j in zip(*np.triu_indices(8)))

    def fit(self, batch: FeatureBatch, labels: np.ndarray | None = None):
        super().fit(batch)
        matrices = _document_spd_moment(batch.emg, self.shrinkage)
        mean_log = np.mean(np.stack([_sym_log(item) for item in matrices]), axis=0)
        values, vectors = np.linalg.eigh((mean_log + mean_log.T) * .5)
        self.reference_ = (vectors * np.exp(values)) @ vectors.T
        return self

    def transform(self, batch: FeatureBatch) -> np.ndarray:
        self._validate(batch)
        inverse_root = _sym_power(self.reference_, -.5)
        matrices = _document_spd_moment(batch.emg, self.shrinkage)
        tangent = np.stack([_sym_log(inverse_root @ item @ inverse_root) for item in matrices])
        return _vech(tangent).astype(np.float32)


class RingRelativeCovarianceV2(_EightChannelFamily):
    """Five robust ring-lag covariance summaries plus optional half-window drift."""
    family_id = "new_v2_ring_relative_covariance"

    def __init__(self, shrinkage: float = 0.05, min_temporal_samples: int = 100) -> None:
        super().__init__()
        if not np.isfinite(shrinkage) or not 0 <= shrinkage < 1:
            raise ValueError("shrinkage must be fixed in [0,1)")
        if min_temporal_samples < 6:
            raise ValueError("temporal summaries require at least six samples")
        self.shrinkage = float(shrinkage)
        self.min_temporal_samples = int(min_temporal_samples)

    def _fit_metadata(self) -> None:
        self.with_temporal_ = self.samples_ >= self.min_temporal_samples
        metrics = ("mean", "median", "std", "q25", "q75") + (("half_mean_abs_delta",)
                   if self.with_temporal_ else ())
        self._names = tuple(f"ring_cov.lag{lag}.{metric}"
                            for lag in range(1, 5) for metric in metrics)

    @staticmethod
    def _lag_values(covariance: np.ndarray, lag: int) -> np.ndarray:
        indices = np.arange(8)
        return covariance[:, indices, (indices + lag) % 8]

    def transform(self, batch: FeatureBatch) -> np.ndarray:
        self._validate(batch)
        x = np.asarray(batch.emg, dtype=np.float64)
        covariance = _normalized_covariance(x, self.shrinkage)
        if self.with_temporal_:
            half = x.shape[1] // 2
            early = _normalized_covariance(x[:, :half], self.shrinkage)
            late = _normalized_covariance(x[:, half:], self.shrinkage)
        columns = []
        for lag in range(1, 5):
            values = self._lag_values(covariance, lag)
            columns.extend((values.mean(axis=1), np.median(values, axis=1),
                            values.std(axis=1), np.quantile(values, 0.25, axis=1),
                            np.quantile(values, 0.75, axis=1)))
            if self.with_temporal_:
                columns.append(np.abs(self._lag_values(late, lag).mean(axis=1)
                                      - self._lag_values(early, lag).mean(axis=1)))
        return np.stack(columns, axis=1).astype(np.float32)


class DocumentRingRelativeCovarianceV2(RingRelativeCovarianceV2):
    """F3c ring summaries of the document-exact uncentered F2a matrix."""
    family_id = "new_v2_document_ring_relative_covariance"

    def __init__(self, *, ring_topology: bool = False, shrinkage: float = .05,
                 min_temporal_samples: int = 100) -> None:
        super().__init__(shrinkage=shrinkage, min_temporal_samples=min_temporal_samples)
        self.ring_topology = bool(ring_topology)

    def fit(self, batch: FeatureBatch, labels: np.ndarray | None = None):
        if not self.ring_topology:
            raise ValueError("verified circular electrode topology is required")
        return super().fit(batch, labels)

    def transform(self, batch: FeatureBatch) -> np.ndarray:
        self._validate(batch)
        x = np.asarray(batch.emg, dtype=np.float64)
        matrix = _document_spd_moment(x, self.shrinkage)
        if self.with_temporal_:
            half = x.shape[1] // 2
            early = _document_spd_moment(x[:, :half], self.shrinkage)
            late = _document_spd_moment(x[:, half:], self.shrinkage)
        columns = []
        for lag in range(1, 5):
            values = self._lag_values(matrix, lag)
            columns.extend((values.mean(axis=1), np.median(values, axis=1),
                            values.std(axis=1), np.quantile(values, .25, axis=1),
                            np.quantile(values, .75, axis=1)))
            if self.with_temporal_:
                columns.append(np.abs(self._lag_values(late, lag).mean(axis=1)
                                      - self._lag_values(early, lag).mean(axis=1)))
        return np.stack(columns, axis=1).astype(np.float32)


NEW_BANK_V2_EXTENSION = {
    "rest_noise_detail": RestNoiseDetailV2,
    "trace_covariance": TraceCovarianceV2,
    "document_trace_covariance": DocumentTraceCovarianceV2,
    "document_spd_tangent": DocumentSpdTangentV2,
    "ring_relative_covariance": RingRelativeCovarianceV2,
    "document_ring_relative_covariance": DocumentRingRelativeCovarianceV2,
}


def new_bank_v2_registry() -> FeatureRegistry:
    registry = FeatureRegistry()
    for family in (*NEW_BANK_V1.values(), *NEW_BANK_V2_EXTENSION.values()):
        registry.register(family.family_id, family)
    return registry
