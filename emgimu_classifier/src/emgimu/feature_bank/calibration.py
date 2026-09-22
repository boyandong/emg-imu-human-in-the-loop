from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping

import numpy as np

from .core import FeatureBatch


EPS = 1e-10


@dataclass(slots=True)
class PersonalNormalizer:
    rest_label: int | str = 0
    center_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, batch: FeatureBatch, labels: np.ndarray) -> "PersonalNormalizer":
        y = np.asarray(labels)
        if len(y) != batch.windows:
            raise ValueError("labels must match calibration windows")
        rest = np.asarray(batch.emg, dtype=np.float64)[y == self.rest_label]
        active = np.asarray(batch.emg, dtype=np.float64)[y != self.rest_label]
        if not len(rest) or not len(active):
            raise ValueError("personal normalization requires rest and active calibration")
        self.center_ = np.median(rest.reshape(-1, batch.channels), axis=0)
        magnitude = np.abs(active - self.center_[None, None, :]).reshape(-1, batch.channels)
        self.scale_ = np.maximum(np.quantile(magnitude, 0.95, axis=0), EPS)
        return self

    def transform(self, batch: FeatureBatch) -> FeatureBatch:
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("personal normalizer must be fit first")
        if batch.channels != len(self.center_):
            raise ValueError("channel count differs from calibration")
        emg = (np.asarray(batch.emg, dtype=np.float64) - self.center_[None, None, :]) / self.scale_[None, None, :]
        return FeatureBatch(emg, batch.sample_rate_hz, batch.imu, batch.posture)


class PersonalAnchor:
    """Gesture-relative coordinates fit only from explicitly supplied calibration rows."""

    def __init__(self, *, metric: str = "standardized_euclidean", prototype: str = "mean") -> None:
        if metric not in {"euclidean", "standardized_euclidean", "cosine"}:
            raise ValueError("unsupported anchor metric")
        if prototype not in {"mean", "median"}:
            raise ValueError("unsupported prototype estimator")
        self.metric = metric
        self.prototype = prototype
        self.classes_: np.ndarray | None = None
        self.prototypes_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.similarity_scale_: float | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "PersonalAnchor":
        x, y = np.asarray(features, dtype=np.float64), np.asarray(labels)
        if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or not x.shape[1] or not np.all(np.isfinite(x)):
            raise ValueError("features and labels must be finite aligned matrices")
        if np.issubdtype(y.dtype, np.number) and not np.all(np.isfinite(y)):
            raise ValueError("anchor labels must be finite")
        classes = np.unique(y)
        if len(classes) < 2:
            raise ValueError("anchors require at least two gesture classes")
        estimator = np.mean if self.prototype == "mean" else np.median
        self.classes_ = classes
        self.prototypes_ = np.stack([estimator(x[y == label], axis=0) for label in classes])
        median = np.median(x, axis=0)
        self.scale_ = np.maximum(1.4826 * np.median(np.abs(x - median), axis=0), EPS)
        self.similarity_scale_ = max(float(np.median(self._distances(x))), EPS)
        return self

    def _distances(self, features: np.ndarray) -> np.ndarray:
        if self.prototypes_ is None:
            raise RuntimeError("anchor must be fit from calibration first")
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.prototypes_.shape[1] or not np.all(np.isfinite(x)):
            raise ValueError("anchor feature dimension mismatch")
        if self.metric == "cosine":
            dot = x @ self.prototypes_.T
            norms = np.linalg.norm(x, axis=1, keepdims=True) * np.linalg.norm(self.prototypes_, axis=1)[None, :]
            distances = 1.0 - dot / np.maximum(norms, EPS)
        else:
            delta = x[:, None, :] - self.prototypes_[None, :, :]
            if self.metric == "standardized_euclidean":
                delta = delta / self.scale_[None, None, :]
            distances = np.linalg.norm(delta, axis=2)
        return distances

    def transform(self, features: np.ndarray) -> np.ndarray:
        distances = self._distances(features)
        ordered = np.sort(distances, axis=1)
        margin = ordered[:, 1] - ordered[:, 0]
        normalized_margin = margin / np.maximum(distances.mean(axis=1), EPS)
        similarity_scale = getattr(self, 'similarity_scale_', None)
        if similarity_scale is None:
            # Legacy fitted artifacts contain only calibration prototypes and scales.
            # Derive a fixed scale from those prototypes without changing fitted state.
            similarity_scale = max(float(np.median(self._distances(self.prototypes_))), EPS)
        similarity = np.exp(-distances / similarity_scale)
        return np.nan_to_num(np.column_stack((distances, similarity, margin, normalized_margin))).astype(np.float32)

    @property
    def feature_names(self) -> tuple[str, ...]:
        if self.classes_ is None:
            raise RuntimeError("anchor must be fit first")
        return tuple(f"F7.distance.{label}" for label in self.classes_) + tuple(
            f"F7.similarity.{label}" for label in self.classes_
        ) + ("F7.margin", "F7.normalized_margin")


@dataclass(frozen=True, slots=True)
class ReliabilityWeights:
    classes: tuple[int | str, ...]
    family_ids: tuple[str, ...]
    population: np.ndarray
    n0: float = 8.0
    temperature: float = 1.0

    def personal(self, calibration: Mapping[str, tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        scores = []
        sample_counts = []
        for family in self.family_ids:
            if family not in calibration:
                scores.append(-np.inf)
                sample_counts.append(0)
                continue
            features, labels = calibration[family]
            x, y = np.asarray(features, dtype=np.float64), np.asarray(labels)
            if x.ndim != 2 or len(x) != len(y) or not np.all(np.isfinite(x)):
                raise ValueError('calibration features must be finite aligned matrices')
            prototypes = {label: x[y == label].mean(axis=0) for label in self.classes if np.any(y == label)}
            if len(prototypes) != len(self.classes):
                scores.append(-np.inf)
                sample_counts.append(0)
                continue
            between = np.mean([np.linalg.norm(prototypes[a] - prototypes[b]) for i, a in enumerate(self.classes) for b in self.classes[i + 1:]])
            within = np.mean([np.linalg.norm(x[y == label] - prototypes[label], axis=1).mean() for label in self.classes])
            scores.append(np.log((between + EPS) / (within + EPS)))
            sample_counts.append(len(x))
        score = np.asarray(scores)
        usable = np.isfinite(score)
        personal = np.zeros_like(score)
        if np.any(usable):
            shifted = (score[usable] - np.max(score[usable])) / max(self.temperature, EPS)
            personal[usable] = np.exp(shifted) / np.exp(shifted).sum()
        if not np.any(usable):
            raise ValueError('no family has complete calibration coverage')
        count = min(np.asarray(sample_counts)[usable])
        alpha = self.n0 / (self.n0 + count)
        combined = alpha * np.asarray(self.population, dtype=np.float64) + (1.0 - alpha) * personal
        combined[~usable] = 0.0
        return combined / np.maximum(combined.sum(), EPS)


def late_fusion(
    probabilities: Mapping[str, np.ndarray],
    family_ids: tuple[str, ...],
    weights: np.ndarray,
    quality: Mapping[str, np.ndarray] | None = None,
) -> np.ndarray:
    available = tuple(item for item in family_ids if item in probabilities)
    arrays = [np.asarray(probabilities[item], dtype=np.float64) for item in available]
    if (not arrays or arrays[0].ndim != 2
            or any(array.shape != arrays[0].shape or not np.isfinite(array).all()
                   for array in arrays)):
        raise ValueError("family probabilities must be non-empty and aligned")
    base = np.asarray(weights, dtype=np.float64)
    if base.shape != (len(family_ids),) or not np.all(np.isfinite(base)) or np.any(base < 0) or base.sum() <= 0:
        raise ValueError("fusion weights must be non-negative and match families")
    base = base[[family_ids.index(item) for item in available]]
    if base.sum() <= EPS:
        base = np.ones(len(available), dtype=np.float64)
    per_row = np.broadcast_to(base[None, :], (arrays[0].shape[0], len(base))).copy()
    if quality is not None:
        for index, family in enumerate(available):
            if family in quality:
                values = np.asarray(quality[family], dtype=np.float64)
                if values.shape != (len(per_row),) or not np.isfinite(values).all():
                    raise ValueError("quality must be finite and aligned with family probabilities")
                per_row[:, index] *= np.clip(values, 0.0, 1.0)
    # A fully rejected window still needs a valid probability distribution.
    rejected = per_row.sum(axis=1) <= EPS
    per_row[rejected] = base
    per_row /= np.maximum(per_row.sum(axis=1, keepdims=True), EPS)
    fused = sum(per_row[:, index, None] * array for index, array in enumerate(arrays))
    return (fused / np.maximum(fused.sum(axis=1, keepdims=True), EPS)).astype(np.float64)


@dataclass(frozen=True, slots=True)
class FusionDecision:
    probabilities: np.ndarray
    labels: tuple[str, ...]
    rejected: np.ndarray
    rejection_reason: tuple[str, ...]


def late_fusion_decision(
    probabilities: Mapping[str, np.ndarray],
    family_ids: tuple[str, ...],
    weights: np.ndarray,
    class_labels: tuple[str, ...],
    quality: Mapping[str, np.ndarray] | None = None,
    *,
    minimum_confidence: float = 0.0,
    unknown_label: str = "Unknown",
) -> FusionDecision:
    """Return an explicit Unknown decision while retaining scoreable probabilities.

    A confidence cutoff must be chosen from source/validation data before this
    function is applied to a held-out set. The default only rejects windows
    with no effective provider after quality routing.
    """
    if (not np.isfinite(minimum_confidence)
            or not 0.0 <= minimum_confidence <= 1.0):
        raise ValueError("minimum_confidence must be between zero and one")
    fused = late_fusion(probabilities, family_ids, weights, quality)
    if (len(class_labels) != fused.shape[1] or len(set(class_labels)) != len(class_labels)
            or unknown_label in class_labels):
        raise ValueError("class labels must be unique, aligned and exclude Unknown")
    available = tuple(family for family in family_ids if family in probabilities)
    base = np.asarray(weights, dtype=np.float64)[
        [family_ids.index(family) for family in available]]
    effective = np.broadcast_to(base[None, :], (len(fused), len(base))).copy()
    if quality is not None:
        for index, family in enumerate(available):
            if family in quality:
                effective[:, index] *= np.clip(
                    np.asarray(quality[family], dtype=np.float64), 0.0, 1.0)
    no_provider = base.sum() <= EPS
    no_quality = effective.sum(axis=1) <= EPS
    low_confidence = np.max(fused, axis=1) < minimum_confidence
    reasons = tuple(
        "no_weighted_provider" if no_provider else
        "all_quality_rejected" if no_quality[row] else
        "low_confidence" if low_confidence[row] else ""
        for row in range(len(fused))
    )
    labels = np.asarray(class_labels, dtype=object)[np.argmax(fused, axis=1)]
    rejected = np.asarray([bool(reason) for reason in reasons], dtype=bool)
    labels[rejected] = unknown_label
    return FusionDecision(fused, tuple(map(str, labels)), rejected, reasons)


class SessionSignature:
    def __init__(self) -> None:
        self.classes_: np.ndarray | None = None
        self.long_term_: np.ndarray | None = None

    def fit_long_term(self, features: np.ndarray, labels: np.ndarray) -> "SessionSignature":
        x, y = np.asarray(features, dtype=np.float64), np.asarray(labels)
        if x.ndim != 2 or y.ndim != 1 or not len(x) or not x.shape[1] or len(x) != len(y) or not np.all(np.isfinite(x)):
            raise ValueError("long-term profile requires finite aligned features and labels")
        if np.issubdtype(y.dtype, np.number) and not np.all(np.isfinite(y)):
            raise ValueError("profile labels must be finite")
        self.classes_ = np.unique(y)
        self.long_term_ = np.stack([x[y == label].mean(axis=0) for label in self.classes_])
        return self

    def from_session_calibration(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        if self.long_term_ is None:
            raise RuntimeError("long-term profile must be fit first")
        x, y = np.asarray(features, dtype=np.float64), np.asarray(labels)
        if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or x.shape[1] != self.long_term_.shape[1] or not np.all(np.isfinite(x)):
            raise ValueError("session calibration requires finite aligned profile coordinates")
        if set(y) != set(self.classes_):
            raise ValueError("session calibration must cover every long-term class and no additional classes")
        local = np.stack([x[y == label].mean(axis=0) for label in self.classes_])
        residual = local - self.long_term_
        norms = np.linalg.norm(residual, axis=1)
        cosine = np.sum(local * self.long_term_, axis=1) / np.maximum(
            np.linalg.norm(local, axis=1) * np.linalg.norm(self.long_term_, axis=1), EPS
        )
        geometry = []
        for first in range(len(self.classes_)):
            for second in range(first + 1, len(self.classes_)):
                geometry.append(
                    np.linalg.norm(local[first] - local[second])
                    - np.linalg.norm(self.long_term_[first] - self.long_term_[second])
                )
        return np.nan_to_num(np.concatenate((norms, cosine, np.asarray(geometry)))).astype(np.float32)

    @property
    def feature_names(self) -> tuple[str, ...]:
        if self.classes_ is None:
            raise RuntimeError("long-term profile must be fit first")
        names = [f"F8.residual_norm.{label}" for label in self.classes_]
        names.extend(f"F8.cosine_agreement.{label}" for label in self.classes_)
        names.extend(f"F8.geometry_delta.{a}.{b}" for a, b in combinations(self.classes_, 2))
        return tuple(names)
