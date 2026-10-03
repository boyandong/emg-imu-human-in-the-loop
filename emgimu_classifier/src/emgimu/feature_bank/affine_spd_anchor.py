"""Trial-balanced F7 prototypes with the document's affine-invariant SPD metric.

Input matrices are source-fixed F2a covariances (with an SPD ridge), not
source- or target-evaluation-fitted tangent coordinates. This independent
candidate leaves the previously evaluated tangent anchor unchanged.
"""
from __future__ import annotations

import numpy as np
from .spec_spatial_v3 import SPD_RIDGE, centered_f2a_matrix


def document_spd_matrices(windows: np.ndarray, *, shrinkage: float = .05) -> np.ndarray:
    """Make document F2a matrices strictly SPD for affine distances."""
    if not np.isfinite(shrinkage) or not 0 <= shrinkage < 1:
        raise ValueError("source-fixed shrinkage must be in [0, 1)")
    return centered_f2a_matrix(windows, shrinkage) + SPD_RIDGE * np.eye(8)


def affine_spd_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Exact affine-invariant log-eigenvalue distance for two SPD matrices."""
    a, b = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or b.shape != a.shape:
        raise ValueError("matching square SPD matrices required")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("finite SPD matrices required")
    values, vectors = np.linalg.eigh((a + a.T) / 2)
    if values.min() <= 0 or np.linalg.eigvalsh((b + b.T) / 2).min() <= 0:
        raise ValueError("positive definite matrices required")
    inverse_root = (vectors * values**-.5) @ vectors.T
    relative = inverse_root @ b @ inverse_root
    eigenvalues = np.linalg.eigvalsh((relative + relative.T) / 2)
    if eigenvalues.min() <= 0:
        raise ValueError("numerically nonpositive relative covariance")
    return float(np.linalg.norm(np.log(eigenvalues)))


class AffineSpdPrototypeAnchor:
    """One covariance per trial, equal trial mass, calibration-only prototypes.

    The class prototype is the arithmetic mean of its trial matrices, matching
    the document's default mean prototype. Distances use the exact stated
    affine-invariant log-matrix formula. This is not a Karcher-mean prototype.
    """

    def __init__(self) -> None:
        self.classes_ = None

    @staticmethod
    def _validate(matrices, trial_ids, labels=None):
        x = np.asarray(matrices, dtype=np.float64)
        ids = np.asarray(trial_ids)
        if x.ndim != 3 or x.shape[0] < 1 or x.shape[1] != x.shape[2]:
            raise ValueError("nonempty [windows, channels, channels] matrices required")
        if ids.ndim != 1 or len(ids) != len(x) or any(not str(i).strip() for i in ids):
            raise ValueError("nonempty trial IDs must align with matrices")
        if not np.isfinite(x).all() or not np.allclose(x, x.transpose(0, 2, 1), atol=1e-10):
            raise ValueError("finite symmetric SPD matrices required")
        if np.min(np.linalg.eigvalsh(x)) <= 0:
            raise ValueError("positive definite matrices required")
        y = None if labels is None else np.asarray(labels)
        if y is not None and (y.ndim != 1 or len(y) != len(x)):
            raise ValueError("calibration labels must align with matrices")
        return x, ids, y

    @staticmethod
    def _trials(x, ids, labels=None):
        trials = np.unique(ids)
        matrices, trial_labels = [], []
        for trial in trials:
            mask = ids == trial
            matrices.append(x[mask].mean(axis=0))
            if labels is not None:
                found = np.unique(labels[mask])
                if len(found) != 1:
                    raise ValueError(f"mixed calibration labels in trial {trial}")
                trial_labels.append(found[0])
        return trials, np.stack(matrices), None if labels is None else np.asarray(trial_labels)

    def fit(self, matrices, labels, trial_ids) -> "AffineSpdPrototypeAnchor":
        x, ids, y = self._validate(matrices, trial_ids, labels)
        if y is None:
            raise ValueError("labeled calibration matrices required")
        trials, trial_matrices, trial_labels = self._trials(x, ids, y)
        classes = np.unique(trial_labels)
        if len(classes) < 2:
            raise ValueError("at least two calibrated classes required")
        prototypes = np.stack([trial_matrices[trial_labels == label].mean(axis=0)
                               for label in classes])
        self.classes_ = classes
        self.prototypes_ = prototypes
        self.calibration_trial_ids_ = frozenset(trials.tolist())
        self.channels_ = x.shape[1]
        return self

    def transform(self, matrices, trial_ids) -> tuple[np.ndarray, np.ndarray]:
        if self.classes_ is None:
            raise RuntimeError("calibration prototypes must be fitted first")
        x, ids, _ = self._validate(matrices, trial_ids)
        if x.shape[1] != self.channels_:
            raise ValueError("SPD channel count differs from calibration")
        if self.calibration_trial_ids_.intersection(ids.tolist()):
            raise ValueError("calibration trials cannot enter evaluation")
        trials, trial_matrices, _ = self._trials(x, ids)
        distances = np.asarray([[affine_spd_distance(matrix, prototype)
                                 for prototype in self.prototypes_]
                                for matrix in trial_matrices], dtype=np.float64)
        return trials, distances

    @property
    def feature_names(self) -> tuple[str, ...]:
        if self.classes_ is None:
            raise RuntimeError("calibration prototypes must be fitted first")
        return tuple(f"F7.affine_spd.distance.{label}" for label in self.classes_)
