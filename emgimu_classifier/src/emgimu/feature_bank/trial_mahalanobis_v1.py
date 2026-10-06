"""Independent-trial budget guard for the optional personal Mahalanobis anchor."""
import numpy as np
from .calibration import DocumentPersonalAnchorV2


class TrialMahalanobisAnchorV1:
    def __init__(self, covariance_shrinkage=.2):
        self.shrinkage = covariance_shrinkage

    @staticmethod
    def _ids(ids, size):
        values = np.asarray(ids, dtype=object)
        if values.ndim != 1 or len(values) != size or not size:
            raise ValueError('Aligned nonempty trial IDs required')
        if any(not isinstance(i, str) or not i.strip() for i in values):
            raise ValueError('Explicit nonempty string trial IDs required')
        return values

    def fit(self, features, labels, *, trial_ids):
        x = np.asarray(features, dtype=float); y = np.asarray(labels)
        if x.ndim != 2 or not x.shape[1] or not np.isfinite(x).all() or y.shape != (len(x),):
            raise ValueError('Finite aligned feature rows and labels required')
        ids = self._ids(trial_ids, len(x)); unique = sorted(set(ids))
        means = []; truth = []
        for trial in unique:
            idx = ids == trial; values = np.unique(y[idx])
            if len(values) != 1:
                raise ValueError('A native calibration trial cannot carry mixed classes')
            means.append(x[idx].mean(axis=0)); truth.append(values[0])
        means = np.stack(means); truth = np.asarray(truth)
        counts = {str(c): int(np.sum(truth == c)) for c in np.unique(truth)}
        if any(n < x.shape[1]+2 for n in counts.values()):
            raise ValueError('Mahalanobis requires dimension + 2 independent trials per class')
        candidate = DocumentPersonalAnchorV2(metric='shrinkage_mahalanobis',
            covariance_shrinkage=self.shrinkage).fit(means, truth)
        # Publish fitted state only after all numerical and provenance checks pass.
        self.anchor_ = candidate
        self.calibration_trial_ids_ = frozenset(unique)
        self.trial_counts_ = counts
        self.dimension_ = x.shape[1]
        return self

    def transform(self, features, *, trial_ids):
        if not hasattr(self, 'anchor_'):
            raise RuntimeError('Independent calibration trial fit required')
        x = np.asarray(features, dtype=float)
        ids = self._ids(trial_ids, len(x))
        if self.calibration_trial_ids_.intersection(ids):
            raise ValueError('Evaluation overlaps calibration trials')
        return self.anchor_.transform(x)
