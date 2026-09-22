"""Personal Anchor in a frozen, source-fitted SPD tangent coordinate system."""
import copy
import hashlib

import numpy as np

from .calibration import PersonalAnchor
from .core import FeatureBatch
from .families import SpdTangentFamily


class SpdTangentPersonalAnchor:
    """Use log-tangent Frobenius distances; source reference never refits.

    The caller must fit ``source_family`` on source/training windows before
    construction. ``fit`` sees only labeled personal calibration windows.
    Evaluation windows enter ``transform`` without labels or state updates.
    This is a tangent-space candidate, not exact affine-invariant geodesics.
    """

    def __init__(self, source_family: SpdTangentFamily) -> None:
        if not isinstance(source_family, SpdTangentFamily) or not source_family.fitted_:
            raise ValueError("A fitted source SPD tangent family is required")
        self.family_ = copy.deepcopy(source_family)
        reference = np.asarray(self.family_.reference_, dtype=np.float64)
        if (reference.ndim != 2 or reference.shape[0] != reference.shape[1]
                or not np.all(np.isfinite(reference)) or np.linalg.eigvalsh(reference).min() <= 0):
            raise ValueError("Source SPD reference must be finite and positive definite")
        self.source_reference_sha256_ = hashlib.sha256(
            np.ascontiguousarray(reference).tobytes()).hexdigest()
        self.anchor_ = PersonalAnchor(metric="euclidean")

    def fit(self, calibration: FeatureBatch, labels) -> "SpdTangentPersonalAnchor":
        candidate = PersonalAnchor(metric="euclidean").fit(self.family_.transform(calibration), labels)
        self.anchor_ = candidate
        return self

    def transform(self, evaluation: FeatureBatch) -> np.ndarray:
        if self.anchor_.classes_ is None:
            raise RuntimeError("Personal SPD anchor requires labeled calibration first")
        return self.anchor_.transform(self.family_.transform(evaluation))

    def _trial_tangents(self, batch: FeatureBatch, trial_ids, labels=None):
        ids = np.asarray(trial_ids)
        if ids.ndim != 1 or len(ids) != batch.windows or not len(ids):
            raise ValueError("trial IDs must match nonempty complete window batches")
        if any(not str(value).strip() for value in ids):
            raise ValueError("trial IDs must be nonempty")
        y = None if labels is None else np.asarray(labels)
        if y is not None and (y.ndim != 1 or len(y) != len(ids)):
            raise ValueError("labels must match calibration windows")
        tangent = self.family_.transform(batch)
        trials = np.unique(ids)
        means, trial_labels = [], []
        for trial in trials:
            selected = ids == trial
            means.append(tangent[selected].mean(axis=0))
            if y is not None:
                values = np.unique(y[selected])
                if len(values) != 1:
                    raise ValueError(f"inconsistent labels within trial {trial}")
                trial_labels.append(values[0])
        return np.stack(means), trials, None if y is None else np.asarray(trial_labels)

    def fit_trials(self, calibration: FeatureBatch, labels, trial_ids) -> "SpdTangentPersonalAnchor":
        """Fit one prototype observation per complete, labeled native trial."""
        features, _, trial_labels = self._trial_tangents(calibration, trial_ids, labels)
        candidate = PersonalAnchor(metric="euclidean").fit(features, trial_labels)
        self.anchor_ = candidate
        return self

    def transform_trials(self, evaluation: FeatureBatch, trial_ids) -> tuple[np.ndarray, np.ndarray]:
        """Return ordered trial IDs and coordinates without updating fitted state."""
        if self.anchor_.classes_ is None:
            raise RuntimeError("Personal SPD anchor requires labeled calibration first")
        features, trials, _ = self._trial_tangents(evaluation, trial_ids)
        return trials, self.anchor_.transform(features)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(name.replace("F7.", "F7.spd_tangent.", 1) for name in self.anchor_.feature_names)
