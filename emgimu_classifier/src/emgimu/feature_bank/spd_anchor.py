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

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(name.replace("F7.", "F7.spd_tangent.", 1) for name in self.anchor_.feature_names)
