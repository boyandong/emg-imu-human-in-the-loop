"""Identity-checked inference fusion; no fitting or calibration from evaluation."""
import numpy as np

PROVIDERS = ('TD24', 'pattern', 'SPD', 'log_bands')


def _ids(values, name):
    values = tuple(values)
    if not values or any(not isinstance(v, str) or not v.strip() for v in values):
        raise ValueError(name+' requires explicit nonempty string trial IDs')
    return values


def fuse_disjoint_session_predictions(probabilities, weights, *, source_trials,
        calibration_trials, evaluation_trials, provider_names, provider_classes, classes):
    """Fuse ordered per-trial probabilities with already source/calibration-fitted weights.

    Caller supplies actual predictor class axes and trial identities. No ground-truth
    evaluation labels are accepted. Repeated windows must be aggregated before calling.
    """
    source = set(_ids(source_trials, 'source'))
    calibration = set(_ids(calibration_trials, 'calibration'))
    evaluation = _ids(evaluation_trials, 'evaluation')
    if len(set(evaluation)) != len(evaluation):
        raise ValueError('Evaluation requires one probability row per unique trial')
    if source & calibration or source & set(evaluation) or calibration & set(evaluation):
        raise ValueError('Source/calibration/evaluation trial overlap')
    if tuple(provider_names) != PROVIDERS:
        raise ValueError('Provider axis must follow TD24, pattern, SPD, log_bands')
    classes = tuple(classes)
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError('Explicit unique class axis required')
    axes = tuple(tuple(axis) for axis in provider_classes)
    if axes != (classes,)*len(PROVIDERS):
        raise ValueError('Every predictor must share the exact ordered class axis')
    p = np.asarray(probabilities, dtype=float)
    w = np.asarray(weights, dtype=float)
    if p.shape != (4, len(evaluation), len(classes)):
        raise ValueError('Probability shape must match provider/trial/class identities')
    if (not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1)
            or not np.allclose(p.sum(axis=2), 1., rtol=0., atol=1e-8)):
        raise ValueError('Normalized finite probabilities required')
    if (w.shape != (4,) or not np.isfinite(w).all() or np.any(w < 0)
            or not np.isclose(w.sum(), 1., rtol=0., atol=1e-8)):
        raise ValueError('Normalized finite four-provider weights required')
    return np.tensordot(w, p, axes=(0, 0))
