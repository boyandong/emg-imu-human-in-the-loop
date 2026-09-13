from __future__ import annotations

from typing import Sequence

import numpy as np


G0_TOKEN_NAMES = (
    "rms", "mav", "standard_deviation", "waveform_length", "iqr", "relative_energy",
)
G5_TOKEN_NAMES = (
    "early_rms", "late_rms", "early_minus_late_rms",
    "log_early_over_late", "activation_slope",
)


def _window(value: np.ndarray) -> np.ndarray:
    x = np.asarray(value, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 3 or x.shape[1] < 1:
        raise ValueError("EMG window must have shape [samples>=3, channels>=1]")
    if not np.isfinite(x).all():
        raise ValueError("EMG window contains NaN or Inf")
    return x


def extract_g0_tokens(window: np.ndarray) -> np.ndarray:
    """Return the frozen traditional feature family as one token per channel."""
    x = _window(window)
    rms = np.sqrt(np.mean(x * x, axis=0))
    mav = np.mean(np.abs(x), axis=0)
    standard_deviation = np.std(x, axis=0)
    waveform_length = np.mean(np.abs(np.diff(x, axis=0)), axis=0)
    iqr = np.percentile(x, 75, axis=0) - np.percentile(x, 25, axis=0)
    energy = rms * rms
    relative_energy = energy / max(float(energy.sum()), 1e-12)
    return np.column_stack((
        rms, mav, standard_deviation, waveform_length, iqr, relative_energy,
    )).astype(np.float32)


def extract_g5_tokens(window: np.ndarray) -> np.ndarray:
    """Return early/late activation dynamics as one token per channel."""
    x = _window(window)
    midpoint = len(x) // 2
    early = np.sqrt(np.mean(x[:midpoint] ** 2, axis=0))
    late = np.sqrt(np.mean(x[midpoint:] ** 2, axis=0))
    difference = early - late
    log_ratio = np.log((early + 1e-12) / (late + 1e-12))
    timeline = np.linspace(-0.5, 0.5, len(x), dtype=np.float64)
    slope = timeline @ x / max(float(timeline @ timeline), 1e-12)
    return np.column_stack((early, late, difference, log_ratio, slope)).astype(np.float32)


def extract_hla_feature_tokens(
    window: np.ndarray,
    groups: Sequence[str] = ("G0", "G5"),
) -> np.ndarray:
    """Build R0 (G0) or R1-core (G0+G5) variable-channel tokens."""
    requested = tuple(groups)
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("feature groups must be nonempty and unique")
    unknown = set(requested) - {"G0", "G5"}
    if unknown:
        raise ValueError(f"unsupported general HLA feature groups: {sorted(unknown)}")
    x = _window(window)
    values = {"G0": extract_g0_tokens(x), "G5": extract_g5_tokens(x)}
    result = np.concatenate([values[group] for group in requested], axis=1)
    if not np.isfinite(result).all():
        raise ValueError("feature extraction produced NaN or Inf")
    return result


def hla_feature_names(groups: Sequence[str] = ("G0", "G5")) -> tuple[str, ...]:
    names = {"G0": G0_TOKEN_NAMES, "G5": G5_TOKEN_NAMES}
    requested = tuple(groups)
    if not requested or len(requested) != len(set(requested)) or set(requested) - set(names):
        raise ValueError("feature groups must be a unique subset of G0 and G5")
    return tuple(f"{group}.{name}" for group in requested for name in names[group])
