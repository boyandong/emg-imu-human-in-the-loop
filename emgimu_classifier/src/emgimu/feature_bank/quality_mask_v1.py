"""Source-frozen, availability-aware F9 candidate quality mask.

This is a bounded rule candidate. It does not certify a hardware fault or
silently replace the legacy fusion quality path.
"""
from __future__ import annotations

import numpy as np


class SourceCalibratedQualityMask:
    """Map F9v2 observations to channel scores using source-only thresholds."""

    COMPONENTS = (
        ("zero", "F9.zero_fraction", None, 0.10, 0.50),
        ("flat", "F9v2.longest_flatline_ratio", None, 0.10, 0.50),
        ("clip", "F9.clip_fraction", "adc_clipping", 0.01, 0.10),
        ("amplitude", "F9.amplitude_z", None, 6.0, None),
        ("correlation", "F9v2.correlation_anomaly", None, 6.0, None),
        ("line", "F9.line_noise_ratio", "line_noise", 10.0, None),
        ("low_frequency", "F9v2.low_frequency_power_ratio", "low_frequency_pre_highpass", 10.0, None),
    )

    def __init__(self, *, source_quantile: float = 0.995) -> None:
        if not np.isfinite(source_quantile) or not 0.9 <= source_quantile < 1.0:
            raise ValueError("source quantile must be in [0.9,1)")
        self.source_quantile = float(source_quantile)
        self.names_: tuple[str, ...] | None = None
        self.channels_: int | None = None
        self.indices_: dict[str, np.ndarray] = {}
        self.thresholds_: dict[str, np.ndarray] = {}
        self.available_: dict[str, bool] = {}

    def fit(self, source_features: np.ndarray, feature_names: tuple[str, ...]) -> "SourceCalibratedQualityMask":
        x = np.asarray(source_features, dtype=np.float64)
        names = tuple(feature_names)
        if x.ndim != 2 or not len(x) or x.shape[1] != len(names) or not np.isfinite(x).all():
            raise ValueError("finite aligned source F9v2 features required")
        if len(names) != len(set(names)):
            raise ValueError("F9v2 feature names must be unique")
        channels = sum(name.startswith("F9.zero_fraction.ch") for name in names)
        if channels < 1:
            raise ValueError("F9v2 channel observations required")
        availability = {}
        for key in ("adc_clipping", "line_noise", "low_frequency_pre_highpass"):
            index = names.index(f"F9v2.available.{key}")
            values = np.unique(x[:, index])
            if len(values) != 1 or values[0] not in (0., 1.):
                raise ValueError("source availability must be a fixed binary contract")
            availability[key] = bool(values[0])
        indices, thresholds = {}, {}
        for key, prefix, required, floor, ceiling in self.COMPONENTS:
            indices[key] = np.asarray([names.index(f"{prefix}.ch{channel}")
                                       for channel in range(1, channels + 1)], dtype=int)
            active = required is None or availability[required]
            if active:
                values = np.abs(x[:, indices[key]]) if key == "amplitude" else x[:, indices[key]]
                if np.any(values < 0):
                    raise ValueError(f"negative source quality component: {key}")
                quantile = np.quantile(values, self.source_quantile, axis=0)
                if ceiling is not None and np.any(quantile >= ceiling):
                    raise ValueError(f"source {key} is too degraded to calibrate a mask")
                thresholds[key] = np.maximum(quantile, floor)
        self.names_ = names
        self.channels_ = channels
        self.indices_ = indices
        self.thresholds_ = thresholds
        self.available_ = availability
        return self

    def transform(self, features: np.ndarray, feature_names: tuple[str, ...]) -> np.ndarray:
        if self.names_ is None:
            raise RuntimeError("quality mask must be fit on source observations")
        x = np.asarray(features, dtype=np.float64)
        if (tuple(feature_names) != self.names_ or x.ndim != 2 or x.shape[1] != len(self.names_)
                or not np.isfinite(x).all()):
            raise ValueError("target F9v2 features differ from frozen source contract")
        for key, expected in self.available_.items():
            values = x[:, self.names_.index(f"F9v2.available.{key}")]
            if not np.all(values == float(expected)):
                raise ValueError("target availability differs from frozen source contract")
        quality = np.ones((len(x), self.channels_), dtype=np.float64)
        for key, threshold in self.thresholds_.items():
            value = x[:, self.indices_[key]]
            if key == "amplitude":
                value = np.abs(value)
            severity = np.clip(value / threshold[None, :] - 1.0, 0.0, 1.0)
            quality *= 1.0 - severity
        bad = (quality < 0.5).sum(axis=1)
        return np.column_stack((quality, bad, quality.mean(axis=1),
                                quality.min(axis=1), quality.var(axis=1))).astype(np.float32)

    @property
    def feature_names(self) -> tuple[str, ...]:
        if self.names_ is None:
            raise RuntimeError("quality mask must be fit first")
        return tuple(f"F9cal.quality.ch{channel}" for channel in range(1, self.channels_ + 1)) + (
            "F9cal.bad_channel_count", "F9cal.mean_quality", "F9cal.min_quality",
            "F9cal.quality_variance",
        )
