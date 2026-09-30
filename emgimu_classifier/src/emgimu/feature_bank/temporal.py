from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from .core import FeatureBatch, FeatureFamily


EPS = 1e-10


@dataclass(frozen=True, slots=True)
class CompleteSequenceBatch(FeatureBatch):
    """Full contiguous sequences, with physical durations independent of bin rate.

    The producer must establish full coverage from native boundaries; neither
    shape nor the nominal rate of a compressed path establishes completeness.
    """
    durations_seconds: np.ndarray | None = None
    full_coverage: bool = False

    def __post_init__(self):
        FeatureBatch.__post_init__(self)
        durations = np.asarray(self.durations_seconds, dtype=float)
        if not self.full_coverage or durations.shape != (self.windows,):
            raise ValueError('Complete sequence coverage and per-sequence native durations are required')
        if not np.isfinite(durations).all() or np.any(durations < 1.):
            raise ValueError('Complete temporal sequences must have native duration >=1 second')

    def take(self, indices):
        index = np.asarray(indices)
        return CompleteSequenceBatch(np.asarray(self.emg)[index], self.sample_rate_hz,
            None if self.imu is None else np.asarray(self.imu)[index],
            None if self.posture is None else np.asarray(self.posture)[index],
            np.asarray(self.durations_seconds)[index], self.full_coverage)


class CuedSequenceAssembler:
    """Build a complete native bout only from explicit, contiguous cue boundaries.

    This does not detect biological onset/offset. A caller must attest that the
    start/end markers cover the whole gesture; missing samples invalidate it.
    """

    def __init__(self, sample_rate_hz: float, channels: int, *, max_duration_s: float = 30.) -> None:
        if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
            raise ValueError("finite positive sample rate required")
        if isinstance(channels, bool) or not isinstance(channels, int) or channels < 1:
            raise ValueError("positive integer channel count required")
        if not np.isfinite(max_duration_s) or max_duration_s < 1.:
            raise ValueError("maximum native duration must be at least one second")
        self.sample_rate_hz = float(sample_rate_hz)
        self.channels = channels
        self.max_samples = int(np.floor(self.sample_rate_hz * max_duration_s))
        self.reset()

    def reset(self) -> None:
        self._trial_id: str | None = None
        self._next_index: int | None = None
        self._parts: list[np.ndarray] = []
        self._samples = 0

    def begin(self, trial_id: str, first_sample_index: int) -> None:
        if self._trial_id is not None:
            raise RuntimeError("finish or reset the current bout before beginning another")
        if not isinstance(trial_id, str) or not trial_id.strip():
            raise ValueError("explicit nonempty trial identity required")
        if isinstance(first_sample_index, bool) or not isinstance(first_sample_index, int) or first_sample_index < 0:
            raise ValueError("nonnegative native first-sample index required")
        self._trial_id = trial_id
        self._next_index = first_sample_index

    def append(self, samples: np.ndarray, first_sample_index: int) -> None:
        if self._trial_id is None:
            raise RuntimeError("explicit bout start required")
        values = np.asarray(samples, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.channels or not len(values) or not np.isfinite(values).all():
            self.reset()
            raise ValueError("finite nonempty native samples with the fitted channel count required")
        if (isinstance(first_sample_index, bool) or not isinstance(first_sample_index, int)
                or first_sample_index != self._next_index):
            self.reset()
            raise ValueError("native sample gap, overlap or reordering invalidates complete coverage")
        if self._samples + len(values) > self.max_samples:
            self.reset()
            raise ValueError("native bout exceeds the source-fixed maximum duration")
        self._parts.append(values.copy())
        self._samples += len(values)
        self._next_index += len(values)

    def finish(self, trial_id: str, end_sample_index: int) -> CompleteSequenceBatch:
        if self._trial_id is None:
            raise RuntimeError("explicit bout start required")
        if (trial_id != self._trial_id or isinstance(end_sample_index, bool)
                or not isinstance(end_sample_index, int) or end_sample_index != self._next_index):
            self.reset()
            raise ValueError("explicit end marker must match the trial and final native sample index")
        duration = self._samples / self.sample_rate_hz
        if duration < 1.:
            self.reset()
            raise ValueError("complete native bout must last at least one second")
        values = np.concatenate(self._parts, axis=0)
        self.reset()
        return CompleteSequenceBatch(values[None], self.sample_rate_hz,
                                     durations_seconds=np.array([duration]), full_coverage=True)


def require_complete_sequences(batch):
    if not isinstance(batch, CompleteSequenceBatch):
        raise ValueError('Temporal path features require explicit complete sequences; short or sparse windows are ineligible')
    # Recheck mutable arrays at the use boundary, not only at construction.
    batch.__post_init__()


def _normalize_path(window: np.ndarray) -> np.ndarray:
    envelope = np.abs(np.asarray(window, dtype=np.float64))
    return envelope / np.maximum(np.linalg.norm(envelope, axis=1, keepdims=True), EPS)


def dtw_distance(first: np.ndarray, second: np.ndarray, band: int) -> float:
    a, b = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("DTW paths must be [time,features] with matching features")
    band = max(int(band), abs(len(a) - len(b)))
    cost = np.full((len(a) + 1, len(b) + 1), np.inf)
    length = np.zeros_like(cost, dtype=np.int32)
    cost[0, 0] = 0.0
    for i in range(1, len(a) + 1):
        for j in range(max(1, i - band), min(len(b), i + band) + 1):
            options = ((cost[i - 1, j], length[i - 1, j]), (cost[i, j - 1], length[i, j - 1]), (cost[i - 1, j - 1], length[i - 1, j - 1]))
            previous_cost, previous_length = min(options, key=lambda item: item[0])
            cost[i, j] = previous_cost + np.linalg.norm(a[i - 1] - b[j - 1])
            length[i, j] = previous_length + 1
    return float(cost[-1, -1] / max(length[-1, -1], 1))


class TemporalTemplateFamily(FeatureFamily):
    family_id = "F5b_dtw_templates"

    def __init__(self, band_fraction: float = 0.1) -> None:
        self.band_fraction = float(band_fraction)
        self.classes_: np.ndarray | None = None
        self.templates_: list[np.ndarray] = []

    def fit(self, batch: FeatureBatch, labels: np.ndarray | None = None) -> "TemporalTemplateFamily":
        require_complete_sequences(batch)
        if not np.isfinite(self.band_fraction) or not 0 < self.band_fraction <= 1:
            raise ValueError('DTW band fraction must be in (0,1]')
        if labels is None:
            raise ValueError("DTW templates require calibration labels")
        y = np.asarray(labels)
        if len(y) != batch.windows:
            raise ValueError("labels must match windows")
        self.classes_ = np.unique(y)
        paths = [_normalize_path(window) for window in batch.emg]
        band = max(1, round(batch.emg.shape[1] * self.band_fraction))
        templates = []
        for label in self.classes_:
            indices = np.flatnonzero(y == label)
            distances = np.zeros((len(indices), len(indices)))
            for row in range(len(indices)):
                for column in range(row + 1, len(indices)):
                    value = dtw_distance(paths[indices[row]], paths[indices[column]], band)
                    distances[row, column] = distances[column, row] = value
            templates.append(paths[indices[int(np.argmin(distances.sum(axis=1)))]] )
        self.templates_ = templates
        self.fitted_ = True
        return self

    def transform(self, batch: FeatureBatch) -> np.ndarray:
        self._check()
        require_complete_sequences(batch)
        band = max(1, round(batch.emg.shape[1] * self.band_fraction))
        return np.asarray([[dtw_distance(_normalize_path(window), template, band) for template in self.templates_] for window in batch.emg], dtype=np.float32)

    @property
    def feature_names(self) -> tuple[str, ...]:
        self._check()
        return tuple(f"F5b.dtw.{label}" for label in self.classes_)


class PathSignatureFamily(FeatureFamily):
    family_id = "F5c_path_signature_order2"

    def __init__(self) -> None:
        self.channels_: int | None = None

    def fit(self, batch: FeatureBatch, labels: np.ndarray | None = None) -> "PathSignatureFamily":
        require_complete_sequences(batch)
        self.channels_ = batch.channels
        self.fitted_ = True
        return self

    def transform(self, batch: FeatureBatch) -> np.ndarray:
        self._check()
        require_complete_sequences(batch)
        if batch.channels != self.channels_:
            raise ValueError("channel count differs from fitted signature")
        paths = np.stack([_normalize_path(window) for window in batch.emg])
        increments = np.diff(paths - paths[:, :1], axis=1)
        level_one = increments.sum(axis=1)
        prefix = np.cumsum(increments, axis=1) - increments
        level_two = np.einsum("ntc,ntd->ncd", prefix, increments) + 0.5 * np.einsum("ntc,ntd->ncd", increments, increments)
        return np.nan_to_num(np.concatenate((level_one, level_two.reshape(len(paths), -1)), axis=1)).astype(np.float32)

    @property
    def feature_names(self) -> tuple[str, ...]:
        self._check()
        first = [f"F5c.signature1.ch{channel + 1}" for channel in range(self.channels_)]
        second = [f"F5c.signature2.ch{a + 1}.ch{b + 1}" for a in range(self.channels_) for b in range(self.channels_)]
        return tuple(first + second)
