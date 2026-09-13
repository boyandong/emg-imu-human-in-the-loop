from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .hla_schema import EMGTrial


@dataclass(frozen=True, slots=True)
class WindowProtocol:
    protocol_id: str = "hla_250ms_50ms_v1"
    window_ms: float = 250.0
    hop_ms: float = 50.0

    def __post_init__(self) -> None:
        if not self.protocol_id or self.window_ms <= 0 or self.hop_ms <= 0:
            raise ValueError("window protocol requires an id and positive durations")
        if self.hop_ms > self.window_ms:
            raise ValueError("hop_ms cannot exceed window_ms")

    def sample_counts(self, sample_rate_hz: float) -> tuple[int, int]:
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        window = int(round(self.window_ms * sample_rate_hz / 1000.0))
        hop = int(round(self.hop_ms * sample_rate_hz / 1000.0))
        if window < 3 or hop < 1:
            raise ValueError("protocol produces an unusable sample count")
        return window, hop


HLA_MAIN_PROTOCOL = WindowProtocol()


@dataclass(frozen=True, slots=True)
class TrialWindows:
    protocol_id: str
    trial_id: str
    emg: np.ndarray
    labels: np.ndarray
    start_timestamp_ms: np.ndarray
    start_sample: np.ndarray

    def __len__(self) -> int:
        return len(self.labels)


def window_trial(
    trial: EMGTrial,
    protocol: WindowProtocol = HLA_MAIN_PROTOCOL,
    *,
    label_space: Literal["task", "canonical"] = "task",
) -> TrialWindows:
    """Window one already-assigned trial without crossing unstable/label boundaries."""
    labels = trial.task_label if label_space == "task" else trial.canonical_label
    window_samples, hop_samples = protocol.sample_counts(trial.sample_rate_hz)
    windows: list[np.ndarray] = []
    window_labels: list[int] = []
    starts: list[int] = []
    stop_start = max(0, len(trial.emg) - window_samples + 1)
    for start in range(0, stop_start, hop_samples):
        stop = start + window_samples
        if not np.all(trial.stable_mask[start:stop]):
            continue
        segment_labels = labels[start:stop]
        if np.any(segment_labels != segment_labels[0]):
            continue
        windows.append(trial.emg[start:stop])
        window_labels.append(int(segment_labels[0]))
        starts.append(start)
    channel_count = trial.emg.shape[1]
    emg = (
        np.stack(windows).astype(np.float32)
        if windows else np.empty((0, window_samples, channel_count), dtype=np.float32)
    )
    start_array = np.asarray(starts, dtype=np.int64)
    return TrialWindows(
        protocol.protocol_id, trial.trial_id, emg,
        np.asarray(window_labels, dtype=np.int16),
        trial.timestamp_ms[start_array] if len(start_array) else np.empty(0, dtype=np.float64),
        start_array,
    )
