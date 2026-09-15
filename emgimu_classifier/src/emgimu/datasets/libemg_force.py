from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np

from emgimu.feature_bank import FeatureBatch


FILE_RE = re.compile(
    r"S(?P<subject>\d+)_(?P<condition>(?:\d+P|MVC|Light|Medium|Hard|Ramp))_C(?P<label>\d+)_R(?P<repetition>\d+)\.csv$",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ForceWindows:
    batch: FeatureBatch
    labels: np.ndarray
    subjects: np.ndarray
    conditions: np.ndarray
    trials: np.ndarray
    repetitions: np.ndarray
    sample_weight: np.ndarray

    def take(self, indices: np.ndarray) -> "ForceWindows":
        index = np.asarray(indices)
        return ForceWindows(
            self.batch.take(index), self.labels[index], self.subjects[index], self.conditions[index],
            self.trials[index], self.repetitions[index], self.sample_weight[index],
        )


def _windows(values: np.ndarray, size: int, hop: int, maximum: int | None) -> list[np.ndarray]:
    starts = np.arange(0, max(len(values) - size + 1, 0), hop, dtype=int)
    if maximum is not None and len(starts) > maximum:
        starts = starts[np.linspace(0, len(starts) - 1, maximum).round().astype(int)]
    return [values[start:start + size] for start in starts]


def load_libemg_force_windows(
    root: str | Path,
    *,
    subjects: Iterable[int],
    conditions: Iterable[str],
    sample_rate_hz: float = 1000.0,
    window_ms: float = 200.0,
    hop_ms: float = 200.0,
    maximum_windows_per_trial: int = 8,
) -> ForceWindows:
    root = Path(root)
    subject_set = set(int(item) for item in subjects)
    condition_set = {str(item).lower() for item in conditions}
    size = round(sample_rate_hz * window_ms / 1000.0)
    hop = round(sample_rate_hz * hop_ms / 1000.0)
    rows: dict[str, list] = {key: [] for key in ("emg", "labels", "subjects", "conditions", "trials", "repetitions", "weight")}
    for path in sorted(root.glob("S*/*.csv")):
        match = FILE_RE.match(path.name)
        if match is None:
            raise ValueError(f"unrecognized LibEMG force filename: {path.name}")
        subject = int(match.group("subject"))
        condition = match.group("condition")
        if subject not in subject_set or condition.lower() not in condition_set:
            continue
        values = np.loadtxt(path, delimiter=",", dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 8 or not np.all(np.isfinite(values)):
            raise ValueError(f"invalid LibEMG force signal: {path}")
        windows = _windows(values, size, hop, maximum_windows_per_trial)
        if not windows:
            raise ValueError(f"trial shorter than one window: {path}")
        trial_id = path.stem
        for window in windows:
            rows["emg"].append(window)
            rows["labels"].append(int(match.group("label")) - 1)
            rows["subjects"].append(subject)
            rows["conditions"].append(condition)
            rows["trials"].append(trial_id)
            rows["repetitions"].append(int(match.group("repetition")))
            rows["weight"].append(1.0 / len(windows))
    if not rows["emg"]:
        raise ValueError("no LibEMG force windows matched the requested split")
    weights = np.asarray(rows["weight"], dtype=np.float64)
    weights *= len(weights) / weights.sum()
    return ForceWindows(
        FeatureBatch(np.stack(rows["emg"]), sample_rate_hz),
        np.asarray(rows["labels"], dtype=np.int64),
        np.asarray(rows["subjects"], dtype=np.int64),
        np.asarray(rows["conditions"]),
        np.asarray(rows["trials"]),
        np.asarray(rows["repetitions"], dtype=np.int64),
        weights,
    )
