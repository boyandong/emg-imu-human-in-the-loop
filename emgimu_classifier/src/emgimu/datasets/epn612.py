from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable
import zipfile

import numpy as np

from emgimu.feature_bank import FeatureBatch


GESTURES = ("noGesture", "fist", "waveIn", "waveOut", "open", "pinch")
GESTURE_TO_LABEL = {name: index for index, name in enumerate(GESTURES)}


@dataclass(frozen=True, slots=True)
class EpnWindows:
    batch: FeatureBatch
    labels: np.ndarray
    users: np.ndarray
    cohort: np.ndarray
    trials: np.ndarray
    sample_weight: np.ndarray

    def take(self, indices: np.ndarray) -> "EpnWindows":
        index = np.asarray(indices)
        return EpnWindows(
            self.batch.take(index), self.labels[index], self.users[index], self.cohort[index],
            self.trials[index], self.sample_weight[index],
        )


def _matrix(mapping: dict, keys: tuple[str, ...]) -> np.ndarray:
    return np.column_stack([np.asarray(mapping[key], dtype=np.float32) for key in keys])


def load_epn612_windows(
    archive: str | Path,
    *,
    users: Iterable[int],
    cohort: str = "trainingJSON",
    source_split: str = "trainingSamples",
    window_ms: float = 200.0,
    windows_per_trial: int = 4,
) -> EpnWindows:
    if cohort not in {"trainingJSON", "testingJSON"}:
        raise ValueError("cohort must be trainingJSON or testingJSON")
    if source_split != "trainingSamples":
        raise ValueError("only labelled trainingSamples are accepted for benchmarking")
    emg_size = round(200.0 * window_ms / 1000.0)
    imu_size = round(50.0 * window_ms / 1000.0)
    if emg_size < 3 or imu_size < 1:
        raise ValueError("window is too short")
    rows: dict[str, list] = {key: [] for key in ("emg", "imu", "labels", "users", "cohort", "trials", "weight")}
    with zipfile.ZipFile(archive) as handle:
        for user in users:
            member = f"EMG-EPN612 Dataset/{cohort}/user{int(user)}/user{int(user)}.json"
            try:
                payload = json.loads(handle.read(member))
            except KeyError as exc:
                raise ValueError(f"EPN612 user is absent: {cohort}/user{user}") from exc
            if payload["generalInfo"]["samplingFrequencyInHertz"] != 200:
                raise ValueError(f"unexpected EPN612 sample rate for user{user}")
            for sample_id, sample in payload[source_split].items():
                gesture = sample.get("gestureName")
                if gesture not in GESTURE_TO_LABEL:
                    raise ValueError(f"missing or unknown gesture label in {member}:{sample_id}")
                emg = _matrix(sample["emg"], tuple(f"ch{index}" for index in range(1, 9)))
                accel = _matrix(sample["accelerometer"], ("x", "y", "z"))
                gyro = _matrix(sample["gyroscope"], ("x", "y", "z"))
                onset = int(sample["startPointforGestureExecution"])
                if gesture == "noGesture":
                    # The source often stores a sentinel near the end for rest trials.
                    onset = max(0, (len(emg) - windows_per_trial * emg_size) // 2)
                else:
                    onset = min(max(0, onset), max(0, len(emg) - windows_per_trial * emg_size))
                made = 0
                for offset in range(windows_per_trial):
                    emg_start = onset + offset * emg_size
                    imu_start = round(emg_start * 50.0 / 200.0)
                    emg_window = emg[emg_start:emg_start + emg_size]
                    imu_window = np.column_stack((accel[imu_start:imu_start + imu_size], gyro[imu_start:imu_start + imu_size]))
                    if len(emg_window) != emg_size or len(imu_window) != imu_size:
                        break
                    rows["emg"].append(emg_window)
                    rows["imu"].append(imu_window)
                    rows["labels"].append(GESTURE_TO_LABEL[gesture])
                    rows["users"].append(int(user))
                    rows["cohort"].append(cohort)
                    rows["trials"].append(f"{cohort}:user{user}:{sample_id}")
                    made += 1
                if made == 0:
                    raise ValueError(f"EPN612 trial has no complete post-onset window: {member}:{sample_id}")
                rows["weight"].extend([1.0 / made] * made)
    if not rows["emg"]:
        raise ValueError("no EPN612 windows matched")
    weights = np.asarray(rows["weight"], dtype=np.float64)
    weights *= len(weights) / weights.sum()
    return EpnWindows(
        FeatureBatch(np.stack(rows["emg"]), 200.0, np.stack(rows["imu"])),
        np.asarray(rows["labels"], dtype=np.int64), np.asarray(rows["users"], dtype=np.int64),
        np.asarray(rows["cohort"]), np.asarray(rows["trials"]), weights,
    )
