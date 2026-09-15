from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import re
from typing import Iterable
import zipfile

import numpy as np

from emgimu.feature_bank import FeatureBatch


PATH_RE = re.compile(r"data/u_(?P<user>\d+)/s_(?P<session>\d+)/g_(?P<gesture>[^/]+)/recording_(?P<speed>slow|medium|fast)_.+\.csv$")


@dataclass(frozen=True, slots=True)
class ManusWindows:
    batch: FeatureBatch
    labels: np.ndarray
    gestures: np.ndarray
    users: np.ndarray
    sessions: np.ndarray
    speeds: np.ndarray
    trials: np.ndarray
    sample_weight: np.ndarray


def load_semg_manus_windows(
    archive: str | Path, *, users: Iterable[int], sessions: Iterable[int], gestures: Iterable[str],
    speeds: Iterable[str] = ("slow", "medium", "fast"), window_ms: float = 200.0, maximum_windows_per_trial: int = 8,
) -> ManusWindows:
    user_set, session_set = {int(x) for x in users}, {int(x) for x in sessions}
    gesture_list = tuple(str(x).removeprefix("g_") for x in gestures)
    gesture_to_label = {name: index for index, name in enumerate(gesture_list)}
    speed_set = {str(x) for x in speeds}
    size = round(200.0 * window_ms / 1000.0)
    rows = {key: [] for key in ("emg", "imu", "labels", "gestures", "users", "sessions", "speeds", "trials", "weight")}
    with zipfile.ZipFile(archive) as handle:
        for member in sorted(handle.namelist()):
            match = PATH_RE.match(member)
            if match is None:
                continue
            user, session = int(match.group("user")), int(match.group("session"))
            gesture, speed = match.group("gesture"), match.group("speed")
            if user not in user_set or session not in session_set or gesture not in gesture_to_label or speed not in speed_set:
                continue
            values = np.loadtxt(BytesIO(handle.read(member)), delimiter=",", comments="#", dtype=np.float32)
            if values.ndim != 2 or values.shape[1] != 42 or not np.all(np.isfinite(values)):
                raise ValueError(f"invalid sEMG-MANUS CSV: {member} {values.shape}")
            starts = np.arange(0, max(len(values) - size + 1, 0), size)
            if len(starts) > maximum_windows_per_trial:
                starts = starts[np.linspace(0, len(starts) - 1, maximum_windows_per_trial).round().astype(int)]
            if not len(starts):
                raise ValueError(f"sEMG-MANUS trial is shorter than one window: {member}")
            for start in starts:
                rows["emg"].append(values[start:start + size, :8])
                rows["imu"].append(values[start:start + size, 12:18])
                rows["labels"].append(gesture_to_label[gesture]); rows["gestures"].append(gesture)
                rows["users"].append(user); rows["sessions"].append(session); rows["speeds"].append(speed)
                rows["trials"].append(member); rows["weight"].append(1.0 / len(starts))
    if not rows["emg"]:
        raise ValueError("no sEMG-MANUS files matched the requested subset")
    weights = np.asarray(rows["weight"], dtype=float); weights *= len(weights) / weights.sum()
    return ManusWindows(FeatureBatch(np.stack(rows["emg"]), 200.0, np.stack(rows["imu"])),
        np.asarray(rows["labels"], dtype=int), np.asarray(rows["gestures"]), np.asarray(rows["users"], dtype=int),
        np.asarray(rows["sessions"], dtype=int), np.asarray(rows["speeds"]), np.asarray(rows["trials"]), weights)
