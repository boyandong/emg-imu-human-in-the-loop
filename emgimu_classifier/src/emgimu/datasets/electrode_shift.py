from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import re
from typing import Iterable
import zipfile

import numpy as np

from emgimu.feature_bank import FeatureBatch


PATH_RE = re.compile(r"CIILData-main/ElectrodeShift/subject(?P<subject>\d+)/(?P<domain>training|trial_[1-4])/R_(?P<rep>\d+)_C_(?P<label>[0-4])\.csv$")
CLASS_NAMES = ("close", "open", "rest", "flexion", "extension")


@dataclass(frozen=True, slots=True)
class ShiftWindows:
    batch: FeatureBatch
    labels: np.ndarray
    subjects: np.ndarray
    domains: np.ndarray
    trials: np.ndarray
    sample_weight: np.ndarray


def load_electrode_shift_windows(archive: str | Path, *, subjects: Iterable[int], domains: Iterable[str],
                                  window_ms: float = 200.0, maximum_windows_per_trial: int = 8) -> ShiftWindows:
    subject_set, domain_set = {int(x) for x in subjects}, {str(x) for x in domains}; size = round(200 * window_ms / 1000)
    rows = {key: [] for key in ("emg", "labels", "subjects", "domains", "trials", "weight")}
    with zipfile.ZipFile(archive) as handle:
        for member in sorted(handle.namelist()):
            match = PATH_RE.match(member)
            if match is None: continue
            subject, domain = int(match.group("subject")), match.group("domain")
            if subject not in subject_set or domain not in domain_set: continue
            values = np.loadtxt(BytesIO(handle.read(member)), delimiter=",", dtype=np.float32)
            if values.ndim != 2 or values.shape[1] != 8 or not np.all(np.isfinite(values)):
                raise ValueError(f"invalid Electrode Shift CSV: {member}")
            starts = np.arange(0, max(len(values) - size + 1, 0), size)
            if len(starts) > maximum_windows_per_trial:
                starts = starts[np.linspace(0, len(starts) - 1, maximum_windows_per_trial).round().astype(int)]
            if not len(starts): raise ValueError(f"Electrode Shift trial too short: {member}")
            for start in starts:
                rows["emg"].append(values[start:start + size]); rows["labels"].append(int(match.group("label")))
                rows["subjects"].append(subject); rows["domains"].append(domain); rows["trials"].append(member)
                rows["weight"].append(1.0 / len(starts))
    if not rows["emg"]: raise ValueError("no Electrode Shift files matched")
    weights = np.asarray(rows["weight"], dtype=float); weights *= len(weights) / weights.sum()
    return ShiftWindows(FeatureBatch(np.stack(rows["emg"]), 200.0), np.asarray(rows["labels"], dtype=int),
        np.asarray(rows["subjects"], dtype=int), np.asarray(rows["domains"]), np.asarray(rows["trials"]), weights)
