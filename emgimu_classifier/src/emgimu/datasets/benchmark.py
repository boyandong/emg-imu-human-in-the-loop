from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from ..state import Gesture


REQUIRED_TRIAL_FIELDS = {
    "subject_id",
    "session_id",
    "trial_id",
    "timestamp_ms",
    "emg",
    "hand_label",
    "posture_label",
    "source_label",
    "source_relabel",
    "stable_mask",
    "benchmark_eligible",
    "source_gesture",
}


class BenchmarkDatasetError(ValueError):
    pass


def _scalar_text(container: Mapping[str, np.ndarray], key: str) -> str:
    value = np.asarray(container[key])
    if value.size != 1:
        raise BenchmarkDatasetError(f"{key} must be a scalar")
    return str(value.reshape(-1)[0])


def _scalar_int(container: Mapping[str, np.ndarray], key: str) -> int:
    value = np.asarray(container[key])
    if value.size != 1:
        raise BenchmarkDatasetError(f"{key} must be a scalar")
    return int(value.reshape(-1)[0])


@dataclass(frozen=True, slots=True)
class BenchmarkTrial:
    path: Path
    subject_id: str
    session_id: str
    trial_id: str
    timestamp_ms: np.ndarray
    emg: np.ndarray
    hand_label: np.ndarray
    posture_label: int
    source_label: np.ndarray
    source_relabel: np.ndarray
    stable_mask: np.ndarray
    benchmark_eligible: bool
    source_gesture: int
    imu: np.ndarray | None = None


def load_benchmark_trial(
    path: str | Path,
    *,
    expected_channels: int | None = None,
    expected_rate_hz: float | None = None,
) -> BenchmarkTrial:
    source = Path(path)
    try:
        with np.load(source, allow_pickle=False) as handle:
            missing = sorted(REQUIRED_TRIAL_FIELDS - set(handle.files))
            if missing:
                raise BenchmarkDatasetError(f"{source} missing fields: {', '.join(missing)}")
            subject_id = _scalar_text(handle, "subject_id")
            session_id = _scalar_text(handle, "session_id")
            trial_id = _scalar_text(handle, "trial_id")
            posture_label = _scalar_int(handle, "posture_label")
            eligible_raw = _scalar_int(handle, "benchmark_eligible")
            source_gesture = _scalar_int(handle, "source_gesture")
            timestamp_ms = np.asarray(handle["timestamp_ms"], dtype=np.float64).reshape(-1)
            emg = np.asarray(handle["emg"], dtype=np.float64)
            hand_label = np.asarray(handle["hand_label"], dtype=np.int16).reshape(-1)
            source_label = np.asarray(handle["source_label"], dtype=np.int16).reshape(-1)
            source_relabel = np.asarray(handle["source_relabel"], dtype=np.int16).reshape(-1)
            stable_mask = np.asarray(handle["stable_mask"], dtype=bool).reshape(-1)
            imu = np.asarray(handle["imu"], dtype=np.float64) if "imu" in handle.files else None
    except BenchmarkDatasetError:
        raise
    except Exception as exc:
        raise BenchmarkDatasetError(f"cannot read trial {source}: {exc}") from exc
    n = len(timestamp_ms)
    if n < 2 or emg.ndim != 2 or emg.shape[0] != n:
        raise BenchmarkDatasetError(f"{source} has invalid timestamp/EMG shapes")
    if any(len(values) != n for values in (hand_label, source_label, source_relabel, stable_mask)):
        raise BenchmarkDatasetError(f"{source} sample arrays have inconsistent lengths")
    if expected_channels is not None and emg.shape[1] != expected_channels:
        raise BenchmarkDatasetError(
            f"{source} has {emg.shape[1]} EMG channels; expected {expected_channels}"
        )
    if not np.isfinite(timestamp_ms).all() or not np.isfinite(emg).all():
        raise BenchmarkDatasetError(f"{source} contains NaN or Inf")
    differences = np.diff(timestamp_ms)
    if np.any(differences <= 0):
        raise BenchmarkDatasetError(f"{source} timestamps are not strictly increasing")
    if expected_rate_hz is not None:
        observed_rate = 1000.0 / float(np.median(differences))
        if not np.isclose(observed_rate, expected_rate_hz, rtol=0.01):
            raise BenchmarkDatasetError(
                f"{source} rate {observed_rate:.3f} Hz differs from {expected_rate_hz:.3f} Hz"
            )
    allowed_hand = {int(item) for item in Gesture}
    invalid_hand = sorted(set(map(int, np.unique(hand_label))) - allowed_hand)
    if invalid_hand:
        raise BenchmarkDatasetError(f"{source} has invalid hand labels: {invalid_hand}")
    invalid_source = sorted(set(map(int, np.unique(source_label))) - set(range(1, 7)))
    invalid_relabel = sorted(set(map(int, np.unique(source_relabel))) - set(range(1, 7)))
    if invalid_source or invalid_relabel:
        raise BenchmarkDatasetError(
            f"{source} has invalid source labels: label={invalid_source}, relabel={invalid_relabel}"
        )

    if eligible_raw not in (0, 1):
        raise BenchmarkDatasetError(f"{source} benchmark_eligible must be boolean")
    if not 1 <= source_gesture <= 6:
        raise BenchmarkDatasetError(f"{source} source_gesture must be in 1..6")
    if imu is not None:
        if imu.ndim != 2 or imu.shape[0] != n or not np.isfinite(imu).all():
            raise BenchmarkDatasetError(f"{source} has invalid optional IMU")
    return BenchmarkTrial(
        path=source,
        subject_id=subject_id,
        session_id=session_id,
        trial_id=trial_id,
        timestamp_ms=timestamp_ms,
        emg=emg,
        hand_label=hand_label,
        posture_label=posture_label,
        source_label=source_label,
        source_relabel=source_relabel,
        stable_mask=stable_mask,
        benchmark_eligible=bool(eligible_raw),
        source_gesture=source_gesture,
        imu=imu,
    )


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise BenchmarkDatasetError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BenchmarkDatasetError(f"{path} must contain a JSON object")
    return value


def check_benchmark_dataset(
    root: str | Path,
    splits_to_check: Iterable[str] | None = None,
) -> dict[str, object]:
    dataset_root = Path(root)
    errors: list[str] = []
    warnings: list[str] = []
    for name in ("manifest.json", "label_map.json", "splits.json"):
        if not (dataset_root / name).is_file():
            errors.append(f"missing {name}")
    if errors:
        return {"status": "error", "errors": errors, "warnings": warnings, "trial_count": 0}

    try:
        manifest = _load_json(dataset_root / "manifest.json")
        splits = _load_json(dataset_root / "splits.json")
        channels = int(manifest["emg_channels"])
        rate = float(manifest["target_sample_rate_hz"])
        has_imu = bool(manifest["has_imu"])
    except (BenchmarkDatasetError, KeyError, TypeError, ValueError) as exc:
        return {"status": "error", "errors": [str(exc)], "warnings": warnings, "trial_count": 0}

    membership: dict[str, str] = {}
    try:
        split_groups = splits["groups"]
        if not isinstance(split_groups, dict):
            raise TypeError("groups must be an object")
        for split in ("train", "validation", "test"):
            rows = split_groups[split]
            if not isinstance(rows, list):
                raise TypeError(f"groups.{split} must be a list")
            for group in rows:
                key = str(group)
                if key in membership:
                    errors.append(f"subject-session group {key} appears in multiple splits")
                membership[key] = split
    except (KeyError, TypeError) as exc:
        errors.append(f"invalid splits.json: {exc}")

    selected_splits = None if splits_to_check is None else set(map(str, splits_to_check))
    if selected_splits is not None and (not selected_splits or selected_splits - {"train", "validation", "test"}):
        errors.append("splits_to_check must be a nonempty subset of train/validation/test")
    trials_root = dataset_root / "trials"
    paths = sorted(trials_root.rglob("*.npz"))
    if selected_splits is not None:
        filtered_paths = []
        for path in paths:
            relative = path.relative_to(trials_root).parts
            path_group = f"{relative[0]}/{relative[1]}" if len(relative) >= 2 else ""
            if membership.get(path_group) in selected_splits:
                filtered_paths.append(path)
        paths = filtered_paths
    if not paths:
        errors.append("no trial NPZ files found")
    trial_ids: set[str] = set()
    group_splits: dict[str, str] = {}
    eligible = 0
    for path in paths:
        try:
            trial = load_benchmark_trial(
                path, expected_channels=channels, expected_rate_hz=rate,
            )
        except BenchmarkDatasetError as exc:
            errors.append(str(exc))
            continue
        if trial.trial_id in trial_ids:
            errors.append(f"duplicate trial_id: {trial.trial_id}")
        trial_ids.add(trial.trial_id)
        group = f"{trial.subject_id}/{trial.session_id}"
        split = membership.get(group)
        if split is None:
            errors.append(f"trial group {group} is absent from splits.json")
        previous = group_splits.get(group)
        if previous is not None and previous != split:
            errors.append(f"trial group {group} crosses splits")
        elif split is not None:
            group_splits[group] = split
        if trial.benchmark_eligible:
            eligible += 1
            if not np.any(trial.stable_mask):
                warnings.append(f"eligible trial has no stable samples: {trial.trial_id}")
        elif np.any(trial.stable_mask):
            errors.append(f"ineligible trial exposes stable samples: {trial.trial_id}")
        if not 1 <= trial.posture_label <= 4:
            errors.append(f"invalid posture label in {trial.trial_id}: {trial.posture_label}")
        if np.any(trial.stable_mask & (trial.source_label != trial.source_relabel)):
            errors.append(f"relabel boundary exposed as stable: {trial.trial_id}")
        if np.any(trial.stable_mask & (trial.hand_label == int(Gesture.UNKNOWN))):
            errors.append(f"Unknown label exposed as stable: {trial.trial_id}")
        if not has_imu and trial.imu is not None:
            errors.append(f"dataset declares no IMU but trial contains imu: {trial.trial_id}")
        if manifest.get("channel_layout") == "named_muscles_non_circular" and channels != 4:
            errors.append("UniBo non-circular layout must remain four channels")

    declared = manifest.get("trial_count")
    if selected_splits is None and declared is not None and int(declared) != len(paths):
        errors.append(f"manifest trial_count={declared}, files={len(paths)}")
    return {
        "status": "ok" if not errors else "error",
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
        "trial_count": len(paths),
        "eligible_trial_count": eligible,
        "subject_session_groups": len(group_splits),
    }


def read_benchmark_report(root: str | Path) -> dict[str, object]:
    dataset_root = Path(root)
    return {
        "manifest": _load_json(dataset_root / "manifest.json"),
        "statistics": _load_json(dataset_root / "reports" / "statistics.json"),
        "integrity": _load_json(dataset_root / "reports" / "integrity.json"),
        "splits": _load_json(dataset_root / "splits.json"),
    }
