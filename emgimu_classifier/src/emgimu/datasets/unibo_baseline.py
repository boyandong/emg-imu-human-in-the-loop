from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import platform
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..baseline import BaselinePredictor, TemperatureScaler, _softmax
from ..metrics import expected_calibration_error, risk_coverage_curve
from ..state import Gesture
from .benchmark import BenchmarkDatasetError, check_benchmark_dataset, load_benchmark_trial


HAND_CLASSES = np.asarray(
    [int(Gesture.NEUTRAL), int(Gesture.PINCH), int(Gesture.FIST), int(Gesture.OPEN)],
    dtype=np.int16,
)
HAND_NAMES = {int(value): Gesture(int(value)).name for value in HAND_CLASSES}
FEATURE_SCHEMA = "unibo_rectified_emg_td24_v1"


@dataclass(frozen=True, slots=True)
class UniBoSvmConfig:
    window_samples: int = 40
    hop_samples: int = 40
    max_train_windows_per_trial_label: int = 6
    c_values: tuple[float, ...] = (1.0, 10.0)
    gamma_values: tuple[str | float, ...] = ("scale",)
    minimum_validation_coverage: float = 0.90
    minimum_class_coverage: float = 0.75
    random_seed: int = 42
    cache_size_mb: int = 4096

    def validate(self) -> None:
        if self.window_samples < 3 or self.hop_samples < 1:
            raise ValueError("window_samples must be >=3 and hop_samples must be positive")
        if self.max_train_windows_per_trial_label < 1:
            raise ValueError("max_train_windows_per_trial_label must be positive")
        if not self.c_values or any(float(value) <= 0 for value in self.c_values):
            raise ValueError("c_values must contain positive values")
        if not self.gamma_values:
            raise ValueError("gamma_values cannot be empty")
        if not 0 < self.minimum_validation_coverage <= 1:
            raise ValueError("minimum_validation_coverage must lie in (0,1]")
        if not 0 < self.minimum_class_coverage <= 1:
            raise ValueError("minimum_class_coverage must lie in (0,1]")


@dataclass(frozen=True, slots=True)
class UniBoWindows:
    features: np.ndarray
    labels: np.ndarray
    trial_id: np.ndarray
    subject_id: np.ndarray
    session_id: np.ndarray
    posture: np.ndarray
    timestamp_ms: np.ndarray

    def __len__(self) -> int:
        return len(self.labels)


@dataclass(frozen=True, slots=True)
class UniBoRawWindows:
    emg: np.ndarray
    labels: np.ndarray
    trial_id: np.ndarray
    subject_id: np.ndarray
    session_id: np.ndarray
    posture: np.ndarray
    timestamp_ms: np.ndarray

    def __len__(self) -> int:
        return len(self.labels)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def extract_unibo_emg_features(window: np.ndarray) -> np.ndarray:
    """Features for the four rectified/envelope-like named UniBo channels."""
    x = np.asarray(window, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 4 or len(x) < 3:
        raise ValueError("UniBo EMG window must have shape [samples>=3,4]")
    if not np.isfinite(x).all():
        raise ValueError("UniBo EMG window contains NaN or Inf")
    rms = np.sqrt(np.mean(x * x, axis=0))
    mav = np.mean(np.abs(x), axis=0)
    standard_deviation = np.std(x, axis=0)
    waveform_length = np.sum(np.abs(np.diff(x, axis=0)), axis=0) / (len(x) - 1)
    interquartile_range = np.percentile(x, 75, axis=0) - np.percentile(x, 25, axis=0)
    energy = rms * rms
    relative_energy = energy / max(float(energy.sum()), 1e-12)
    return np.concatenate([
        rms, mav, standard_deviation, waveform_length, interquartile_range, relative_energy,
    ]).astype(np.float32)


def _uniform_subset(indices: Sequence[int], maximum: int | None) -> list[int]:
    values = list(indices)
    if maximum is None or len(values) <= maximum:
        return values
    positions = np.linspace(0, len(values) - 1, maximum)
    return [values[int(round(position))] for position in positions]


def _empty_windows() -> UniBoWindows:
    return UniBoWindows(
        features=np.empty((0, 24), dtype=np.float32),
        labels=np.empty(0, dtype=np.int16),
        trial_id=np.empty(0, dtype="U1"),
        subject_id=np.empty(0, dtype="U1"),
        session_id=np.empty(0, dtype="U1"),
        posture=np.empty(0, dtype=np.int16),
        timestamp_ms=np.empty(0, dtype=np.float64),
    )


def load_unibo_windows(
    dataset_root: str | Path,
    splits_to_load: Iterable[str],
    *,
    window_samples: int,
    hop_samples: int,
    max_windows_per_trial_label: Mapping[str, int | None] | None = None,
) -> dict[str, UniBoWindows]:
    root = Path(dataset_root)
    requested = tuple(dict.fromkeys(splits_to_load))
    if not requested or set(requested) - {"train", "validation", "test"}:
        raise ValueError("splits_to_load must contain train, validation, or test")
    split_document = json.loads((root / "splits.json").read_text(encoding="utf-8"))
    membership: dict[str, str] = {}
    for split, groups in split_document["groups"].items():
        for group in groups:
            if group in membership:
                raise BenchmarkDatasetError(f"subject/day group appears twice: {group}")
            membership[str(group)] = str(split)

    rows: dict[str, dict[str, list[Any]]] = {
        split: {
            "features": [], "labels": [], "trial_id": [], "subject_id": [],
            "session_id": [], "posture": [], "timestamp_ms": [],
        }
        for split in requested
    }
    for path in sorted((root / "trials").rglob("*.npz")):
        trial = load_benchmark_trial(path, expected_channels=4, expected_rate_hz=200.0)
        if not trial.benchmark_eligible:
            continue
        group = f"{trial.subject_id}/{trial.session_id}"
        split = membership.get(group)
        if split not in rows:
            continue
        candidates: dict[int, list[int]] = {int(label): [] for label in HAND_CLASSES}
        for end in range(window_samples, len(trial.emg) + 1, hop_samples):
            start = end - window_samples
            if not bool(np.all(trial.stable_mask[start:end])):
                continue
            labels = np.unique(trial.hand_label[start:end])
            if len(labels) != 1 or int(labels[0]) not in candidates:
                continue
            candidates[int(labels[0])].append(end)
        maximum = None if max_windows_per_trial_label is None else max_windows_per_trial_label.get(split)
        for label in HAND_CLASSES:
            for end in _uniform_subset(candidates[int(label)], maximum):
                start = end - window_samples
                rows[split]["features"].append(extract_unibo_emg_features(trial.emg[start:end]))
                rows[split]["labels"].append(int(label))
                rows[split]["trial_id"].append(trial.trial_id)
                rows[split]["subject_id"].append(trial.subject_id)
                rows[split]["session_id"].append(trial.session_id)
                rows[split]["posture"].append(trial.posture_label)
                rows[split]["timestamp_ms"].append(float(trial.timestamp_ms[end - 1]))

    result: dict[str, UniBoWindows] = {}
    for split, values in rows.items():
        if not values["features"]:
            result[split] = _empty_windows()
            continue
        result[split] = UniBoWindows(
            features=np.stack(values["features"]).astype(np.float32),
            labels=np.asarray(values["labels"], dtype=np.int16),
            trial_id=np.asarray(values["trial_id"]),
            subject_id=np.asarray(values["subject_id"]),
            session_id=np.asarray(values["session_id"]),
            posture=np.asarray(values["posture"], dtype=np.int16),
            timestamp_ms=np.asarray(values["timestamp_ms"], dtype=np.float64),
        )
    return result


def load_unibo_raw_windows(
    dataset_root: str | Path,
    splits_to_load: Iterable[str],
    *,
    window_samples: int,
    hop_samples: int,
    max_windows_per_trial_label: Mapping[str, int | None] | None = None,
) -> dict[str, UniBoRawWindows]:
    """Load native four-channel windows without inventing an eight-channel layout."""
    root = Path(dataset_root)
    requested = tuple(dict.fromkeys(splits_to_load))
    if not requested or set(requested) - {"train", "validation", "test"}:
        raise ValueError("splits_to_load must contain train, validation, or test")
    split_document = json.loads((root / "splits.json").read_text(encoding="utf-8"))
    membership: dict[str, str] = {}
    for split, groups in split_document["groups"].items():
        for group in groups:
            if group in membership:
                raise BenchmarkDatasetError(f"subject/day group appears twice: {group}")
            membership[str(group)] = str(split)

    rows: dict[str, dict[str, list[Any]]] = {
        split: {
            "emg": [], "labels": [], "trial_id": [], "subject_id": [],
            "session_id": [], "posture": [], "timestamp_ms": [],
        }
        for split in requested
    }
    for path in sorted((root / "trials").rglob("*.npz")):
        trial = load_benchmark_trial(path, expected_channels=4, expected_rate_hz=200.0)
        if not trial.benchmark_eligible:
            continue
        group = f"{trial.subject_id}/{trial.session_id}"
        split = membership.get(group)
        if split not in rows:
            continue
        candidates: dict[int, list[int]] = {int(label): [] for label in HAND_CLASSES}
        for end in range(window_samples, len(trial.emg) + 1, hop_samples):
            start = end - window_samples
            if not bool(np.all(trial.stable_mask[start:end])):
                continue
            labels = np.unique(trial.hand_label[start:end])
            if len(labels) != 1 or int(labels[0]) not in candidates:
                continue
            candidates[int(labels[0])].append(end)
        maximum = None if max_windows_per_trial_label is None else max_windows_per_trial_label.get(split)
        for label in HAND_CLASSES:
            for end in _uniform_subset(candidates[int(label)], maximum):
                start = end - window_samples
                rows[split]["emg"].append(np.asarray(trial.emg[start:end], dtype=np.float32))
                rows[split]["labels"].append(int(label))
                rows[split]["trial_id"].append(trial.trial_id)
                rows[split]["subject_id"].append(trial.subject_id)
                rows[split]["session_id"].append(trial.session_id)
                rows[split]["posture"].append(trial.posture_label)
                rows[split]["timestamp_ms"].append(float(trial.timestamp_ms[end - 1]))

    result: dict[str, UniBoRawWindows] = {}
    for split, values in rows.items():
        if not values["emg"]:
            result[split] = UniBoRawWindows(
                emg=np.empty((0, window_samples, 4), dtype=np.float32),
                labels=np.empty(0, dtype=np.int16), trial_id=np.empty(0, dtype="U1"),
                subject_id=np.empty(0, dtype="U1"), session_id=np.empty(0, dtype="U1"),
                posture=np.empty(0, dtype=np.int16), timestamp_ms=np.empty(0, dtype=np.float64),
            )
            continue
        result[split] = UniBoRawWindows(
            emg=np.stack(values["emg"]).astype(np.float32),
            labels=np.asarray(values["labels"], dtype=np.int16),
            trial_id=np.asarray(values["trial_id"]),
            subject_id=np.asarray(values["subject_id"]),
            session_id=np.asarray(values["session_id"]),
            posture=np.asarray(values["posture"], dtype=np.int16),
            timestamp_ms=np.asarray(values["timestamp_ms"], dtype=np.float64),
        )
    return result


def hierarchical_segment_weights(windows: UniBoWindows) -> np.ndarray:
    """Equal subject/day -> trial -> truth segment -> window weighting."""
    if not len(windows):
        raise ValueError("cannot weight empty windows")
    groups = np.char.add(np.char.add(windows.subject_id.astype(str), "/"), windows.session_id.astype(str))
    weights = np.zeros(len(windows), dtype=np.float64)
    unique_groups = np.unique(groups)
    for group in unique_groups:
        group_mask = groups == group
        trials = np.unique(windows.trial_id[group_mask])
        for trial in trials:
            trial_mask = group_mask & (windows.trial_id == trial)
            labels = np.unique(windows.labels[trial_mask])
            for label in labels:
                mask = trial_mask & (windows.labels == label)
                weights[mask] = 1.0 / (
                    len(unique_groups) * len(trials) * len(labels) * int(np.count_nonzero(mask))
                )
    if np.any(weights <= 0):
        raise RuntimeError("hierarchical weighting left unweighted windows")
    return weights * len(weights)


def _new_svm(c_value: float, gamma: str | float, cache_size_mb: int) -> Any:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    return Pipeline([
        ("scale", StandardScaler()),
        ("svm", SVC(
            C=float(c_value), gamma=gamma, kernel="rbf", class_weight="balanced",
            decision_function_shape="ovr", cache_size=float(cache_size_mb),
        )),
    ])


def _decision_logits(model: Any, features: np.ndarray) -> np.ndarray:
    logits = np.asarray(model.decision_function(features), dtype=np.float64)
    if logits.ndim == 1:
        logits = np.column_stack([-logits, logits])
    return logits


def _metric_summary(
    truth: np.ndarray,
    predicted: np.ndarray,
    confidence: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=HAND_CLASSES, sample_weight=weights, zero_division=0,
    )
    coverage = float(np.average(predicted >= 0, weights=weights))
    return {
        "accuracy": float(accuracy_score(truth, predicted, sample_weight=weights)),
        "macro_f1": float(np.mean(f1)),
        "coverage": coverage,
        "accepted_risk": (
            float(np.average(predicted[predicted >= 0] != truth[predicted >= 0],
                             weights=weights[predicted >= 0]))
            if np.any(predicted >= 0) else None
        ),
        "ece": float(expected_calibration_error(
            truth, predicted, confidence, sample_weight=weights,
        )),
        "per_class": {
            HAND_NAMES[int(label)]: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "weighted_support": float(support[index]),
                "coverage": float(np.average(
                    predicted[truth == label] >= 0, weights=weights[truth == label],
                )),
            }
            for index, label in enumerate(HAND_CLASSES)
        },
        "confusion_matrix": confusion_matrix(
            truth, predicted, labels=np.r_[-1, HAND_CLASSES],
        ).astype(int).tolist(),
        "confusion_labels": ["UNKNOWN", *[HAND_NAMES[int(label)] for label in HAND_CLASSES]],
    }


def _select_threshold(
    probabilities: np.ndarray,
    truth: np.ndarray,
    weights: np.ndarray,
    *,
    minimum_coverage: float,
    minimum_class_coverage: float,
) -> float:
    indices = probabilities.argmax(axis=1)
    classes = HAND_CLASSES[indices]
    return BaselinePredictor._select_threshold(
        probabilities, indices, classes, truth, sample_weight=weights,
        min_coverage=minimum_coverage, min_class_coverage=minimum_class_coverage,
    )


def _aggregate_trial_label_segments(
    windows: UniBoWindows,
    probabilities: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    keys = np.asarray([
        f"{trial}\x1f{int(label)}" for trial, label in zip(windows.trial_id, windows.labels)
    ])
    truths: list[int] = []
    predicted: list[int] = []
    confidence: list[float] = []
    segment_weights: list[float] = []
    groups = np.char.add(np.char.add(windows.subject_id.astype(str), "/"), windows.session_id.astype(str))
    unique_groups = np.unique(groups)
    for group in unique_groups:
        group_mask = groups == group
        group_keys = np.unique(keys[group_mask])
        for key in group_keys:
            mask = group_mask & (keys == key)
            mean_probability = np.mean(probabilities[mask], axis=0)
            index = int(np.argmax(mean_probability))
            q = float(mean_probability[index])
            truths.append(int(windows.labels[np.flatnonzero(mask)[0]]))
            predicted.append(int(HAND_CLASSES[index]) if q >= threshold else int(Gesture.UNKNOWN))
            confidence.append(q)
            segment_weights.append(1.0 / (len(unique_groups) * len(group_keys)))
    raw_weights = np.asarray(segment_weights, dtype=np.float64)
    return (
        np.asarray(truths, dtype=np.int16),
        np.asarray(predicted, dtype=np.int16),
        np.asarray(confidence, dtype=np.float64),
        raw_weights * len(raw_weights),
    )


def _stratified_metrics(
    windows: UniBoWindows,
    predicted: np.ndarray,
    weights: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    from sklearn.metrics import f1_score

    dimensions = {
        "subject": windows.subject_id,
        "day": windows.session_id,
        "posture": windows.posture.astype(str),
        "subject_day": np.char.add(
            np.char.add(windows.subject_id.astype(str), "/"), windows.session_id.astype(str),
        ),
    }
    result: dict[str, dict[str, float | int]] = {}
    for dimension, values in dimensions.items():
        for value in np.unique(values):
            mask = values == value
            result[f"{dimension}:{value}"] = {
                "windows": int(np.count_nonzero(mask)),
                "macro_f1": float(f1_score(
                    windows.labels[mask], predicted[mask], labels=HAND_CLASSES,
                    average="macro", sample_weight=weights[mask], zero_division=0,
                )),
                "coverage": float(np.average(predicted[mask] >= 0, weights=weights[mask])),
            }
    return result


def _predict(
    model: Any,
    temperature: float,
    threshold: float,
    windows: UniBoWindows,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logits = _decision_logits(model, windows.features)
    probabilities = _softmax(logits, temperature)
    indices = probabilities.argmax(axis=1)
    raw = HAND_CLASSES[indices].astype(np.int16)
    confidence = probabilities[np.arange(len(probabilities)), indices]
    selected = np.where(confidence >= threshold, raw, int(Gesture.UNKNOWN)).astype(np.int16)
    return raw, selected, probabilities


def evaluate_unibo_probabilities(
    split: str,
    windows: UniBoWindows | UniBoRawWindows,
    probabilities: np.ndarray,
    threshold: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (len(windows), len(HAND_CLASSES)):
        raise ValueError("probabilities must have shape [windows,4]")
    indices = probabilities.argmax(axis=1)
    raw = HAND_CLASSES[indices].astype(np.int16)
    confidence = probabilities.max(axis=1)
    selected = np.where(
        confidence >= threshold, raw, int(Gesture.UNKNOWN),
    ).astype(np.int16)
    weights = hierarchical_segment_weights(windows)
    segment_truth, segment_selected, segment_confidence, segment_weights = (
        _aggregate_trial_label_segments(windows, probabilities, threshold)
    )
    segment_raw_truth, segment_raw, _, segment_raw_weights = (
        _aggregate_trial_label_segments(windows, probabilities, 0.0)
    )
    if not np.array_equal(segment_truth, segment_raw_truth):
        raise RuntimeError("segment aggregation order changed")
    summary = {
        "split": split,
        "windows": len(windows),
        "trials": int(len(np.unique(windows.trial_id))),
        "subject_day_groups": int(len(np.unique(np.char.add(
            np.char.add(windows.subject_id.astype(str), "/"), windows.session_id.astype(str),
        )))),
        "window_raw": _metric_summary(windows.labels, raw, confidence, weights),
        "window_selective": _metric_summary(windows.labels, selected, confidence, weights),
        "trial_label_segment_raw": _metric_summary(
            segment_truth, segment_raw, segment_confidence, segment_raw_weights,
        ),
        "trial_label_segment_selective": _metric_summary(
            segment_truth, segment_selected, segment_confidence, segment_weights,
        ),
        "stratified_selective": _stratified_metrics(windows, selected, weights),
        "risk_coverage": [
            {"threshold": float(row[0]), "coverage": float(row[1]), "risk": float(row[2])}
            for row in risk_coverage_curve(
                windows.labels, raw, confidence, sample_weight=weights,
            )
        ],
    }
    arrays = {
        "raw": raw,
        "selected": selected,
        "probabilities": probabilities,
        "confidence": confidence,
        "weights": weights,
    }
    return summary, arrays


def _evaluate_split(
    split: str,
    model: Any,
    temperature: float,
    threshold: float,
    windows: UniBoWindows,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    _, _, probabilities = _predict(model, temperature, threshold, windows)
    return evaluate_unibo_probabilities(split, windows, probabilities, threshold)


def _write_predictions(
    path: Path,
    split_rows: Sequence[
        tuple[str, UniBoWindows | UniBoRawWindows, dict[str, np.ndarray]]
    ],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "split", "subject_id", "session_id", "trial_id", "posture", "timestamp_ms",
            "true_label", "raw_prediction", "selective_prediction", "confidence", "weight",
            *[f"p_{HAND_NAMES[int(label)].lower()}" for label in HAND_CLASSES],
        ])
        for split, windows, arrays in split_rows:
            for index in range(len(windows)):
                writer.writerow([
                    split, windows.subject_id[index], windows.session_id[index],
                    windows.trial_id[index], int(windows.posture[index]),
                    float(windows.timestamp_ms[index]), int(windows.labels[index]),
                    int(arrays["raw"][index]), int(arrays["selected"][index]),
                    float(arrays["confidence"][index]), float(arrays["weights"][index]),
                    *[float(value) for value in arrays["probabilities"][index]],
                ])


def _write_confusion_matrices(path: Path, metrics: Mapping[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "level", "mode", "actual", "predicted", "count"])
        for split, split_metrics in metrics.items():
            if split not in {"validation", "test"}:
                continue
            for key in (
                "window_raw", "window_selective", "trial_label_segment_raw",
                "trial_label_segment_selective",
            ):
                level = "window" if key.startswith("window") else "trial_label_segment"
                mode = "selective" if key.endswith("selective") else "raw"
                report = split_metrics[key]
                labels = report["confusion_labels"]
                for row_index, actual in enumerate(labels):
                    for column_index, predicted in enumerate(labels):
                        writer.writerow([
                            split, level, mode, actual, predicted,
                            report["confusion_matrix"][row_index][column_index],
                        ])


def _write_risk_coverage(path: Path, metrics: Mapping[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "threshold", "coverage", "risk"])
        for split in ("validation", "test"):
            if split not in metrics:
                continue
            for row in metrics[split]["risk_coverage"]:
                writer.writerow([split, row["threshold"], row["coverage"], row["risk"]])


def _git_metadata(repository_root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *arguments], cwd=repository_root, text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "working_tree_dirty": bool(run("status", "--porcelain")),
    }


def run_svm_baseline(
    dataset_root: str | Path,
    output_root: str | Path,
    run_id: str,
    *,
    config: UniBoSvmConfig | None = None,
    evaluate_test: bool = False,
) -> dict[str, Any]:
    cfg = config or UniBoSvmConfig()
    cfg.validate()
    dataset = Path(dataset_root).resolve()
    output = Path(output_root).resolve()
    final_model = output / "models" / run_id
    final_result = output / "results" / run_id
    if final_model.exists() or final_result.exists():
        raise FileExistsError(f"run_id already exists and will not be overwritten: {run_id}")
    integrity = check_benchmark_dataset(dataset)
    if integrity["status"] != "ok":
        raise BenchmarkDatasetError(f"benchmark integrity failed: {integrity['errors']}")

    output.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{run_id}.tmp-", dir=output))
    model_stage = stage / "model"
    result_stage = stage / "result"
    model_stage.mkdir()
    result_stage.mkdir()
    started = time.time()
    try:
        print(json.dumps({"event": "loading_train_validation"}), flush=True)
        loaded = load_unibo_windows(
            dataset, ("train", "validation"), window_samples=cfg.window_samples,
            hop_samples=cfg.hop_samples,
            max_windows_per_trial_label={
                "train": cfg.max_train_windows_per_trial_label,
                "validation": None,
            },
        )
        train = loaded["train"]
        validation = loaded["validation"]
        if set(np.unique(train.labels)) != set(HAND_CLASSES) or set(np.unique(validation.labels)) != set(HAND_CLASSES):
            raise RuntimeError("train and validation must both contain all four hand classes")
        train_weights = hierarchical_segment_weights(train)
        validation_weights = hierarchical_segment_weights(validation)
        print(json.dumps({
            "event": "windows_ready", "train_windows": len(train),
            "validation_windows": len(validation), "train_trials": len(np.unique(train.trial_id)),
            "validation_trials": len(np.unique(validation.trial_id)),
        }), flush=True)

        from sklearn.metrics import f1_score
        search_rows: list[dict[str, Any]] = []
        best: tuple[float, float, Any, dict[str, Any]] | None = None
        for c_value in cfg.c_values:
            for gamma in cfg.gamma_values:
                fit_started = time.time()
                model = _new_svm(c_value, gamma, cfg.cache_size_mb)
                model.fit(train.features, train.labels, svm__sample_weight=train_weights)
                prediction = model.predict(validation.features)
                macro_f1 = float(f1_score(
                    validation.labels, prediction, labels=HAND_CLASSES, average="macro",
                    sample_weight=validation_weights, zero_division=0,
                ))
                elapsed = time.time() - fit_started
                row = {
                    "C": float(c_value), "gamma": gamma, "validation_macro_f1": macro_f1,
                    "fit_and_predict_seconds": elapsed,
                    "support_vectors": int(np.sum(model.named_steps["svm"].n_support_)),
                }
                search_rows.append(row)
                print(json.dumps({"event": "candidate_complete", **row}), flush=True)
                candidate = (macro_f1, -elapsed, model, row)
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
        if best is None:
            raise RuntimeError("hyperparameter search produced no model")
        model = best[2]
        chosen = best[3]
        validation_logits = _decision_logits(model, validation.features)
        temperature = TemperatureScaler().fit(
            validation_logits, validation.labels, HAND_CLASSES, validation_weights,
        ).temperature
        validation_probabilities = _softmax(validation_logits, temperature)
        threshold = _select_threshold(
            validation_probabilities, validation.labels, validation_weights,
            minimum_coverage=cfg.minimum_validation_coverage,
            minimum_class_coverage=cfg.minimum_class_coverage,
        )
        print(json.dumps({
            "event": "model_selected", "C": chosen["C"], "gamma": chosen["gamma"],
            "temperature": temperature, "threshold": threshold,
        }), flush=True)

        configuration = {
            **asdict(cfg),
            "c_values": list(cfg.c_values),
            "gamma_values": list(cfg.gamma_values),
            "run_id": run_id,
            "dataset_root": str(dataset),
            "feature_schema": FEATURE_SCHEMA,
            "sample_rate_hz": 200,
            "window_ms": cfg.window_samples * 5,
            "hop_ms": cfg.hop_samples * 5,
            "split_protocol": "Day 1-5 train; Day 6 validation; Day 7-8 test",
            "weighting": "equal_subject_day_then_trial_then_truth_segment_then_window_v1",
            "test_evaluated": bool(evaluate_test),
        }
        artifact = {
            "artifact_format_version": 1,
            "model_kind": "unibo_four_channel_rbf_svm",
            "model": model,
            "classes": HAND_CLASSES,
            "class_names": HAND_NAMES,
            "temperature": temperature,
            "threshold": threshold,
            "chosen_hyperparameters": chosen,
            "config": configuration,
        }
        model_path = model_stage / "model.pkl"
        with model_path.open("wb") as stream:
            pickle.dump(artifact, stream)
        model_sha256 = _sha256(model_path)

        validation_metrics, validation_arrays = _evaluate_split(
            "validation", model, temperature, threshold, validation,
        )
        metrics: dict[str, Any] = {"validation": validation_metrics}
        prediction_rows: list[tuple[str, UniBoWindows, dict[str, np.ndarray]]] = [
            ("validation", validation, validation_arrays),
        ]

        if evaluate_test:
            print(json.dumps({
                "event": "model_frozen_opening_test", "model_sha256": model_sha256,
            }), flush=True)
            test = load_unibo_windows(
                dataset, ("test",), window_samples=cfg.window_samples,
                hop_samples=cfg.hop_samples, max_windows_per_trial_label={"test": None},
            )["test"]
            test_metrics, test_arrays = _evaluate_split(
                "test", model, temperature, threshold, test,
            )
            metrics["test"] = test_metrics
            prediction_rows.append(("test", test, test_arrays))

        _json_dump(result_stage / "config.json", configuration)
        _json_dump(result_stage / "metrics.json", metrics)
        with (result_stage / "hyperparameter_search.csv").open(
            "w", newline="", encoding="utf-8",
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(search_rows[0]))
            writer.writeheader()
            writer.writerows(search_rows)
        _write_predictions(result_stage / "predictions.csv", prediction_rows)
        _write_confusion_matrices(result_stage / "confusion_matrix.csv", metrics)
        _write_risk_coverage(result_stage / "risk_coverage.csv", metrics)

        import scipy
        import sklearn

        source_file = Path(__file__).resolve()
        repository_root = Path(__file__).resolve().parents[4]
        run_manifest = {
            "run_id": run_id,
            "status": "complete",
            "model_sha256": model_sha256,
            "benchmark_manifest_sha256": _sha256(dataset / "manifest.json"),
            "splits_sha256": _sha256(dataset / "splits.json"),
            "runner_sha256": _sha256(source_file),
            "elapsed_seconds": time.time() - started,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "dependencies": {
                "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
            },
            "git": _git_metadata(repository_root),
            "integrity": integrity,
            "test_opened_after_model_hash": bool(evaluate_test),
        }
        _json_dump(result_stage / "run_manifest.json", run_manifest)
        final_model.parent.mkdir(parents=True, exist_ok=True)
        final_result.parent.mkdir(parents=True, exist_ok=True)
        model_stage.replace(final_model)
        result_stage.replace(final_result)
        shutil.rmtree(stage, ignore_errors=True)
        result = {
            "run_id": run_id,
            "model": str(final_model / "model.pkl"),
            "results": str(final_result),
            "model_sha256": model_sha256,
            "chosen_hyperparameters": chosen,
            "temperature": temperature,
            "threshold": threshold,
            "metrics": metrics,
        }
        print(json.dumps({
            "event": "complete", "run_id": run_id, "results": str(final_result),
        }), flush=True)
        return result
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the strict UniBo four-class H RBF-SVM baseline")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--window-samples", type=int, default=40)
    parser.add_argument("--hop-samples", type=int, default=40)
    parser.add_argument("--max-train-windows-per-trial-label", type=int, default=6)
    parser.add_argument("--c-values", type=float, nargs="+", default=[1.0, 10.0])
    parser.add_argument("--cache-size-mb", type=int, default=4096)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    config = UniBoSvmConfig(
        window_samples=arguments.window_samples,
        hop_samples=arguments.hop_samples,
        max_train_windows_per_trial_label=arguments.max_train_windows_per_trial_label,
        c_values=tuple(arguments.c_values),
        cache_size_mb=arguments.cache_size_mb,
    )
    result = run_svm_baseline(
        arguments.dataset, arguments.output_root, arguments.run_id,
        config=config, evaluate_test=arguments.evaluate_test,
    )
    # ASCII escaping keeps the CLI usable in Windows shells whose redirected
    # stdout still reports a legacy code page despite a UTF-8 filesystem.
    print(json.dumps(result, indent=2, ensure_ascii=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
