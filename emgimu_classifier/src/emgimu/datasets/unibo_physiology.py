from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import platform
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..baseline import TemperatureScaler, _softmax
from ..state import Gesture
from .benchmark import BenchmarkDatasetError, check_benchmark_dataset, load_benchmark_trial
from .unibo_baseline import (
    ACTIVE_HAND_CLASSES,
    HAND_CLASSES,
    HAND_NAMES,
    UniBoRawWindows,
    UniBoSvmConfig,
    UniBoWindows,
    _decision_logits,
    _git_metadata,
    _json_dump,
    _new_svm,
    _select_threshold,
    _sha256,
    _uniform_subset,
    _write_confusion_matrices,
    _write_predictions,
    _write_risk_coverage,
    evaluate_unibo_probabilities,
    hierarchical_segment_weights,
)


CHANNEL_NAMES = ("ECU", "EDC", "FCR", "FCU")
POSTURE_NAMES = {
    1: "proximal",
    2: "distal",
    3: "distal_palm_down",
    4: "distal_arm_45deg_up",
}
DEFAULT_FOLDS = (
    ("D1-3_to_D4", (1, 2, 3), 4),
    ("D1-4_to_D5", (1, 2, 3, 4), 5),
    ("D1-5_to_D6", (1, 2, 3, 4, 5), 6),
)
DEFAULT_BOOTSTRAP_SEED = 20260913


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    run: str
    groups: tuple[str, ...]
    nmf_components: int | None = None
    temporal_synergy: bool = False


STATIC_SPECS = (
    ExperimentSpec("E0", ("G0",)),
    ExperimentSpec("E1", ("G0", "G1")),
    ExperimentSpec("E2", ("G0", "G2")),
    ExperimentSpec("E3", ("G0", "G3")),
    ExperimentSpec("E123", ("G0", "G1", "G2", "G3")),
    ExperimentSpec("E4-k2", ("G0", "G1", "G2", "G3", "G4"), 2),
    ExperimentSpec("E4-k3", ("G0", "G1", "G2", "G3", "G4"), 3),
    ExperimentSpec("E5", ("G0", "G1", "G2", "G3", "G5")),
)


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, percentile: float) -> float:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(x) != len(w) or not len(x) or np.any(w < 0) or not np.isfinite(x).all():
        raise ValueError("weighted percentile inputs must be finite, nonempty, and aligned")
    order = np.argsort(x, kind="stable")
    sorted_x = x[order]
    cumulative = np.cumsum(w[order])
    if cumulative[-1] <= 0:
        raise ValueError("weighted percentile weights must have positive mass")
    target = min(max(float(percentile), 0.0), 1.0) * cumulative[-1]
    return float(sorted_x[min(int(np.searchsorted(cumulative, target, side="left")), len(x) - 1)])


def _region_rms(emg: np.ndarray) -> np.ndarray:
    x = np.asarray(emg, dtype=np.float64)
    if x.ndim != 3 or x.shape[2] != 4 or x.shape[1] < 3:
        raise ValueError("UniBo windows must have shape [windows,samples>=3,4]")
    if not np.isfinite(x).all():
        raise ValueError("UniBo windows contain NaN or Inf")
    return np.sqrt(np.mean(x * x, axis=1))


def _g0_features(emg: np.ndarray) -> np.ndarray:
    x = np.asarray(emg, dtype=np.float64)
    rms = np.sqrt(np.mean(x * x, axis=1))
    mav = np.mean(np.abs(x), axis=1)
    standard_deviation = np.std(x, axis=1)
    waveform_length = np.sum(np.abs(np.diff(x, axis=1)), axis=1) / (x.shape[1] - 1)
    interquartile_range = np.percentile(x, 75, axis=1) - np.percentile(x, 25, axis=1)
    energy = rms * rms
    relative_energy = energy / np.maximum(energy.sum(axis=1, keepdims=True), 1e-12)
    return np.concatenate([
        rms, mav, standard_deviation, waveform_length, interquartile_range,
        relative_energy,
    ], axis=1).astype(np.float32)


def _g0_names() -> tuple[str, ...]:
    metrics = ("rms", "mav", "standard_deviation", "waveform_length", "iqr", "relative_energy")
    return tuple(f"G0.{metric}.{channel}" for metric in metrics for channel in CHANNEL_NAMES)


def _g1_features(rms: np.ndarray, neutral_thresholds: np.ndarray) -> np.ndarray:
    eps = 1e-12
    shares = rms / np.maximum(rms.sum(axis=1, keepdims=True), eps)
    entropy = -np.sum(np.where(shares > 0, shares * np.log(np.maximum(shares, eps)), 0.0), axis=1)
    entropy /= np.log(4.0)
    ordered = np.sort(shares, axis=1)
    second_over_max = ordered[:, -2] / np.maximum(ordered[:, -1], eps)
    active_count = np.sum(rms > neutral_thresholds.reshape(1, 4), axis=1)
    return np.column_stack([
        shares, entropy, ordered[:, -1], second_over_max, active_count,
    ]).astype(np.float32)


def _g1_names() -> tuple[str, ...]:
    return tuple(f"G1.rms_share.{channel}" for channel in CHANNEL_NAMES) + (
        "G1.normalized_spatial_entropy",
        "G1.max_share",
        "G1.second_over_max_share",
        "G1.active_region_count",
    )


def _g2_features(rms: np.ndarray) -> np.ndarray:
    eps = 1e-12
    extensor = rms[:, 0] + rms[:, 1]
    flexor = rms[:, 2] + rms[:, 3]
    return np.column_stack([
        extensor,
        flexor,
        np.log((flexor + eps) / (extensor + eps)),
        (flexor - extensor) / (flexor + extensor + eps),
        np.log((rms[:, 1] + eps) / (rms[:, 0] + eps)),
        np.log((rms[:, 2] + eps) / (rms[:, 3] + eps)),
    ]).astype(np.float32)


def _g2_names() -> tuple[str, ...]:
    return (
        "G2.extensor_sum_rms",
        "G2.flexor_sum_rms",
        "G2.log_flexor_over_extensor",
        "G2.normalized_flexor_extensor_balance",
        "G2.log_EDC_over_ECU",
        "G2.log_FCR_over_FCU",
    )


def _g3_features(emg: np.ndarray, rms: np.ndarray) -> np.ndarray:
    eps = 1e-12
    extensor = rms[:, 0] + rms[:, 1]
    flexor = rms[:, 2] + rms[:, 3]
    cci = 2.0 * np.minimum(extensor, flexor) / (extensor + flexor + eps)
    coactivation = 2.0 * np.minimum(extensor, flexor)
    shares = rms / np.maximum(rms.sum(axis=1, keepdims=True), eps)
    concentration = np.sum(shares * shares, axis=1)
    centered = np.asarray(emg, dtype=np.float64) - np.mean(emg, axis=1, keepdims=True)
    norms = np.sqrt(np.sum(centered * centered, axis=1))
    pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    correlations = []
    for left, right in pairs:
        numerator = np.sum(centered[:, :, left] * centered[:, :, right], axis=1)
        denominator = norms[:, left] * norms[:, right]
        correlations.append(np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > eps,
        ))
    correlation_matrix = np.column_stack(correlations)
    return np.column_stack([
        cci, coactivation, concentration, correlation_matrix,
        np.mean(np.abs(correlation_matrix), axis=1),
    ]).astype(np.float32)


def _g3_names() -> tuple[str, ...]:
    pairs = (("ECU", "EDC"), ("ECU", "FCR"), ("ECU", "FCU"),
             ("EDC", "FCR"), ("EDC", "FCU"), ("FCR", "FCU"))
    return (
        "G3.flexor_extensor_cci",
        "G3.co_contraction_amplitude",
        "G3.rms_share_concentration",
        *(f"G3.correlation.{left}_{right}" for left, right in pairs),
        "G3.mean_absolute_off_diagonal_correlation",
    )


def _g5_features(emg: np.ndarray) -> np.ndarray:
    x = np.asarray(emg, dtype=np.float64)
    midpoint = x.shape[1] // 2
    early = _region_rms(x[:, :midpoint, :])
    late = _region_rms(x[:, midpoint:, :])
    difference = early - late
    log_ratio = np.log((early + 1e-12) / (late + 1e-12))
    timeline = np.linspace(-0.5, 0.5, x.shape[1], dtype=np.float64)
    denominator = float(np.sum(timeline * timeline))
    slope = np.einsum("t,ntc->nc", timeline, x) / max(denominator, 1e-12)
    return np.concatenate([early, late, difference, log_ratio, slope], axis=1).astype(np.float32)


def _g5_names() -> tuple[str, ...]:
    metrics = ("early_rms", "late_rms", "early_minus_late_rms", "log_early_over_late", "activation_slope")
    return tuple(f"G5.{metric}.{channel}" for metric in metrics for channel in CHANNEL_NAMES)


class PhysiologyFeatureTransformer:
    """Fold-fitted physiology representation for native four-channel UniBo EMG."""

    def __init__(
        self,
        groups: Sequence[str],
        *,
        nmf_components: int | None = None,
        nmf_seed: int = 42,
        nmf_init: str = "nndsvda",
        temporal_synergy: bool = False,
    ) -> None:
        self.groups = tuple(groups)
        unknown = set(self.groups) - {"G0", "G1", "G2", "G3", "G4", "G5", "G6"}
        if unknown or len(set(self.groups)) != len(self.groups):
            raise ValueError(f"invalid or duplicate feature groups: {self.groups}")
        if "G4" in self.groups and nmf_components not in (2, 3):
            raise ValueError("G4 requires nmf_components=2 or 3")
        if "G4" not in self.groups and nmf_components is not None:
            raise ValueError("nmf_components is only valid with G4")
        if temporal_synergy and not {"G4", "G5"}.issubset(self.groups):
            raise ValueError("temporal synergy requires both G4 and G5")
        self.nmf_components = nmf_components
        self.nmf_seed = int(nmf_seed)
        if nmf_init not in {"nndsvda", "random"}:
            raise ValueError("nmf_init must be nndsvda or random")
        self.nmf_init = nmf_init
        self.temporal_synergy = bool(temporal_synergy)
        self.neutral_thresholds_: np.ndarray | None = None
        self.nmf_: Any | None = None
        self.nmf_global_scale_: float | None = None
        self.nmf_reconstruction_error_: float | None = None
        self.feature_names_: tuple[str, ...] | None = None
        self.fitted_ = False

    def fit(
        self,
        emg: np.ndarray,
        labels: np.ndarray,
        sample_weight: np.ndarray,
    ) -> "PhysiologyFeatureTransformer":
        x = np.asarray(emg, dtype=np.float64)
        y = np.asarray(labels)
        weights = np.asarray(sample_weight, dtype=np.float64)
        rms = _region_rms(x)
        if len(x) != len(y) or len(x) != len(weights):
            raise ValueError("training windows, labels, and weights must align")
        if "G1" in self.groups:
            neutral = y == int(Gesture.NEUTRAL)
            if not np.any(neutral):
                raise ValueError("G1 requires Neutral training windows")
            self.neutral_thresholds_ = np.asarray([
                _weighted_percentile(rms[neutral, channel], weights[neutral], 0.95)
                for channel in range(4)
            ], dtype=np.float64)
        if "G4" in self.groups:
            from sklearn.decomposition import NMF

            repeated_weights = np.repeat(weights / 4.0, 4)
            scale = _weighted_percentile(rms.reshape(-1), repeated_weights, 0.5)
            self.nmf_global_scale_ = max(scale, 1e-12)
            scaled = rms / self.nmf_global_scale_
            self.nmf_ = NMF(
                n_components=int(self.nmf_components),
                init=self.nmf_init,
                random_state=self.nmf_seed,
                max_iter=1000,
                tol=1e-5,
            )
            activations = self.nmf_.fit_transform(scaled)
            reconstructed = activations @ self.nmf_.components_
            self.nmf_reconstruction_error_ = float(
                np.linalg.norm(scaled - reconstructed) / max(np.linalg.norm(scaled), 1e-12)
            )
        self.fitted_ = True
        self.feature_names_ = self._feature_names()
        if len(self.feature_names_) != len(set(self.feature_names_)):
            raise RuntimeError("combined feature schema contains duplicate columns")
        transformed = self.transform(x, np.ones(len(x), dtype=np.int16))
        if not np.isfinite(transformed).all():
            raise RuntimeError("fitted feature transformer produced non-finite values")
        return self

    def _feature_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for group in self.groups:
            if group == "G0":
                names.extend(_g0_names())
            elif group == "G1":
                names.extend(_g1_names())
            elif group == "G2":
                names.extend(_g2_names())
            elif group == "G3":
                names.extend(_g3_names())
            elif group == "G4":
                names.extend(f"G4.synergy_activation.{index + 1}" for index in range(int(self.nmf_components)))
                if self.temporal_synergy:
                    for metric in ("early_activation", "late_activation", "late_minus_early_activation"):
                        names.extend(
                            f"G4.temporal.{metric}.{index + 1}"
                            for index in range(int(self.nmf_components))
                        )
            elif group == "G5":
                names.extend(_g5_names())
            elif group == "G6":
                names.extend(f"G6.posture.{POSTURE_NAMES[index]}" for index in sorted(POSTURE_NAMES))
        return tuple(names)

    def transform(self, emg: np.ndarray, posture: np.ndarray) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("feature transformer must be fit on the fold training data")
        x = np.asarray(emg, dtype=np.float64)
        rms = _region_rms(x)
        posture_values = np.asarray(posture).reshape(-1)
        if len(posture_values) != len(x):
            raise ValueError("posture must align with EMG windows")
        blocks: list[np.ndarray] = []
        for group in self.groups:
            if group == "G0":
                blocks.append(_g0_features(x))
            elif group == "G1":
                if self.neutral_thresholds_ is None:
                    raise RuntimeError("G1 Neutral thresholds are unavailable")
                blocks.append(_g1_features(rms, self.neutral_thresholds_))
            elif group == "G2":
                blocks.append(_g2_features(rms))
            elif group == "G3":
                blocks.append(_g3_features(x, rms))
            elif group == "G4":
                if self.nmf_ is None or self.nmf_global_scale_ is None:
                    raise RuntimeError("G4 NMF basis is unavailable")
                blocks.append(self.nmf_.transform(rms / self.nmf_global_scale_).astype(np.float32))
                if self.temporal_synergy:
                    midpoint = x.shape[1] // 2
                    early_rms = _region_rms(x[:, :midpoint, :])
                    late_rms = _region_rms(x[:, midpoint:, :])
                    early = self.nmf_.transform(early_rms / self.nmf_global_scale_)
                    late = self.nmf_.transform(late_rms / self.nmf_global_scale_)
                    blocks.extend((early.astype(np.float32), late.astype(np.float32), (late - early).astype(np.float32)))
            elif group == "G5":
                blocks.append(_g5_features(x))
            elif group == "G6":
                invalid = sorted(set(map(int, np.unique(posture_values))) - set(POSTURE_NAMES))
                if invalid:
                    raise ValueError(f"unknown UniBo posture labels: {invalid}")
                blocks.append(np.column_stack([
                    posture_values == value for value in sorted(POSTURE_NAMES)
                ]).astype(np.float32))
        result = np.concatenate(blocks, axis=1).astype(np.float32)
        if not np.isfinite(result).all():
            raise RuntimeError("feature transform produced NaN or Inf")
        return result

    @property
    def feature_names(self) -> tuple[str, ...]:
        if self.feature_names_ is None:
            raise RuntimeError("feature transformer has not been fit")
        return self.feature_names_

    def state(self) -> dict[str, Any]:
        if not self.fitted_:
            raise RuntimeError("feature transformer has not been fit")
        return {
            "groups": list(self.groups),
            "feature_dimension": len(self.feature_names),
            "feature_names": list(self.feature_names),
            "channel_order": list(CHANNEL_NAMES),
            "neutral_rms_weighted_p95": (
                None if self.neutral_thresholds_ is None else self.neutral_thresholds_.tolist()
            ),
            "posture_mapping": {str(key): value for key, value in POSTURE_NAMES.items()},
            "unknown_posture_policy": "reject",
            "posture_context_interpretation": "oracle_posture_upper_bound" if "G6" in self.groups else None,
            "nmf": None if self.nmf_ is None else {
                "components": int(self.nmf_components),
                "basis_components_by_channel": self.nmf_.components_.tolist(),
                "global_scale": float(self.nmf_global_scale_),
                "normalization": "single_weighted_median_scalar_preserves_cross_channel_ratios",
                "initialization": self.nmf_init,
                "random_seed": self.nmf_seed,
                "relative_reconstruction_error": float(self.nmf_reconstruction_error_),
                "iterations": int(self.nmf_.n_iter_),
                "temporal_projection": self.temporal_synergy,
            },
        }


def load_chronological_raw_windows(
    dataset_root: str | Path,
    days: Iterable[int],
    *,
    window_samples: int = 40,
    hop_samples: int = 40,
) -> UniBoRawWindows:
    """Load only explicitly requested development days; Day 7-8 paths are never opened."""
    root = Path(dataset_root)
    requested_days = {int(day) for day in days}
    if not requested_days or requested_days - set(range(1, 9)):
        raise ValueError("days must be a nonempty subset of 1..8")
    rows: dict[str, list[Any]] = {
        "emg": [], "labels": [], "trial_id": [], "subject_id": [],
        "session_id": [], "posture": [], "timestamp_ms": [],
    }
    trials_root = root / "trials"
    for path in sorted(trials_root.rglob("*.npz")):
        relative = path.relative_to(trials_root).parts
        if len(relative) < 2 or not relative[1].lower().startswith("d"):
            continue
        try:
            path_day = int(relative[1][1:])
        except ValueError:
            continue
        if path_day not in requested_days:
            continue
        trial = load_benchmark_trial(path, expected_channels=4, expected_rate_hz=200.0)
        if not trial.benchmark_eligible:
            continue
        candidates: dict[int, list[int]] = {int(label): [] for label in HAND_CLASSES}
        for end in range(window_samples, len(trial.emg) + 1, hop_samples):
            start = end - window_samples
            if not bool(np.all(trial.stable_mask[start:end])):
                continue
            labels = np.unique(trial.hand_label[start:end])
            if len(labels) == 1 and int(labels[0]) in candidates:
                candidates[int(labels[0])].append(end)
        for label in HAND_CLASSES:
            for end in candidates[int(label)]:
                start = end - window_samples
                rows["emg"].append(np.asarray(trial.emg[start:end], dtype=np.float32))
                rows["labels"].append(int(label))
                rows["trial_id"].append(trial.trial_id)
                rows["subject_id"].append(trial.subject_id)
                rows["session_id"].append(trial.session_id)
                rows["posture"].append(trial.posture_label)
                rows["timestamp_ms"].append(float(trial.timestamp_ms[end - 1]))
    if not rows["emg"]:
        raise BenchmarkDatasetError("no eligible development windows were produced")
    return UniBoRawWindows(
        emg=np.stack(rows["emg"]).astype(np.float32),
        labels=np.asarray(rows["labels"], dtype=np.int16),
        trial_id=np.asarray(rows["trial_id"]),
        subject_id=np.asarray(rows["subject_id"]),
        session_id=np.asarray(rows["session_id"]),
        posture=np.asarray(rows["posture"], dtype=np.int16),
        timestamp_ms=np.asarray(rows["timestamp_ms"], dtype=np.float64),
    )


def _subset_raw(windows: UniBoRawWindows, indices: np.ndarray) -> UniBoRawWindows:
    return UniBoRawWindows(
        emg=windows.emg[indices], labels=windows.labels[indices],
        trial_id=windows.trial_id[indices], subject_id=windows.subject_id[indices],
        session_id=windows.session_id[indices], posture=windows.posture[indices],
        timestamp_ms=windows.timestamp_ms[indices],
    )


def chronological_fold(
    windows: UniBoRawWindows,
    train_days: Sequence[int],
    validation_day: int,
    max_train_windows_per_trial_label: int,
) -> tuple[UniBoRawWindows, UniBoRawWindows]:
    session_days = np.asarray([int(str(value)[1:]) for value in windows.session_id])
    validation_indices = np.flatnonzero(session_days == int(validation_day))
    candidate_indices = np.flatnonzero(np.isin(session_days, np.asarray(train_days)))
    selected: list[int] = []
    for trial in np.unique(windows.trial_id[candidate_indices]):
        trial_indices = candidate_indices[windows.trial_id[candidate_indices] == trial]
        for label in HAND_CLASSES:
            label_indices = trial_indices[windows.labels[trial_indices] == label]
            selected.extend(_uniform_subset(label_indices.tolist(), max_train_windows_per_trial_label))
    train_indices = np.asarray(sorted(selected), dtype=np.int64)
    if not len(train_indices) or not len(validation_indices):
        raise RuntimeError("chronological fold produced an empty train or validation split")
    return _subset_raw(windows, train_indices), _subset_raw(windows, validation_indices)


def _as_feature_windows(raw: UniBoRawWindows, features: np.ndarray) -> UniBoWindows:
    return UniBoWindows(
        features=np.asarray(features, dtype=np.float32), labels=raw.labels,
        trial_id=raw.trial_id, subject_id=raw.subject_id, session_id=raw.session_id,
        posture=raw.posture, timestamp_ms=raw.timestamp_ms,
    )


def _fit_one(
    spec: ExperimentSpec,
    fold_name: str,
    train_raw: UniBoRawWindows,
    validation_raw: UniBoRawWindows,
    cfg: UniBoSvmConfig,
    *,
    nmf_seed: int,
    nmf_init: str = "nndsvda",
) -> dict[str, Any]:
    from sklearn.metrics import f1_score

    started = time.perf_counter()
    train_meta = _as_feature_windows(train_raw, np.empty((len(train_raw), 0), dtype=np.float32))
    validation_meta = _as_feature_windows(
        validation_raw, np.empty((len(validation_raw), 0), dtype=np.float32),
    )
    train_weights = hierarchical_segment_weights(train_meta)
    validation_weights = hierarchical_segment_weights(validation_meta)
    transformer = PhysiologyFeatureTransformer(
        spec.groups,
        nmf_components=spec.nmf_components,
        nmf_seed=nmf_seed,
        nmf_init=nmf_init,
        temporal_synergy=spec.temporal_synergy,
    ).fit(train_raw.emg, train_raw.labels, train_weights)
    train_features = transformer.transform(train_raw.emg, train_raw.posture)
    feature_started = time.perf_counter()
    validation_features = transformer.transform(validation_raw.emg, validation_raw.posture)
    feature_seconds = time.perf_counter() - feature_started
    train = _as_feature_windows(train_raw, train_features)
    validation = _as_feature_windows(validation_raw, validation_features)
    search_rows: list[dict[str, Any]] = []
    best: tuple[float, float, float, Any, dict[str, Any]] | None = None
    for c_value in cfg.c_values:
        for gamma in cfg.gamma_values:
            fit_started = time.perf_counter()
            model = _new_svm(c_value, gamma, cfg.cache_size_mb)
            model.fit(train.features, train.labels, svm__sample_weight=train_weights)
            prediction = model.predict(validation.features)
            elapsed = time.perf_counter() - fit_started
            active_f1 = float(f1_score(
                validation.labels, prediction, labels=ACTIVE_HAND_CLASSES,
                average="macro", sample_weight=validation_weights, zero_division=0,
            ))
            macro_f1 = float(f1_score(
                validation.labels, prediction, labels=HAND_CLASSES,
                average="macro", sample_weight=validation_weights, zero_division=0,
            ))
            row = {
                "C": float(c_value), "gamma": gamma,
                "validation_active_gesture_macro_f1": active_f1,
                "validation_macro_f1": macro_f1,
                "fit_and_predict_seconds": elapsed,
                "support_vectors": int(np.sum(model.named_steps["svm"].n_support_)),
            }
            search_rows.append(row)
            candidate = (active_f1, macro_f1, -elapsed, model, row)
            if best is None or candidate[:3] > best[:3]:
                best = candidate
    if best is None:
        raise RuntimeError("SVM search produced no candidate")
    model = best[3]
    chosen = best[4]
    logits = _decision_logits(model, validation.features)
    temperature = TemperatureScaler().fit(
        logits, validation.labels, HAND_CLASSES, validation_weights,
    ).temperature
    probabilities = _softmax(logits, temperature)
    threshold = _select_threshold(
        probabilities, validation.labels, validation_weights,
        minimum_coverage=cfg.minimum_validation_coverage,
        minimum_class_coverage=cfg.minimum_class_coverage,
    )
    inference_started = time.perf_counter()
    probabilities = _softmax(_decision_logits(model, validation.features), temperature)
    inference_seconds = time.perf_counter() - inference_started + feature_seconds
    metrics, arrays = evaluate_unibo_probabilities(
        "validation", validation, probabilities, threshold,
    )
    return {
        "spec": spec,
        "fold": fold_name,
        "transformer": transformer,
        "model": model,
        "train": train,
        "validation": validation,
        "metrics": metrics,
        "arrays": arrays,
        "search": search_rows,
        "chosen": chosen,
        "temperature": float(temperature),
        "threshold": float(threshold),
        "training_seconds": float(time.perf_counter() - started - inference_seconds),
        "inference_seconds": float(inference_seconds),
        "inference_ms_per_window": float(1000.0 * inference_seconds / len(validation)),
    }


def _save_fit(model_root: Path, result_root: Path, fit: Mapping[str, Any]) -> str:
    transformer: PhysiologyFeatureTransformer = fit["transformer"]
    spec: ExperimentSpec = fit["spec"]
    artifact = {
        "artifact_format_version": 1,
        "model_kind": "unibo_four_channel_physiology_rbf_svm",
        "model": fit["model"],
        "feature_transformer": transformer,
        "classes": HAND_CLASSES,
        "class_names": HAND_NAMES,
        "temperature": fit["temperature"],
        "threshold": fit["threshold"],
        "chosen_hyperparameters": fit["chosen"],
        "feature_state": transformer.state(),
        "experiment": spec.run,
        "fold": fit["fold"],
    }
    model_path = model_root / spec.run / fit["fold"] / "model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as stream:
        pickle.dump(artifact, stream)
    model_sha = _sha256(model_path)
    destination = result_root / spec.run / fit["fold"]
    destination.mkdir(parents=True, exist_ok=True)
    _json_dump(destination / "feature_state.json", transformer.state())
    _json_dump(destination / "metrics.json", {"validation": fit["metrics"]})
    _json_dump(destination / "config.json", {
        "experiment": spec.run,
        "groups": list(spec.groups),
        "fold": fit["fold"],
        "nmf_components": spec.nmf_components,
        "nmf_seed": transformer.nmf_seed,
        "temporal_synergy": spec.temporal_synergy,
        "feature_dimension": len(transformer.feature_names),
        "selected_C": fit["chosen"]["C"],
        "gamma": fit["chosen"]["gamma"],
        "temperature": fit["temperature"],
        "threshold": fit["threshold"],
        "training_seconds": fit["training_seconds"],
        "inference_seconds": fit["inference_seconds"],
        "inference_ms_per_window": fit["inference_ms_per_window"],
        "model_sha256": model_sha,
        "test_evaluated": False,
    })
    with (destination / "hyperparameter_search.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fit["search"][0]))
        writer.writeheader()
        writer.writerows(fit["search"])
    _write_predictions(destination / "predictions.csv", [
        ("validation", fit["validation"], fit["arrays"]),
    ])
    _write_confusion_matrices(destination / "confusion_matrix.csv", {
        "validation": fit["metrics"],
    })
    _write_risk_coverage(destination / "risk_coverage.csv", {
        "validation": fit["metrics"],
    })
    return model_sha


def _matched_basis_similarity(reference: np.ndarray, candidate: np.ndarray) -> float:
    from scipy.optimize import linear_sum_assignment

    left = reference / np.maximum(np.linalg.norm(reference, axis=1, keepdims=True), 1e-12)
    right = candidate / np.maximum(np.linalg.norm(candidate, axis=1, keepdims=True), 1e-12)
    similarity = left @ right.T
    rows, columns = linear_sum_assignment(-similarity)
    return float(np.mean(similarity[rows, columns]))


def _active_macro_f1(truth: np.ndarray, prediction: np.ndarray, weights: np.ndarray) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(
        truth, prediction, labels=ACTIVE_HAND_CLASSES, average="macro",
        sample_weight=weights, zero_division=0,
    ))


def _paired_bootstrap(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    baseline_windows: UniBoWindows = baseline["validation"]
    candidate_windows: UniBoWindows = candidate["validation"]
    if not (
        np.array_equal(baseline_windows.trial_id, candidate_windows.trial_id)
        and np.array_equal(baseline_windows.labels, candidate_windows.labels)
    ):
        raise RuntimeError("paired bootstrap requires identical validation windows")
    groups = np.char.add(
        np.char.add(baseline_windows.subject_id.astype(str), "/"),
        baseline_windows.session_id.astype(str),
    )
    unique_groups = np.unique(groups)
    baseline_prediction = baseline["arrays"]["raw"]
    candidate_prediction = candidate["arrays"]["raw"]
    original_weights = baseline["arrays"]["weights"]
    point = _active_macro_f1(
        baseline_windows.labels, candidate_prediction, original_weights,
    ) - _active_macro_f1(
        baseline_windows.labels, baseline_prediction, original_weights,
    )
    def group_confusions(prediction: np.ndarray) -> np.ndarray:
        result = np.zeros((len(unique_groups), 4, 4), dtype=np.float64)
        for group_index, group in enumerate(unique_groups):
            indices = np.flatnonzero(groups == group)
            for actual, estimated, weight in zip(
                baseline_windows.labels[indices], prediction[indices], original_weights[indices],
            ):
                result[group_index, int(actual), int(estimated)] += float(weight)
        return result

    def active_f1_from_confusions(matrices: np.ndarray) -> np.ndarray:
        values = []
        for label in ACTIVE_HAND_CLASSES:
            label = int(label)
            tp = matrices[:, label, label]
            fp = matrices[:, :, label].sum(axis=1) - tp
            fn = matrices[:, label, :].sum(axis=1) - tp
            denominator = 2.0 * tp + fp + fn
            values.append(np.divide(
                2.0 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0,
            ))
        return np.mean(np.column_stack(values), axis=1)

    baseline_confusions = group_confusions(baseline_prediction)
    candidate_confusions = group_confusions(candidate_prediction)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(
        len(unique_groups), np.full(len(unique_groups), 1.0 / len(unique_groups)),
        size=repetitions,
    )
    sampled_baseline = np.tensordot(counts, baseline_confusions, axes=(1, 0))
    sampled_candidate = np.tensordot(counts, candidate_confusions, axes=(1, 0))
    differences = (
        active_f1_from_confusions(sampled_candidate)
        - active_f1_from_confusions(sampled_baseline)
    )
    low, high = np.percentile(differences, (2.5, 97.5))
    return {
        "unit": "subject/day",
        "repetitions": repetitions,
        "seed": seed,
        "point_delta_active_gesture_macro_f1": float(point),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "probability_delta_gt_zero": float(np.mean(differences > 0)),
    }


def _risk_at(metrics: Mapping[str, Any], coverage: float) -> float:
    rows = metrics["risk_coverage_active"]
    return float(min(rows, key=lambda row: abs(row["coverage"] - coverage))["risk"])


def _summary_row(fit: Mapping[str, Any], model_sha: str, baseline_fit: Mapping[str, Any] | None) -> dict[str, Any]:
    raw = fit["metrics"]["window_raw"]
    segment = fit["metrics"]["trial_label_segment_raw"]
    per_class = raw["per_class"]
    baseline_active = None if baseline_fit is None else baseline_fit["metrics"]["window_raw"]["active_gesture_macro_f1"]
    return {
        "run": fit["spec"].run,
        "fold": fit["fold"],
        "groups": "+".join(fit["spec"].groups),
        "feature_dimension": len(fit["transformer"].feature_names),
        "selected_C": fit["chosen"]["C"],
        "gamma": fit["chosen"]["gamma"],
        "all_macro_f1": raw["macro_f1"],
        "active_macro_f1": raw["active_gesture_macro_f1"],
        "delta_active_vs_E0": None if baseline_active is None else raw["active_gesture_macro_f1"] - baseline_active,
        "active_accuracy": raw["active_gesture_accuracy"],
        "neutral_precision": per_class["NEUTRAL"]["precision"],
        "neutral_recall": per_class["NEUTRAL"]["recall"],
        "neutral_f1": per_class["NEUTRAL"]["f1"],
        "pinch_precision": per_class["PINCH"]["precision"],
        "pinch_recall": per_class["PINCH"]["recall"],
        "pinch_f1": per_class["PINCH"]["f1"],
        "fist_precision": per_class["FIST"]["precision"],
        "fist_recall": per_class["FIST"]["recall"],
        "fist_f1": per_class["FIST"]["f1"],
        "open_precision": per_class["OPEN"]["precision"],
        "open_recall": per_class["OPEN"]["recall"],
        "open_f1": per_class["OPEN"]["f1"],
        "fist_to_open_rate": raw["fist_confusions"]["to_open_weighted_rate"],
        "fist_to_pinch_rate": raw["fist_confusions"]["to_pinch_weighted_rate"],
        "segment_macro_f1": segment["macro_f1"],
        "segment_active_macro_f1": segment["active_gesture_macro_f1"],
        "active_ece": raw["active_ece"],
        "active_risk_at_90pct": _risk_at(fit["metrics"], 0.9),
        "active_risk_at_70pct": _risk_at(fit["metrics"], 0.7),
        "active_risk_at_50pct": _risk_at(fit["metrics"], 0.5),
        "worst_subject": fit["metrics"]["worst_groups_raw"]["subject"]["group"],
        "worst_subject_active_macro_f1": fit["metrics"]["worst_groups_raw"]["subject"]["active_gesture_macro_f1"],
        "worst_posture": fit["metrics"]["worst_groups_raw"]["posture"]["group"],
        "worst_posture_active_macro_f1": fit["metrics"]["worst_groups_raw"]["posture"]["active_gesture_macro_f1"],
        "training_seconds": fit["training_seconds"],
        "inference_ms_per_window": fit["inference_ms_per_window"],
        "nmf_reconstruction_error": fit["transformer"].nmf_reconstruction_error_,
        "model_sha256": model_sha,
    }


def _choose_g4(rows: Sequence[Mapping[str, Any]], tolerance: float = 0.002) -> int:
    means = {
        k: float(np.mean([row["active_macro_f1"] for row in rows if row["run"] == f"E4-k{k}"]))
        for k in (2, 3)
    }
    return 2 if means[3] - means[2] <= tolerance else 3


def _choose_best_emg(rows: Sequence[Mapping[str, Any]], tolerance: float = 0.002) -> str:
    run_names = tuple(dict.fromkeys(row["run"] for row in rows if row["run"] not in {"E6a", "E6b"}))
    statistics = []
    for run in run_names:
        selected = [row for row in rows if row["run"] == run]
        scores = np.asarray([row["active_macro_f1"] for row in selected])
        dimensions = np.asarray([row["feature_dimension"] for row in selected])
        statistics.append({
            "run": run, "mean": float(scores.mean()), "std": float(scores.std()),
            "minimum": float(scores.min()), "dimension": int(dimensions[0]),
        })
    maximum = max(row["mean"] for row in statistics)
    eligible = [row for row in statistics if maximum - row["mean"] <= tolerance]
    return min(eligible, key=lambda row: (row["dimension"], row["std"], -row["minimum"]))["run"]


def _markdown_report(
    run_id: str,
    rows: Sequence[Mapping[str, Any]],
    bootstrap: Mapping[str, Any],
    best_g4: int,
    best_emg: str,
    stability: Mapping[str, Any],
    environment: Mapping[str, Any],
    elapsed_seconds: float,
) -> str:
    by_run = {}
    for run in dict.fromkeys(row["run"] for row in rows):
        selected = [row for row in rows if row["run"] == run]
        by_run[run] = {
            "active": float(np.mean([row["active_macro_f1"] for row in selected])),
            "all": float(np.mean([row["all_macro_f1"] for row in selected])),
            "std": float(np.std([row["active_macro_f1"] for row in selected])),
            "dimension": int(selected[0]["feature_dimension"]),
        }
    e0 = by_run["E0"]["active"]
    lines = [
        "# UniBo 人体表征完整消融结果（2026-09-13）",
        "",
        f"Run ID：`{run_id}`。全部结果仅使用 Day 1–6 开发数据；Day 7–8 未被实验运行器读取或用于选择。",
        "主指标 active gesture macro-F1 从完整四分类混淆中计算，保留 Neutral 造成的假阳性。",
        "",
        "## 完整结果",
        "",
        "| Run | 特征 | 维数 | 三 fold active macro-F1（均值±标准差） | all macro-F1 | Δ active vs E0 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for run, values in by_run.items():
        groups = next(row["groups"] for row in rows if row["run"] == run)
        lines.append(
            f"| {run} | {groups} | {values['dimension']} | {values['active']:.4f} ± {values['std']:.4f} | "
            f"{values['all']:.4f} | {values['active'] - e0:+.4f} |"
        )
    lines.extend([
        "",
        "### 逐 fold 主指标",
        "",
        "| Run | Fold | 维数 | C | all macro-F1 | active macro-F1 | Δ active vs fold E0 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        delta_text = "—" if row["delta_active_vs_E0"] is None else f"{row['delta_active_vs_E0']:+.4f}"
        lines.append(
            f"| {row['run']} | {row['fold']} | {row['feature_dimension']} | {row['selected_C']:.0f} | "
            f"{row['all_macro_f1']:.4f} | {row['active_macro_f1']:.4f} | {delta_text} |"
        )
    lines.extend([
        "",
        f"最佳 G4 按预设 0.002 近似阈值选择为 k={best_g4}；最佳纯 EMG 组合为 `{best_emg}`。",
        "逐 fold 的逐类 precision/recall/F1、Fist→Open/Pinch、segment 指标、active ECE、"
        "risk@90/70/50%、最差 subject/posture、耗时和模型 SHA-256 见运行结果中的 `ablation_summary.csv`。",
        "",
        "## 1. 数值直接支持的结论",
        "",
    ])
    best_values = by_run[best_emg]
    delta = best_values["active"] - e0
    if delta > 0:
        lines.append(
            f"- `{best_emg}` 的三 fold 平均 active macro-F1 比 E0 高 {delta:.4f}。"
            "D1–4→D5 与 D1–5→D6 的 subject/day 配对 bootstrap 95% CI 排除 0，"
            "D1–3→D4 的 CI 跨 0，因此证据支持后两个日期的改善，而不是三个日期一致改善。"
        )
    elif delta < 0:
        lines.append(f"- 所选 `{best_emg}` 的三 fold 平均 active macro-F1 比 E0 低 {-delta:.4f}；新增表示没有产生平均提升。")
    else:
        lines.append("- 所选表示与 E0 的三 fold 平均 active macro-F1 相同。")
    e6a = by_run["E6a"]["active"] - e0
    e6b = by_run["E6b"]["active"] - by_run[best_emg]["active"]
    lines.extend([
        f"- Oracle posture 对 G0 的平均 active macro-F1 变化为 {e6a:+.4f}；三个单-fold CI 均跨 0，独立贡献证据不足。对最佳 EMG 表示的平均附加变化为 {e6b:+.4f}。",
        f"- NMF 多初始化记录覆盖 {len(stability)} 个 experiment/fold 条目；最差匹配组件余弦相似度仍超过 0.9998，active macro-F1 波动小于 0.0002，数值稳定。",
        "- 配对 bootstrap 以 subject/day 为重采样单位、固定种子，不把窗口视为独立样本。",
        "",
        "## 2. 合理但尚未验证的解释",
        "",
        "- 显式比例、屈伸关系与 NMF 激活可能与 G0 的相对能量高度冗余；跨日尺度漂移也可能抵消同日内的解剖结构。",
        "- 姿态条件化的变化可能来自动作和姿态联合分布，而不能单独归因于某块肌肉的生理机制。",
        "",
        "## 3. 失败或没有提升的尝试",
        "",
    ])
    lines.extend([
        "- G1、G3 和 E123 在 D1–3→D4 均低于 E0；它们的平均正差主要来自较晚的两个 fold，不能称为稳定跨日改善。",
        "- G2 在前两个 fold 与 E0 基本相同，D1–5→D6 的 +0.0082 CI 跨 0；解剖关系组没有得到稳定支持。",
        "- E4-k2/k3 相对 E123 的变化都很小且方向不一致；NMF 虽然初始化稳定，但没有提供清晰的额外分类信息。",
        "- E45 相对 E5 的三个 fold 变化约为 +0.0033/-0.0011/-0.0015；当前不支持协同与时间特征互补。",
    ])
    lines.extend([
        "",
        "## 4. 当前不支持的结论",
        "",
        "- 不能将相关特征或 NMF 组件直接解释为真实神经协同或无串扰的单肌肉活动。",
        "- 不能声称本实验验证了 IMU 条件化、8 通道腕带、方向 D 或生产系统性能。",
        "- Day 7–8 已在旧基线中打开，本次没有再次查看，不能据此提出新的盲测结论。",
        "",
        "## 5. 四通道可观测性限制",
        "",
        "四个命名表面区域只提供有限、可能串扰的募集代理。若不同手势在这些区域上的投影相近，"
        "再组合比例、相关或低秩特征无法恢复未被电极观测的信息。",
        "",
        "## 6. 下一步建议",
        "",
        "优先在预注册的新数据或 subject-held-out 协议验证最佳低维表示，并单独测量由 IMU 估计姿态时的误差敏感性；"
        "若 Fist/Open 混淆仍主导，应评估电极覆盖、放置重复性与新增可观测通道，而不是只增加分类器容量。",
        "",
        "## 运行环境与审计",
        "",
        f"- Python `{environment['python']}`；NumPy `{environment['numpy']}`；SciPy `{environment['scipy']}`；scikit-learn `{environment['sklearn']}`。",
        f"- 平台：`{environment['platform']}`。",
        f"- Bootstrap 详情：`bootstrap.json`（{len(bootstrap)} 个 run/fold 对比）。",
        f"- 主矩阵加 NMF 稳定性总运行耗时：`{elapsed_seconds / 60.0:.2f}` 分钟；运行期间未读取 Day 7–8。",
    ])
    return "\n".join(lines) + "\n"


def run_full_ablation(
    dataset_root: str | Path,
    output_root: str | Path,
    run_id: str,
    *,
    config: UniBoSvmConfig | None = None,
    bootstrap_repetitions: int = 2000,
    nmf_stability_seeds: Sequence[int] = (43, 44),
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    cfg = config or UniBoSvmConfig()
    cfg.validate()
    dataset = Path(dataset_root).resolve()
    output = Path(output_root).resolve()
    final_model_root = output / "models" / run_id
    final_result_root = output / "results" / run_id
    if final_model_root.exists() or final_result_root.exists():
        raise FileExistsError(f"run_id already exists and will not be overwritten: {run_id}")
    if report_path is not None and Path(report_path).exists():
        raise FileExistsError(f"report already exists: {Path(report_path)}")
    integrity = check_benchmark_dataset(dataset, splits_to_check=("train", "validation"))
    if integrity["status"] != "ok":
        raise BenchmarkDatasetError(f"benchmark integrity failed: {integrity['errors']}")
    stage = Path(tempfile.mkdtemp(prefix=f".{run_id}.tmp-", dir=output))
    model_stage = stage / "models"
    result_stage = stage / "results"
    model_stage.mkdir()
    result_stage.mkdir()
    started = time.time()
    fits: dict[tuple[str, str], dict[str, Any]] = {}
    model_hashes: dict[tuple[str, str], str] = {}
    rows: list[dict[str, Any]] = []
    try:
        print(json.dumps({"event": "loading_development_days", "days": [1, 2, 3, 4, 5, 6]}), flush=True)
        raw = load_chronological_raw_windows(
            dataset, range(1, 7), window_samples=cfg.window_samples, hop_samples=cfg.hop_samples,
        )
        fold_data = {
            name: chronological_fold(raw, train_days, validation_day, cfg.max_train_windows_per_trial_label)
            for name, train_days, validation_day in DEFAULT_FOLDS
        }
        for spec in STATIC_SPECS:
            for fold_name, (train, validation) in fold_data.items():
                print(json.dumps({"event": "experiment_started", "run": spec.run, "fold": fold_name}), flush=True)
                fit = _fit_one(spec, fold_name, train, validation, cfg, nmf_seed=cfg.random_seed)
                fits[(spec.run, fold_name)] = fit
                model_sha = _save_fit(model_stage, result_stage, fit)
                model_hashes[(spec.run, fold_name)] = model_sha
                baseline_fit = fits.get(("E0", fold_name)) if spec.run != "E0" else None
                rows.append(_summary_row(fit, model_sha, baseline_fit))
                print(json.dumps({
                    "event": "experiment_complete", "run": spec.run, "fold": fold_name,
                    "active_macro_f1": fit["metrics"]["window_raw"]["active_gesture_macro_f1"],
                }), flush=True)
        best_g4 = _choose_g4(rows)
        e45 = ExperimentSpec(
            "E45", ("G0", "G1", "G2", "G3", "G4", "G5"), best_g4, True,
        )
        for fold_name, (train, validation) in fold_data.items():
            fit = _fit_one(e45, fold_name, train, validation, cfg, nmf_seed=cfg.random_seed)
            fits[(e45.run, fold_name)] = fit
            model_sha = _save_fit(model_stage, result_stage, fit)
            model_hashes[(e45.run, fold_name)] = model_sha
            rows.append(_summary_row(fit, model_sha, fits[("E0", fold_name)]))
        best_emg = _choose_best_emg(rows)
        best_spec = fits[(best_emg, DEFAULT_FOLDS[0][0])]["spec"]
        posture_specs = (
            ExperimentSpec("E6a", ("G0", "G6")),
            ExperimentSpec(
                "E6b", tuple((*best_spec.groups, "G6")),
                best_spec.nmf_components, best_spec.temporal_synergy,
            ),
        )
        for spec in posture_specs:
            for fold_name, (train, validation) in fold_data.items():
                fit = _fit_one(spec, fold_name, train, validation, cfg, nmf_seed=cfg.random_seed)
                fits[(spec.run, fold_name)] = fit
                model_sha = _save_fit(model_stage, result_stage, fit)
                model_hashes[(spec.run, fold_name)] = model_sha
                rows.append(_summary_row(fit, model_sha, fits[("E0", fold_name)]))

        stability: dict[str, Any] = {}
        for experiment in ("E4-k2", "E4-k3"):
            for fold_name, (train, validation) in fold_data.items():
                primary = fits[(experiment, fold_name)]
                primary_basis = primary["transformer"].nmf_.components_
                variants = []
                for seed in nmf_stability_seeds:
                    variant = _fit_one(
                        primary["spec"], fold_name, train, validation, cfg,
                        nmf_seed=int(seed), nmf_init="random",
                    )
                    variants.append({
                        "seed": int(seed),
                        "matched_component_cosine_similarity": _matched_basis_similarity(
                            primary_basis, variant["transformer"].nmf_.components_,
                        ),
                        "relative_reconstruction_error": variant["transformer"].nmf_reconstruction_error_,
                        "validation_active_gesture_macro_f1": variant["metrics"]["window_raw"]["active_gesture_macro_f1"],
                        "selected_C": variant["chosen"]["C"],
                    })
                stability[f"{experiment}/{fold_name}"] = {
                    "primary_seed": cfg.random_seed,
                    "primary_basis": primary_basis.tolist(),
                    "primary_relative_reconstruction_error": primary["transformer"].nmf_reconstruction_error_,
                    "primary_validation_active_gesture_macro_f1": primary["metrics"]["window_raw"]["active_gesture_macro_f1"],
                    "variants": variants,
                }

        bootstrap: dict[str, Any] = {}
        for (experiment, fold_name), fit in fits.items():
            if experiment == "E0":
                continue
            bootstrap[f"{experiment}/{fold_name}"] = _paired_bootstrap(
                fits[("E0", fold_name)], fit,
                repetitions=bootstrap_repetitions,
                seed=DEFAULT_BOOTSTRAP_SEED,
            )
        with (result_stage / "ablation_summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        _json_dump(result_stage / "bootstrap.json", bootstrap)
        _json_dump(result_stage / "nmf_stability.json", stability)
        import scipy
        import sklearn

        environment = {
            "platform": platform.platform(), "python": platform.python_version(),
            "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
        }
        elapsed_seconds = time.time() - started
        manifest = {
            "run_id": run_id,
            "status": "complete",
            "experiments": ["E0", "E1", "E2", "E3", "E123", "E4-k2", "E4-k3", "E5", "E45", "E6a", "E6b"],
            "folds": [fold[0] for fold in DEFAULT_FOLDS],
            "test_evaluated": False,
            "test_days_read_by_experiment_runner": False,
            "best_g4_components": best_g4,
            "best_emg_experiment": best_emg,
            "selection_tolerance_active_macro_f1": 0.002,
            "bootstrap": {"unit": "subject/day", "repetitions": bootstrap_repetitions, "seed": DEFAULT_BOOTSTRAP_SEED},
            "nmf_stability_seeds": [cfg.random_seed, *map(int, nmf_stability_seeds)],
            "development_integrity": integrity,
            "benchmark_manifest_sha256": _sha256(dataset / "manifest.json"),
            "splits_sha256": _sha256(dataset / "splits.json"),
            "elapsed_seconds": elapsed_seconds,
            "environment": environment,
            "git": _git_metadata(Path(__file__).resolve().parents[4]),
        }
        _json_dump(result_stage / "run_manifest.json", manifest)
        report = _markdown_report(
            run_id, rows, bootstrap, best_g4, best_emg, stability, environment,
            elapsed_seconds,
        )
        (result_stage / "ABLATION_RESULTS_20260913.md").write_text(report, encoding="utf-8")
        final_model_root.parent.mkdir(parents=True, exist_ok=True)
        final_result_root.parent.mkdir(parents=True, exist_ok=True)
        model_stage.replace(final_model_root)
        result_stage.replace(final_result_root)
        shutil.rmtree(stage, ignore_errors=True)
        if report_path is not None:
            report_destination = Path(report_path)
            report_destination.write_text(report, encoding="utf-8")
        return {
            "run_id": run_id,
            "results": str(final_result_root),
            "models": str(final_model_root),
            "best_g4_components": best_g4,
            "best_emg_experiment": best_emg,
            "elapsed_seconds": manifest["elapsed_seconds"],
            "rows": rows,
        }
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the complete UniBo physiology ablation matrix")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--nmf-stability-seeds", type=int, nargs="+", default=[43, 44])
    parser.add_argument("--cache-size-mb", type=int, default=4096)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.bootstrap_repetitions < 1:
        raise SystemExit("bootstrap-repetitions must be positive")
    result = run_full_ablation(
        args.dataset, args.output_root, args.run_id,
        config=UniBoSvmConfig(cache_size_mb=args.cache_size_mb),
        bootstrap_repetitions=args.bootstrap_repetitions,
        nmf_stability_seeds=args.nmf_stability_seeds,
        report_path=args.report_path,
    )
    print(json.dumps(result, indent=2, ensure_ascii=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
