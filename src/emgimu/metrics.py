from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    direction_macro_f1: float
    gesture_macro_f1: float
    joint_accuracy: float
    direction_coverage: float
    gesture_coverage: float
    direction_ece: float
    gesture_ece: float
    p50_latency_ms: float | None
    p95_latency_ms: float | None
    direction_labels: tuple[int, ...]
    gesture_labels: tuple[int, ...]
    direction_confusion_matrix: tuple[tuple[int, ...], ...]
    gesture_confusion_matrix: tuple[tuple[int, ...], ...]
    direction_class_coverage: dict[int, float]
    gesture_class_coverage: dict[int, float]
    direction_risk_coverage: tuple[tuple[float, float, float], ...]
    gesture_risk_coverage: tuple[tuple[float, float, float], ...]

    @property
    def meets_v1_target(self) -> bool:
        latency_ok = self.p95_latency_ms is not None and self.p95_latency_ms <= 300.0
        return (
            self.direction_macro_f1 >= 0.85
            and self.gesture_macro_f1 >= 0.85
            and self.joint_accuracy >= 0.75
            and latency_ok
        )


def expected_calibration_error(
    truth: np.ndarray,
    predicted: np.ndarray,
    confidence: np.ndarray,
    *,
    bins: int = 10,
    sample_weight: np.ndarray | None = None,
) -> float:
    truth = np.asarray(truth)
    predicted = np.asarray(predicted)
    confidence = np.asarray(confidence, dtype=np.float64)
    if len(truth) == 0:
        return float("nan")
    weights = _metric_weights(len(truth), sample_weight)
    total_weight = float(weights.sum())
    result = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        if index == bins - 1:
            mask = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            mask = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if not np.any(mask):
            continue
        accuracy = np.average(truth[mask] == predicted[mask], weights=weights[mask])
        mean_confidence = np.average(confidence[mask], weights=weights[mask])
        result += float(weights[mask].sum() / total_weight) * abs(
            float(accuracy) - float(mean_confidence)
        )
    return result


def _metric_weights(length: int, sample_weight: np.ndarray | None) -> np.ndarray:
    weights = (
        np.ones(length, dtype=np.float64)
        if sample_weight is None else np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    )
    if len(weights) != length or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("sample weights must be finite, positive, and match predictions")
    return weights


def per_class_coverage(
    truth: np.ndarray,
    predicted: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> dict[int, float]:
    truth = np.asarray(truth)
    predicted = np.asarray(predicted)
    weights = _metric_weights(len(truth), sample_weight)
    return {
        int(label): float(np.average(
            predicted[truth == label] >= 0, weights=weights[truth == label],
        ))
        for label in np.unique(truth)
    }


def risk_coverage_curve(
    truth: np.ndarray,
    predicted: np.ndarray,
    confidence: np.ndarray,
    *,
    coverage_points: tuple[float, ...] = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5),
    sample_weight: np.ndarray | None = None,
) -> tuple[tuple[float, float, float], ...]:
    """Return (threshold, realized coverage, accepted error risk) points."""
    truth = np.asarray(truth)
    predicted = np.asarray(predicted)
    confidence = np.asarray(confidence, dtype=np.float64)
    if not len(truth):
        return ()
    weights = _metric_weights(len(truth), sample_weight)
    total_weight = float(weights.sum())
    eligible = np.flatnonzero(predicted >= 0)
    if not len(eligible):
        return ()
    order = eligible[np.argsort(confidence[eligible])[::-1]]
    cumulative_weight = np.cumsum(weights[order])
    maximum_coverage = float(cumulative_weight[-1] / total_weight)
    rows: list[tuple[float, float, float]] = []
    targets = sorted({min(maximum_coverage, point) for point in coverage_points}, reverse=True)
    for target in targets:
        target_weight = target * total_weight
        count = int(np.searchsorted(cumulative_weight, target_weight, side="left")) + 1
        count = max(1, min(len(order), count))
        selected = order[:count]
        threshold = float(confidence[selected[-1]])
        coverage = float(weights[selected].sum() / total_weight)
        risk = float(np.average(
            predicted[selected] != truth[selected], weights=weights[selected],
        ))
        rows.append((threshold, coverage, risk))
    return tuple(rows)


def evaluate_predictions(
    direction_truth: np.ndarray,
    gesture_truth: np.ndarray,
    direction_predicted: np.ndarray,
    gesture_predicted: np.ndarray,
    q_direction: np.ndarray,
    q_gesture: np.ndarray,
    *,
    latency_ms: np.ndarray | None = None,
    sample_weight: np.ndarray | None = None,
) -> EvaluationResult:
    d_true = np.asarray(direction_truth)
    h_true = np.asarray(gesture_truth)
    d_pred = np.asarray(direction_predicted)
    h_pred = np.asarray(gesture_predicted)
    if not (len(d_true) == len(h_true) == len(d_pred) == len(h_pred)):
        raise ValueError("prediction arrays must have equal lengths")
    weights = _metric_weights(len(d_true), sample_weight)
    p50 = None
    p95 = None
    if latency_ms is not None and len(latency_ms):
        latency_values = np.asarray(latency_ms, dtype=np.float64)
        p50 = float(np.percentile(latency_values, 50))
        p95 = float(np.percentile(latency_values, 95))
    d_labels = tuple(map(int, sorted(set(d_true.tolist()) | {-1})))
    h_labels = tuple(map(int, sorted(set(h_true.tolist()) | {-1})))
    def confusion(truth: np.ndarray, predicted: np.ndarray, labels: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
        positions = {label: index for index, label in enumerate(labels)}
        matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
        for actual, estimate in zip(truth, predicted):
            if int(actual) in positions and int(estimate) in positions:
                matrix[positions[int(actual)], positions[int(estimate)]] += 1
        return tuple(tuple(map(int, row)) for row in matrix)
    def macro_f1(truth: np.ndarray, predicted: np.ndarray) -> float:
        values = []
        for label in np.unique(truth):
            tp = np.sum(weights[(truth == label) & (predicted == label)])
            fp = np.sum(weights[(truth != label) & (predicted == label)])
            fn = np.sum(weights[(truth == label) & (predicted != label)])
            denominator = 2 * tp + fp + fn
            values.append(0.0 if denominator == 0 else float(2 * tp / denominator))
        return float(np.mean(values)) if values else float("nan")
    return EvaluationResult(
        direction_macro_f1=macro_f1(d_true, d_pred),
        gesture_macro_f1=macro_f1(h_true, h_pred),
        joint_accuracy=float(np.average(
            (d_true == d_pred) & (h_true == h_pred), weights=weights,
        )),
        direction_coverage=float(np.average(d_pred >= 0, weights=weights)),
        gesture_coverage=float(np.average(h_pred >= 0, weights=weights)),
        direction_ece=expected_calibration_error(
            d_true, d_pred, q_direction, sample_weight=weights,
        ),
        gesture_ece=expected_calibration_error(
            h_true, h_pred, q_gesture, sample_weight=weights,
        ),
        p50_latency_ms=p50,
        p95_latency_ms=p95,
        direction_labels=d_labels,
        gesture_labels=h_labels,
        direction_confusion_matrix=confusion(d_true, d_pred, d_labels),
        gesture_confusion_matrix=confusion(h_true, h_pred, h_labels),
        direction_class_coverage=per_class_coverage(d_true, d_pred, weights),
        gesture_class_coverage=per_class_coverage(h_true, h_pred, weights),
        direction_risk_coverage=risk_coverage_curve(
            d_true, d_pred, q_direction, sample_weight=weights,
        ),
        gesture_risk_coverage=risk_coverage_curve(
            h_true, h_pred, q_gesture, sample_weight=weights,
        ),
    )


def first_stable_latency(
    onset_ms: float,
    state_timestamps_ms: np.ndarray,
    predicted_labels: np.ndarray,
    target_label: int,
    *,
    maximum_ms: float = 1000.0,
) -> float | None:
    timestamps = np.asarray(state_timestamps_ms, dtype=np.float64)
    labels = np.asarray(predicted_labels)
    mask = (timestamps >= onset_ms) & (timestamps <= onset_ms + maximum_ms) & (labels == target_label)
    matches = np.flatnonzero(mask)
    return None if not len(matches) else float(timestamps[matches[0]] - onset_ms)
