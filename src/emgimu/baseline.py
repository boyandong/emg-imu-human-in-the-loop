from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .signal import extract_emg_features, extract_imu_features
from .state import Direction, Gesture


def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float64) / max(float(temperature), 1e-4)
    if scaled.ndim == 1:
        scaled = np.column_stack([-scaled, scaled])
    scaled -= scaled.max(axis=1, keepdims=True)
    values = np.exp(scaled)
    return values / values.sum(axis=1, keepdims=True)


def _decision_logits(model: Any, values: np.ndarray) -> np.ndarray:
    logits = np.asarray(model.decision_function(values))
    if logits.ndim == 1:
        logits = np.column_stack([-logits, logits])
    return logits


def _validated_weights(length: int, sample_weight: np.ndarray | None) -> np.ndarray:
    weights = (
        np.ones(length, dtype=np.float64)
        if sample_weight is None else np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    )
    if len(weights) != length or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("sample weights must be finite, positive, and match the samples")
    return weights


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    ordered_values = np.asarray(values)[order]
    ordered_weights = np.asarray(weights, dtype=np.float64)[order]
    cutoff = 0.5 * float(ordered_weights.sum())
    return float(ordered_values[np.searchsorted(np.cumsum(ordered_weights), cutoff, side="left")])


def _macro_f1(
    truth: np.ndarray,
    predicted: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> float:
    """Macro F1 over real truth classes; rejected (-1) predictions count as errors."""
    truth = np.asarray(truth)
    predicted = np.asarray(predicted)
    weights = _validated_weights(len(truth), sample_weight)
    scores: list[float] = []
    for label in np.unique(truth):
        true_positive = np.sum(weights[(truth == label) & (predicted == label)])
        false_positive = np.sum(weights[(truth != label) & (predicted == label)])
        false_negative = np.sum(weights[(truth == label) & (predicted != label)])
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else float(2 * true_positive / denominator))
    return float(np.mean(scores)) if scores else float("nan")


@dataclass(slots=True)
class TemperatureScaler:
    temperature: float = 1.0

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        classes: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "TemperatureScaler":
        labels = np.asarray(labels)
        weights = _validated_weights(len(labels), sample_weight)
        class_to_index = {int(value): index for index, value in enumerate(classes)}
        indices = np.array([class_to_index[int(value)] for value in labels], dtype=int)
        best = (float("inf"), 1.0)
        for temperature in np.geomspace(0.25, 4.0, 81):
            probabilities = _softmax(logits, temperature)
            losses = -np.log(np.clip(probabilities[np.arange(len(indices)), indices], 1e-9, 1.0))
            nll = np.average(losses, weights=weights)
            if nll < best[0]:
                best = (float(nll), float(temperature))
        self.temperature = best[1]
        return self


@dataclass(frozen=True, slots=True)
class HeadPrediction:
    direction: Direction
    gesture: Gesture
    q_direction: float
    q_gesture: float
    direction_margin: float = 0.0
    gesture_margin: float = 0.0
    direction_drift: float = 0.0
    gesture_drift: float = 0.0
    direction_confidence_rejected: bool = False
    gesture_confidence_rejected: bool = False
    direction_drift_rejected: bool = False
    gesture_drift_rejected: bool = False
    direction_disagreement_rejected: bool = False
    gesture_disagreement_rejected: bool = False


@dataclass(slots=True)
class RobustDomainGuard:
    """Class-conditional robust feature envelope used for OOD rejection."""

    centers: dict[int, np.ndarray]
    scales: dict[int, np.ndarray]

    @classmethod
    def fit(
        cls,
        values: np.ndarray,
        labels: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "RobustDomainGuard":
        x = np.asarray(values, dtype=np.float64)
        y = np.asarray(labels)
        weights = _validated_weights(len(x), sample_weight)
        centers: dict[int, np.ndarray] = {}
        scales: dict[int, np.ndarray] = {}
        for label in np.unique(y):
            rows = x[y == label]
            row_weights = weights[y == label]
            center = np.asarray([
                _weighted_median(rows[:, index], row_weights) for index in range(rows.shape[1])
            ])
            deviations = np.abs(rows - center)
            mad = np.asarray([
                _weighted_median(deviations[:, index], row_weights)
                for index in range(rows.shape[1])
            ]) * 1.4826
            fallback = np.sqrt(np.average((rows - center) ** 2, axis=0, weights=row_weights))
            centers[int(label)] = center
            scales[int(label)] = np.maximum(mad, np.maximum(fallback * 0.1, 1e-6))
        return cls(centers, scales)

    def score(self, values: np.ndarray, label: int) -> float:
        if int(label) not in self.centers:
            return 1.0
        z = np.abs(np.asarray(values, dtype=np.float64).reshape(-1) - self.centers[int(label)])
        z /= self.scales[int(label)]
        # Median aggregation is resistant to one noisy feature while still
        # detecting broad cross-session shifts.  The exponential maps to [0,1].
        return float(1.0 - np.exp(-np.median(z) / 4.0))


class BaselinePredictor:
    """Independent RBF-SVM heads with validation-only calibration/rejection."""

    def __init__(self) -> None:
        self.direction_model: Any | None = None
        self.gesture_model: Any | None = None
        self.direction_temperature = TemperatureScaler()
        self.gesture_temperature = TemperatureScaler()
        self.direction_threshold = 0.0
        self.gesture_threshold = 0.0
        self.direction_domain_guard: RobustDomainGuard | None = None
        self.gesture_domain_guard: RobustDomainGuard | None = None
        self.direction_drift_threshold = 1.0
        self.gesture_drift_threshold = 1.0
        self.direction_logit_bias: np.ndarray | None = None
        self.gesture_logit_bias: np.ndarray | None = None
        self.metadata: dict[str, Any] = {
            "artifact_format_version": 2,
            "model_kind": "independent_rbf_svm",
            "feature_schema": "emg48_imu36_v1",
            "window_ms": 200,
            "hop_ms": 40,
            "sample_rate_hz": 200,
            "test_session_opened": False,
        }

    @staticmethod
    def _new_svm() -> Any:
        try:
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.svm import SVC
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("scikit-learn is required to train the baseline") from exc
        return Pipeline([
            ("scale", StandardScaler()),
            ("svm", SVC(C=10.0, gamma="scale", kernel="rbf", class_weight="balanced")),
        ])

    @staticmethod
    def _select_threshold(
        probabilities: np.ndarray,
        predicted_indices: np.ndarray,
        predicted_classes: np.ndarray,
        truth: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
        min_coverage: float = 0.90,
        min_class_coverage: float = 0.75,
    ) -> float:
        confidence = probabilities[np.arange(len(probabilities)), predicted_indices]
        weights = _validated_weights(len(truth), sample_weight)
        best = (-1.0, -float("inf"), 0.0)
        for threshold in np.linspace(0.0, 0.95, 96):
            accepted = confidence >= threshold
            coverage = float(np.average(accepted, weights=weights))
            if coverage < min_coverage:
                continue
            class_coverage = [
                float(np.average(accepted[truth == label], weights=weights[truth == label]))
                for label in np.unique(truth)
            ]
            if class_coverage and min(class_coverage) < min_class_coverage:
                continue
            output = np.where(accepted, predicted_classes, -1)
            score = _macro_f1(truth, output, weights)
            risk = float(np.average(
                predicted_classes[accepted] != truth[accepted], weights=weights[accepted],
            ))
            candidate = (score, -risk, float(threshold))
            if candidate > best:
                best = candidate
        return best[2]

    @staticmethod
    def _select_drift_threshold(
        scores: np.ndarray,
        predicted: np.ndarray,
        truth: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
        min_coverage: float = 0.90,
        min_class_coverage: float = 0.75,
    ) -> float:
        scores = np.asarray(scores, dtype=np.float64)
        predicted = np.asarray(predicted)
        truth = np.asarray(truth)
        weights = _validated_weights(len(truth), sample_weight)
        if not len(scores):
            return 1.0
        best = (float("inf"), -1.0, 1.0)
        for threshold in np.unique(np.r_[scores, 1.0]):
            accepted = scores <= threshold
            coverage = float(np.average(accepted, weights=weights))
            if coverage < min_coverage or not np.any(accepted):
                continue
            class_coverage = [
                float(np.average(accepted[truth == label], weights=weights[truth == label]))
                for label in np.unique(truth)
            ]
            if class_coverage and min(class_coverage) < min_class_coverage:
                continue
            risk = float(np.average(
                predicted[accepted] != truth[accepted], weights=weights[accepted],
            ))
            candidate = (risk, -coverage, float(threshold))
            if candidate < best:
                best = candidate
        return float(best[2])

    def fit(
        self,
        train_emg_features: np.ndarray,
        train_imu_features: np.ndarray,
        train_direction: np.ndarray,
        train_gesture: np.ndarray,
        *,
        validation_emg_features: np.ndarray,
        validation_imu_features: np.ndarray,
        validation_direction: np.ndarray,
        validation_gesture: np.ndarray,
        train_sample_weight: np.ndarray | None = None,
        validation_sample_weight: np.ndarray | None = None,
    ) -> "BaselinePredictor":
        train_weights = _validated_weights(len(train_direction), train_sample_weight)
        validation_weights = _validated_weights(len(validation_direction), validation_sample_weight)
        self.metadata["window_weighting"] = (
            "caller_supplied_v1" if train_sample_weight is not None else "uniform_windows_v1"
        )
        self.direction_model = self._new_svm().fit(
            train_imu_features, train_direction, svm__sample_weight=train_weights,
        )
        self.gesture_model = self._new_svm().fit(
            train_emg_features, train_gesture, svm__sample_weight=train_weights,
        )
        self.direction_domain_guard = RobustDomainGuard.fit(
            train_imu_features, train_direction, train_weights,
        )
        self.gesture_domain_guard = RobustDomainGuard.fit(
            train_emg_features, train_gesture, train_weights,
        )

        d_logits = _decision_logits(self.direction_model, validation_imu_features)
        h_logits = _decision_logits(self.gesture_model, validation_emg_features)
        d_classes = np.asarray(self.direction_model.classes_)
        h_classes = np.asarray(self.gesture_model.classes_)
        self.direction_temperature.fit(
            d_logits, validation_direction, d_classes, validation_weights,
        )
        self.gesture_temperature.fit(
            h_logits, validation_gesture, h_classes, validation_weights,
        )
        d_prob = _softmax(d_logits, self.direction_temperature.temperature)
        h_prob = _softmax(h_logits, self.gesture_temperature.temperature)
        d_index = d_prob.argmax(axis=1)
        h_index = h_prob.argmax(axis=1)
        self.direction_threshold = self._select_threshold(
            d_prob, d_index, d_classes[d_index], validation_direction,
            sample_weight=validation_weights,
        )
        self.gesture_threshold = self._select_threshold(
            h_prob, h_index, h_classes[h_index], validation_gesture,
            sample_weight=validation_weights,
        )
        d_predicted = d_classes[d_index]
        h_predicted = h_classes[h_index]
        d_drift = np.asarray([
            self.direction_domain_guard.score(row, int(label))
            for row, label in zip(validation_imu_features, d_predicted)
        ])
        h_drift = np.asarray([
            self.gesture_domain_guard.score(row, int(label))
            for row, label in zip(validation_emg_features, h_predicted)
        ])
        self.direction_drift_threshold = self._select_drift_threshold(
            d_drift, d_predicted, validation_direction,
            sample_weight=validation_weights,
        )
        self.gesture_drift_threshold = self._select_drift_threshold(
            h_drift, h_predicted, validation_gesture,
            sample_weight=validation_weights,
        )
        self.metadata["domain_guard"] = {
            "method": "class_conditional_robust_median_z",
            "direction_threshold": self.direction_drift_threshold,
            "gesture_threshold": self.gesture_drift_threshold,
            "minimum_validation_coverage": 0.90,
            "minimum_per_class_validation_coverage": 0.75,
        }
        self.metadata["classes"] = {
            "direction": [int(value) for value in self.direction_model.classes_],
            "gesture": [int(value) for value in self.gesture_model.classes_],
        }
        self.reset_session_adapter()
        return self

    @staticmethod
    def _fit_bounded_bias(
        logits: np.ndarray,
        labels: np.ndarray,
        classes: np.ndarray,
        *,
        max_abs_bias: float,
        min_samples_per_class: int,
        steps: int,
        learning_rate: float,
        l2: float,
    ) -> np.ndarray:
        labels = np.asarray(labels)
        classes = np.asarray(classes)
        counts = {int(label): int(np.sum(labels == label)) for label in classes}
        missing = {label: count for label, count in counts.items() if count < min_samples_per_class}
        if missing:
            raise ValueError(
                "bounded session adaptation requires every model class; "
                f"insufficient samples: {missing}"
            )
        class_to_index = {int(value): index for index, value in enumerate(classes)}
        targets = np.asarray([class_to_index[int(value)] for value in labels], dtype=np.int64)
        bias = np.zeros(len(classes), dtype=np.float64)
        for _ in range(int(steps)):
            probabilities = _softmax(np.asarray(logits) + bias, 1.0)
            gradient = probabilities.mean(axis=0)
            gradient -= np.bincount(targets, minlength=len(classes)) / len(targets)
            gradient += l2 * bias
            bias -= learning_rate * gradient
            bias -= bias.mean()
            bias = np.clip(bias, -max_abs_bias, max_abs_bias)
        return bias

    def fit_bounded_session_adapter(
        self,
        emg_features: np.ndarray,
        imu_features: np.ndarray,
        direction: np.ndarray,
        gesture: np.ndarray,
        *,
        max_abs_bias: float = 0.35,
        min_samples_per_class: int = 2,
        steps: int = 200,
        learning_rate: float = 0.1,
        l2: float = 0.5,
    ) -> "BaselinePredictor":
        """Fit a small labeled-session output bias without changing either SVM."""
        self._require_fit()
        if not 0.0 < max_abs_bias <= 1.0:
            raise ValueError("max_abs_bias must lie in (0,1]")
        if min_samples_per_class < 1 or steps < 1 or learning_rate <= 0 or l2 < 0:
            raise ValueError("invalid bounded session adaptation settings")
        emg = np.asarray(emg_features)
        imu = np.asarray(imu_features)
        direction = np.asarray(direction)
        gesture = np.asarray(gesture)
        if not (len(emg) == len(imu) == len(direction) == len(gesture)):
            raise ValueError("session adaptation arrays must have equal lengths")
        if not np.isfinite(emg).all() or not np.isfinite(imu).all():
            raise ValueError("session adaptation features must be finite")
        d_classes = np.asarray(self.direction_model.classes_)
        h_classes = np.asarray(self.gesture_model.classes_)
        invalid_direction = sorted(set(map(int, np.unique(direction))) - set(map(int, d_classes)))
        invalid_gesture = sorted(set(map(int, np.unique(gesture))) - set(map(int, h_classes)))
        if invalid_direction or invalid_gesture:
            raise ValueError(
                f"session adaptation labels are outside model classes: "
                f"direction={invalid_direction}, gesture={invalid_gesture}"
            )
        d_logits = _decision_logits(self.direction_model, imu) / max(
            self.direction_temperature.temperature, 1e-4,
        )
        h_logits = _decision_logits(self.gesture_model, emg) / max(
            self.gesture_temperature.temperature, 1e-4,
        )
        direction_bias = self._fit_bounded_bias(
            d_logits, direction, d_classes, max_abs_bias=max_abs_bias,
            min_samples_per_class=min_samples_per_class, steps=steps,
            learning_rate=learning_rate, l2=l2,
        )
        gesture_bias = self._fit_bounded_bias(
            h_logits, gesture, h_classes, max_abs_bias=max_abs_bias,
            min_samples_per_class=min_samples_per_class, steps=steps,
            learning_rate=learning_rate, l2=l2,
        )
        # Commit both heads together so a gesture-side validation failure cannot
        # leave a half-adapted predictor in memory.
        self.direction_logit_bias = direction_bias
        self.gesture_logit_bias = gesture_bias
        self.metadata["session_adapter"] = {
            "method": "bounded_calibrated_logit_bias",
            "max_abs_bias": float(max_abs_bias),
            "min_samples_per_class": int(min_samples_per_class),
            "sample_count": int(len(direction)),
            "base_artifact_sha256": self.metadata.get("loaded_artifact_sha256"),
            "direction_bias": self.direction_logit_bias.tolist(),
            "gesture_bias": self.gesture_logit_bias.tolist(),
        }
        return self

    def reset_session_adapter(self) -> None:
        self.direction_logit_bias = None
        self.gesture_logit_bias = None
        if hasattr(self, "metadata"):
            self.metadata.pop("session_adapter", None)

    def _require_fit(self) -> None:
        if self.direction_model is None or self.gesture_model is None:
            raise RuntimeError("baseline predictor has not been fitted")

    def predict_features(self, emg_features: np.ndarray, imu_features: np.ndarray) -> HeadPrediction:
        self._require_fit()
        emg_features = np.asarray(emg_features).reshape(1, -1)
        imu_features = np.asarray(imu_features).reshape(1, -1)
        d_logits = _decision_logits(self.direction_model, imu_features) / max(
            self.direction_temperature.temperature, 1e-4,
        )
        h_logits = _decision_logits(self.gesture_model, emg_features) / max(
            self.gesture_temperature.temperature, 1e-4,
        )
        d_bias = getattr(self, "direction_logit_bias", None)
        h_bias = getattr(self, "gesture_logit_bias", None)
        d_prob = _softmax(d_logits + (0.0 if d_bias is None else d_bias), 1.0)[0]
        h_prob = _softmax(h_logits + (0.0 if h_bias is None else h_bias), 1.0)[0]
        d_index, h_index = int(d_prob.argmax()), int(h_prob.argmax())
        qd, qh = float(d_prob[d_index]), float(h_prob[h_index])
        d_margin = qd - float(np.partition(d_prob, -2)[-2]) if len(d_prob) > 1 else qd
        h_margin = qh - float(np.partition(h_prob, -2)[-2]) if len(h_prob) > 1 else qh
        d_class = int(self.direction_model.classes_[d_index])
        h_class = int(self.gesture_model.classes_[h_index])
        d_guard = getattr(self, "direction_domain_guard", None)
        h_guard = getattr(self, "gesture_domain_guard", None)
        d_drift = 0.0 if d_guard is None else d_guard.score(imu_features[0], d_class)
        h_drift = 0.0 if h_guard is None else h_guard.score(emg_features[0], h_class)
        d_confidence_rejected = qd < self.direction_threshold
        h_confidence_rejected = qh < self.gesture_threshold
        d_drift_rejected = d_drift > getattr(self, "direction_drift_threshold", 1.0)
        h_drift_rejected = h_drift > getattr(self, "gesture_drift_threshold", 1.0)
        d_ok = not d_confidence_rejected and not d_drift_rejected
        h_ok = not h_confidence_rejected and not h_drift_rejected
        direction = Direction(d_class) if d_ok else Direction.UNKNOWN
        gesture = Gesture(h_class) if h_ok else Gesture.UNKNOWN
        return HeadPrediction(
            direction=direction, gesture=gesture, q_direction=qd, q_gesture=qh,
            direction_margin=d_margin, gesture_margin=h_margin,
            direction_drift=d_drift, gesture_drift=h_drift,
            direction_confidence_rejected=d_confidence_rejected,
            gesture_confidence_rejected=h_confidence_rejected,
            direction_drift_rejected=d_drift_rejected,
            gesture_drift_rejected=h_drift_rejected,
        )

    def predict(self, normalized_emg: np.ndarray, normalized_imu: np.ndarray) -> HeadPrediction:
        return self.predict_features(
            extract_emg_features(normalized_emg),
            extract_imu_features(normalized_imu),
        )

    def save(self, path: str | Path) -> None:
        self._require_fit()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as stream:
            pickle.dump(self, stream)

    @classmethod
    def load(cls, path: str | Path) -> "BaselinePredictor":
        source = Path(path)
        with source.open("rb") as stream:
            value = pickle.load(stream)
        if not isinstance(value, cls):
            raise TypeError("artifact is not a BaselinePredictor")
        if not hasattr(value, "metadata"):
            value.metadata = {}
        value.metadata["loaded_artifact_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        return value


def fit_lda_sanity_baselines(
    emg_features: np.ndarray,
    imu_features: np.ndarray,
    gestures: Iterable[int],
    directions: Iterable[int],
) -> tuple[Any, Any]:
    try:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("scikit-learn is required to train LDA baselines") from exc
    gesture_model = Pipeline([
        ("scale", StandardScaler()),
        ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    ]).fit(emg_features, np.asarray(list(gestures)))
    direction_model = Pipeline([
        ("scale", StandardScaler()),
        ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    ]).fit(imu_features, np.asarray(list(directions)))
    return gesture_model, direction_model
