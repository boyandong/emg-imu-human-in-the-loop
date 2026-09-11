import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from emgimu.baseline import BaselinePredictor
from emgimu.metrics import evaluate_predictions, first_stable_latency
from emgimu.state import Direction, Gesture


@unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn is not installed")
class BaselineMetricsTests(unittest.TestCase):
    def test_svm_fit_calibrate_reject_and_predict(self):
        rng = np.random.default_rng(4)
        train_count = 240
        validation_count = 120
        train_d = np.arange(train_count) % 7
        train_h = np.arange(train_count) % 4
        validation_d = np.arange(validation_count) % 7
        validation_h = np.arange(validation_count) % 4
        train_imu = rng.normal(0, 0.05, (train_count, 36))
        train_emg = rng.normal(0, 0.05, (train_count, 48))
        validation_imu = rng.normal(0, 0.05, (validation_count, 36))
        validation_emg = rng.normal(0, 0.05, (validation_count, 48))
        train_imu[np.arange(train_count), train_d] += 3
        train_emg[np.arange(train_count), train_h] += 3
        validation_imu[np.arange(validation_count), validation_d] += 3
        validation_emg[np.arange(validation_count), validation_h] += 3
        predictor = BaselinePredictor().fit(
            train_emg, train_imu, train_d, train_h,
            validation_emg_features=validation_emg,
            validation_imu_features=validation_imu,
            validation_direction=validation_d,
            validation_gesture=validation_h,
        )
        prediction = predictor.predict_features(validation_emg[10], validation_imu[10])
        self.assertEqual(prediction.direction, Direction(validation_d[10]))
        self.assertEqual(prediction.gesture, Gesture(validation_h[10]))
        self.assertGreaterEqual(prediction.direction_margin, 0.0)
        self.assertGreaterEqual(prediction.gesture_margin, 0.0)
        shifted = predictor.predict_features(np.full(48, 100.0), np.full(36, 100.0))
        self.assertGreater(shifted.direction_drift, predictor.direction_drift_threshold)
        self.assertGreater(shifted.gesture_drift, predictor.gesture_drift_threshold)
        self.assertEqual(shifted.direction, Direction.UNKNOWN)
        self.assertEqual(shifted.gesture, Gesture.UNKNOWN)

        direction_model_id = id(predictor.direction_model)
        gesture_model_id = id(predictor.gesture_model)
        predictor.fit_bounded_session_adapter(
            validation_emg, validation_imu, validation_d, validation_h,
            max_abs_bias=0.2,
        )
        self.assertLessEqual(float(np.max(np.abs(predictor.direction_logit_bias))), 0.2)
        self.assertLessEqual(float(np.max(np.abs(predictor.gesture_logit_bias))), 0.2)
        self.assertEqual(id(predictor.direction_model), direction_model_id)
        self.assertEqual(id(predictor.gesture_model), gesture_model_id)
        self.assertIn("session_adapter", predictor.metadata)
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "adapted.pkl"
            predictor.save(artifact)
            restored = BaselinePredictor.load(artifact)
            np.testing.assert_allclose(
                restored.direction_logit_bias, predictor.direction_logit_bias,
            )
            np.testing.assert_allclose(
                restored.gesture_logit_bias, predictor.gesture_logit_bias,
            )
        predictor.reset_session_adapter()
        self.assertIsNone(predictor.direction_logit_bias)
        self.assertNotIn("session_adapter", predictor.metadata)
        with self.assertRaisesRegex(ValueError, "every model class"):
            predictor.fit_bounded_session_adapter(
                validation_emg, validation_imu, validation_d,
                np.zeros_like(validation_h), max_abs_bias=0.2,
            )
        self.assertIsNone(predictor.direction_logit_bias)
        self.assertIsNone(predictor.gesture_logit_bias)

    def test_bounded_adapter_rejects_missing_classes(self):
        predictor = BaselinePredictor()
        predictor.direction_model = type("Classes", (), {"classes_": np.arange(2)})()
        predictor.gesture_model = type("Classes", (), {"classes_": np.arange(2)})()
        with self.assertRaisesRegex(ValueError, "every model class"):
            BaselinePredictor._fit_bounded_bias(
                np.zeros((3, 2)), np.zeros(3, dtype=int), np.arange(2),
                max_abs_bias=0.2, min_samples_per_class=1,
                steps=2, learning_rate=0.1, l2=0.5,
            )

    def test_threshold_search_protects_each_class_coverage(self):
        truth = np.array([0] * 8 + [1] * 2)
        predicted = truth.copy()
        probabilities = np.array(
            [[0.9, 0.1]] * 8 + [[0.8, 0.2]] * 2,
            dtype=float,
        )
        indices = predicted.copy()
        threshold = BaselinePredictor._select_threshold(
            probabilities, indices, predicted, truth,
            min_coverage=0.9, min_class_coverage=0.75,
        )
        self.assertLessEqual(threshold, 0.2)

    def test_unknown_counts_as_error(self):
        result = evaluate_predictions(
            np.array([0, 1]), np.array([0, 1]),
            np.array([0, -1]), np.array([0, -1]),
            np.array([0.9, 0.2]), np.array([0.9, 0.2]),
            latency_ms=np.array([200, 280]),
        )
        self.assertLess(result.direction_macro_f1, 1)
        self.assertEqual(result.joint_accuracy, 0.5)
        self.assertEqual(result.p50_latency_ms, 240)
        self.assertTrue(result.p95_latency_ms <= 300)
        self.assertEqual(result.direction_labels, (-1, 0, 1))
        self.assertEqual(result.direction_class_coverage[0], 1.0)
        self.assertEqual(result.direction_class_coverage[1], 0.0)
        self.assertTrue(result.direction_risk_coverage)

    def test_metrics_can_equalize_long_and_short_trials(self):
        truth = np.asarray([0, 0, 0, 1])
        predicted = np.asarray([0, 0, 0, 0])
        confidence = np.asarray([0.9, 0.9, 0.9, 0.9])
        result = evaluate_predictions(
            truth, truth, predicted, predicted, confidence, confidence,
            sample_weight=np.asarray([2 / 3, 2 / 3, 2 / 3, 2.0]),
        )
        self.assertAlmostEqual(result.joint_accuracy, 0.5)
        self.assertEqual(sum(map(sum, result.direction_confusion_matrix)), 4)

    def test_latency_matches_first_correct_stable_state(self):
        latency = first_stable_latency(100, np.array([80, 120, 160]), np.array([0, 0, 2]), 2)
        self.assertEqual(latency, 60)


if __name__ == "__main__":
    unittest.main()
