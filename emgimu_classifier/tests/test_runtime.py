import importlib.util
import unittest

import numpy as np

from emgimu.baseline import HeadPrediction
from emgimu.calibration import SessionCalibration
from emgimu.runtime import HumanStateEstimator
from emgimu.state import Direction, Gesture, Phase, QualityFlag


class FixedPredictor:
    def predict(self, normalized_emg, normalized_imu):
        return HeadPrediction(Direction.RIGHT, Gesture.FIST, 0.95, 0.90)


class ArmFirstPredictor:
    def __init__(self):
        self.calls = 0

    def predict(self, normalized_emg, normalized_imu):
        self.calls += 1
        gesture = Gesture.NEUTRAL if self.calls == 1 else Gesture.FIST
        return HeadPrediction(Direction.RIGHT, gesture, 0.95, 0.90)


class DriftRejectingPredictor:
    def predict(self, normalized_emg, normalized_imu):
        return HeadPrediction(
            Direction.UNKNOWN, Gesture.FIST, 0.95, 0.90,
            direction_drift=0.9, direction_drift_rejected=True,
        )


class FiniteInputPredictor(FixedPredictor):
    def predict(self, normalized_emg, normalized_imu):
        if not np.isfinite(normalized_emg).all() or not np.isfinite(normalized_imu).all():
            raise AssertionError("predictor received non-finite runtime input")
        return super().predict(normalized_emg, normalized_imu)


@unittest.skipUnless(importlib.util.find_spec("scipy"), "scipy is required by runtime filtering")
class RuntimeTests(unittest.TestCase):
    def test_causal_output_rate_and_hysteresis(self):
        estimator = HumanStateEstimator(FixedPredictor(), SessionCalibration.identity())
        rng = np.random.default_rng(3)
        states = estimator.push_batch(
            np.arange(80) * 5,
            rng.normal(size=(80, 8)),
            rng.normal(size=(80, 3)),
            rng.normal(size=(80, 3)),
        )
        self.assertEqual(len(states), 6)  # at sample 40, then every 8 samples
        self.assertEqual(states[0].direction, Direction.NONE)
        self.assertEqual(states[0].phase.arm, Phase.ONSET)
        self.assertEqual(states[1].direction, Direction.RIGHT)
        self.assertTrue(states[-1].signal_quality.flags & QualityFlag.CALIBRATED)

    def test_out_of_order_sample_is_ignored(self):
        estimator = HumanStateEstimator(FixedPredictor(), SessionCalibration.identity())
        sample = np.ones(8)
        self.assertIsNone(estimator.push_sample(10, sample, np.ones(3), np.ones(3)))
        self.assertIsNone(estimator.push_sample(9, sample, np.ones(3), np.ones(3)))
        self.assertEqual(estimator.last_timestamp_ms, 10)

    def test_bad_channels_force_gesture_unknown(self):
        estimator = HumanStateEstimator(FixedPredictor(), SessionCalibration.identity())
        rng = np.random.default_rng(12)
        states = estimator.push_batch(
            np.arange(80) * 5,
            rng.normal(size=(80, 8)),
            rng.normal(size=(80, 3)),
            rng.normal(size=(80, 3)),
            channel_quality=np.asarray([1, 1, 1, 1, 1, 0, 0, 0], dtype=float),
        )
        self.assertTrue(states)
        self.assertTrue(all(state.gesture == Gesture.UNKNOWN for state in states))
        self.assertTrue(all(state.signal_quality.flags & QualityFlag.BAD_CHANNELS for state in states))

    def test_per_sample_channel_quality_is_supported(self):
        estimator = HumanStateEstimator(FixedPredictor(), SessionCalibration.identity())
        rng = np.random.default_rng(13)
        channel_quality = np.ones((80, 8), dtype=float)
        channel_quality[:, 5:] = 0.0
        states = estimator.push_batch(
            np.arange(80) * 5,
            rng.normal(size=(80, 8)),
            rng.normal(size=(80, 3)),
            rng.normal(size=(80, 3)),
            channel_quality=channel_quality,
        )
        self.assertTrue(states)
        self.assertTrue(all(state.gesture == Gesture.UNKNOWN for state in states))

    def test_nonfinite_sensor_input_fails_closed_without_crashing_predictor(self):
        estimator = HumanStateEstimator(FiniteInputPredictor(), SessionCalibration.identity())
        rng = np.random.default_rng(14)
        emg = rng.normal(size=(80, 8))
        accel = np.full((80, 3), np.nan)
        gyro = np.full((80, 3), np.inf)
        states = estimator.push_batch(np.arange(80) * 5, emg, accel, gyro)
        self.assertTrue(states)
        self.assertTrue(all(state.direction == Direction.UNKNOWN for state in states))
        self.assertTrue(all(not (state.signal_quality.flags & QualityFlag.IMU_VALID) for state in states))

    def test_onset_lag_uses_hand_minus_arm_sign(self):
        estimator = HumanStateEstimator(ArmFirstPredictor(), SessionCalibration.identity())
        rng = np.random.default_rng(21)
        estimator.push_batch(
            np.arange(56) * 5,
            rng.normal(size=(56, 8)),
            rng.normal(size=(56, 3)),
            rng.normal(size=(56, 3)),
        )
        self.assertEqual(estimator.current_onset_lag_ms, 40.0)

    def test_model_rejection_reason_reaches_quality_flags(self):
        estimator = HumanStateEstimator(DriftRejectingPredictor(), SessionCalibration.identity())
        rng = np.random.default_rng(22)
        states = estimator.push_batch(
            np.arange(48) * 5,
            rng.normal(size=(48, 8)), rng.normal(size=(48, 3)), rng.normal(size=(48, 3)),
        )
        self.assertTrue(states[-1].signal_quality.flags & QualityFlag.MODEL_DRIFT)
        self.assertEqual(states[-1].direction_drift, 0.9)

    def test_model_calibration_contract_is_enforced(self):
        predictor = FixedPredictor()
        predictor.metadata = {
            "sample_rate_hz": 200,
            "calibration_contract": {"versions": [2], "emg_scale_modes": ["global"]},
        }
        with self.assertRaisesRegex(ValueError, "calibration modes"):
            HumanStateEstimator(predictor, SessionCalibration.identity())

    def test_uncertain_calibration_rejects_only_affected_head(self):
        calibration = SessionCalibration.identity()
        calibration.emg_channel_shift_evaluated = True
        calibration.emg_channel_shift_confident = False
        calibration.body_rotation_fitted = True
        calibration.body_rotation_residual = 0.1
        estimator = HumanStateEstimator(FixedPredictor(), calibration)
        rng = np.random.default_rng(23)
        states = estimator.push_batch(
            np.arange(48) * 5,
            rng.normal(size=(48, 8)), rng.normal(size=(48, 3)), rng.normal(size=(48, 3)),
        )
        self.assertEqual(states[-1].gesture, Gesture.UNKNOWN)
        self.assertNotEqual(states[-1].direction, Direction.UNKNOWN)
        self.assertTrue(states[-1].signal_quality.flags & QualityFlag.CALIBRATION_UNCERTAIN)

        calibration.emg_channel_shift_confident = True
        calibration.body_rotation_residual = 0.8
        estimator = HumanStateEstimator(FixedPredictor(), calibration)
        states = estimator.push_batch(
            np.arange(48) * 5,
            rng.normal(size=(48, 8)), rng.normal(size=(48, 3)), rng.normal(size=(48, 3)),
        )
        self.assertEqual(states[-1].direction, Direction.UNKNOWN)
        self.assertNotEqual(states[-1].gesture, Gesture.UNKNOWN)


if __name__ == "__main__":
    unittest.main()
