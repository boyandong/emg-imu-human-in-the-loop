import unittest

import numpy as np

from emgimu.calibration import SessionCalibration, find_circular_channel_shift, fit_body_rotation, fit_session_calibration
from emgimu.state import Direction, Gesture


class CalibrationTests(unittest.TestCase):
    def test_channel_shift_recovers_rotated_profile(self):
        reference = np.arange(1.0, 9.0)
        current = np.roll(reference, 3)
        shift = find_circular_channel_shift(current, reference)
        np.testing.assert_allclose(np.roll(current, shift), reference)

    def test_body_rotation_maps_guided_vectors(self):
        # Device axes are body y, -body x, body z.
        observed = {
            Direction.RIGHT: np.array([0.0, -1.0, 0.0]),
            Direction.FORWARD: np.array([1.0, 0.0, 0.0]),
            Direction.UP: np.array([0.0, 0.0, 1.0]),
        }
        rotation = fit_body_rotation(observed)
        for direction, vector in observed.items():
            mapped = vector @ rotation.T
            expected = {
                Direction.RIGHT: [1, 0, 0], Direction.FORWARD: [0, 1, 0], Direction.UP: [0, 0, 1],
            }[direction]
            np.testing.assert_allclose(mapped, expected, atol=1e-6)

    def test_fit_calibration_is_finite(self):
        rng = np.random.default_rng(1)
        calibration = fit_session_calibration(
            rng.normal(0, 0.01, (200, 8)), rng.normal(0, 1, (600, 8)),
            np.tile([0, 0, 9.81], (200, 1)) + rng.normal(0, 0.01, (200, 3)),
            rng.normal(0, 0.01, (200, 3)),
            rng.normal(0, 1, (600, 3)) + [0, 0, 9.81], rng.normal(0, 1, (600, 3)),
        )
        self.assertTrue(np.isfinite(calibration.transform_emg(np.zeros((2, 8)))).all())
        self.assertEqual(calibration.emg_scale_mode, "global")

    def test_multigesture_shift_has_margin_and_preserves_spatial_ratio(self):
        rng = np.random.default_rng(7)
        reference = {
            Gesture.FIST: np.array([8, 6, 3, 1, 1, 1, 2, 4], dtype=float),
            Gesture.PINCH: np.array([1, 2, 7, 8, 4, 1, 1, 1], dtype=float),
            Gesture.OPEN: np.array([2, 1, 1, 2, 5, 8, 6, 3], dtype=float),
        }
        t = np.arange(400) / 200.0
        gesture_blocks = {
            gesture: np.sin(2 * np.pi * 35 * t)[:, None] * np.roll(profile, 2)[None, :]
            for gesture, profile in reference.items()
        }
        calibration = fit_session_calibration(
            rng.normal(0, 0.01, (400, 8)), gesture_blocks,
            np.tile([0, 0, 9.81], (400, 1)), rng.normal(0, 0.01, (400, 3)),
            rng.normal(0, 1, (600, 3)) + [0, 0, 9.81], rng.normal(0, 1, (600, 3)),
            reference_emg_profiles=reference,
        )
        self.assertEqual(calibration.emg_channel_shift, 6)
        self.assertTrue(calibration.emg_channel_shift_confident)
        self.assertGreater(calibration.emg_channel_shift_margin, 0.1)
        values = calibration.emg_center + calibration.emg_global_scale * np.arange(1, 9)
        normalized = np.roll(calibration.transform_emg(values), -calibration.emg_channel_shift)
        np.testing.assert_allclose(normalized, np.arange(1, 9), atol=1e-6)

    def test_legacy_json_uses_legacy_scaling_contract(self):
        legacy = SessionCalibration.identity().to_dict()
        for key in (
            "calibration_version", "emg_scale_mode", "emg_global_scale",
            "emg_channel_shift_errors", "emg_channel_shift_margin",
            "emg_channel_shift_confident", "emg_channel_shift_evaluated",
            "body_rotation_residual", "body_rotation_fitted",
        ):
            legacy.pop(key)
        restored = SessionCalibration.from_dict(legacy)
        self.assertEqual(restored.calibration_version, 1)
        self.assertEqual(restored.emg_scale_mode, "per_channel")


if __name__ == "__main__":
    unittest.main()
