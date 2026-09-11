from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from .state import Direction, Gesture


BODY_DIRECTION_VECTORS: dict[Direction, np.ndarray] = {
    Direction.RIGHT: np.array([1.0, 0.0, 0.0]),
    Direction.LEFT: np.array([-1.0, 0.0, 0.0]),
    Direction.FORWARD: np.array([0.0, 1.0, 0.0]),
    Direction.BACKWARD: np.array([0.0, -1.0, 0.0]),
    Direction.UP: np.array([0.0, 0.0, 1.0]),
    Direction.DOWN: np.array([0.0, 0.0, -1.0]),
}


def _samples(value: np.ndarray, channels: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != channels:
        raise ValueError(f"{name} must have shape [samples,{channels}]")
    if len(array) < 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} needs at least two finite samples")
    return array


def find_circular_channel_shift(current: np.ndarray, reference: np.ndarray) -> int:
    """Return the roll applied to current so its energy profile matches reference."""
    current = np.asarray(current, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    if current.shape != (8,) or reference.shape != (8,):
        raise ValueError("channel profiles must each contain 8 values")
    current = current / max(float(np.linalg.norm(current)), 1e-12)
    reference = reference / max(float(np.linalg.norm(reference)), 1e-12)
    errors = [np.mean((np.roll(current, shift) - reference) ** 2) for shift in range(8)]
    return int(np.argmin(errors))


def circular_channel_shift_errors(current: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return normalized matching errors for every circular channel shift."""
    current = np.asarray(current, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    if current.shape != (8,) or reference.shape != (8,):
        raise ValueError("channel profiles must each contain 8 values")
    current = current / max(float(np.linalg.norm(current)), 1e-12)
    reference = reference / max(float(np.linalg.norm(reference)), 1e-12)
    return np.asarray([
        np.mean((np.roll(current, shift) - reference) ** 2) for shift in range(8)
    ], dtype=np.float64)


def fit_body_rotation(
    observed_direction_vectors: Mapping[Direction, np.ndarray] | None,
) -> np.ndarray:
    """Fit an orthogonal device-to-body transform from guided movements."""
    if not observed_direction_vectors:
        return np.eye(3, dtype=np.float64)
    observed: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for raw_direction, raw_vector in observed_direction_vectors.items():
        direction = Direction(raw_direction)
        if direction not in BODY_DIRECTION_VECTORS:
            continue
        vector = np.asarray(raw_vector, dtype=np.float64).reshape(-1)
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError(f"invalid calibration vector for {direction.name}")
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-9:
            continue
        observed.append(vector / norm)
        targets.append(BODY_DIRECTION_VECTORS[direction])
    if len(observed) < 3:
        raise ValueError("at least three non-degenerate guided directions are required")
    x = np.stack(observed)
    y = np.stack(targets)
    raw, *_ = np.linalg.lstsq(x, y, rcond=None)
    u, _, vt = np.linalg.svd(raw)
    rotation_row = u @ vt
    if np.linalg.det(rotation_row) < 0:
        u[:, -1] *= -1
        rotation_row = u @ vt
    # Samples are row vectors: body = device @ rotation.T.
    return rotation_row.T


@dataclass(slots=True)
class SessionCalibration:
    emg_center: np.ndarray
    emg_scale: np.ndarray
    gravity_device: np.ndarray
    gyro_bias: np.ndarray
    device_to_body: np.ndarray = field(default_factory=lambda: np.eye(3))
    emg_channel_shift: int = 0
    accel_motion_scale: float = 1.0
    gyro_motion_scale: float = 1.0
    sample_rate_hz: int = 200
    notch_50hz: bool = False
    calibration_version: int = 2
    emg_scale_mode: str = "per_channel"
    emg_global_scale: float = 1.0
    emg_channel_shift_errors: np.ndarray = field(default_factory=lambda: np.zeros(8))
    emg_channel_shift_margin: float = 0.0
    emg_channel_shift_confident: bool = False
    emg_channel_shift_evaluated: bool = False
    body_rotation_residual: float = 0.0
    body_rotation_fitted: bool = False

    def __post_init__(self) -> None:
        self.emg_center = np.asarray(self.emg_center, dtype=np.float64).reshape(8)
        self.emg_scale = np.asarray(self.emg_scale, dtype=np.float64).reshape(8)
        self.gravity_device = np.asarray(self.gravity_device, dtype=np.float64).reshape(3)
        self.gyro_bias = np.asarray(self.gyro_bias, dtype=np.float64).reshape(3)
        self.device_to_body = np.asarray(self.device_to_body, dtype=np.float64).reshape(3, 3)
        if np.any(self.emg_scale <= 0):
            raise ValueError("all EMG scales must be positive")
        if self.accel_motion_scale <= 0 or self.gyro_motion_scale <= 0:
            raise ValueError("motion scales must be positive")
        if self.emg_global_scale <= 0:
            raise ValueError("global EMG scale must be positive")
        if self.emg_scale_mode not in {"per_channel", "global"}:
            raise ValueError("emg_scale_mode must be per_channel or global")
        self.emg_channel_shift = int(self.emg_channel_shift) % 8
        self.emg_channel_shift_errors = np.asarray(
            self.emg_channel_shift_errors, dtype=np.float64,
        ).reshape(8)
        if not np.isfinite(self.emg_channel_shift_errors).all():
            raise ValueError("channel shift errors must be finite")
        self.emg_channel_shift_margin = float(max(self.emg_channel_shift_margin, 0.0))
        self.body_rotation_residual = float(max(self.body_rotation_residual, 0.0))

    @classmethod
    def identity(cls, sample_rate_hz: int = 200) -> "SessionCalibration":
        return cls(
            emg_center=np.zeros(8), emg_scale=np.ones(8),
            gravity_device=np.zeros(3), gyro_bias=np.zeros(3),
            sample_rate_hz=sample_rate_hz,
        )

    def transform_emg(self, emg: np.ndarray) -> np.ndarray:
        array = np.asarray(emg, dtype=np.float64)
        scale = self.emg_scale if self.emg_scale_mode == "per_channel" else self.emg_global_scale
        normalized = (array - self.emg_center) / scale
        return np.roll(normalized, self.emg_channel_shift, axis=-1)

    def transform_imu(self, accel: np.ndarray, gyro: np.ndarray) -> np.ndarray:
        accel_array = np.asarray(accel, dtype=np.float64)
        gyro_array = np.asarray(gyro, dtype=np.float64)
        linear = (accel_array - self.gravity_device) @ self.device_to_body.T
        angular = (gyro_array - self.gyro_bias) @ self.device_to_body.T
        return np.concatenate(
            [linear / self.accel_motion_scale, angular / self.gyro_motion_scale], axis=-1,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "emg_center": self.emg_center.tolist(),
            "emg_scale": self.emg_scale.tolist(),
            "gravity_device": self.gravity_device.tolist(),
            "gyro_bias": self.gyro_bias.tolist(),
            "device_to_body": self.device_to_body.tolist(),
            "emg_channel_shift": self.emg_channel_shift,
            "accel_motion_scale": self.accel_motion_scale,
            "gyro_motion_scale": self.gyro_motion_scale,
            "sample_rate_hz": self.sample_rate_hz,
            "notch_50hz": self.notch_50hz,
            "calibration_version": self.calibration_version,
            "emg_scale_mode": self.emg_scale_mode,
            "emg_global_scale": self.emg_global_scale,
            "emg_channel_shift_errors": self.emg_channel_shift_errors.tolist(),
            "emg_channel_shift_margin": self.emg_channel_shift_margin,
            "emg_channel_shift_confident": self.emg_channel_shift_confident,
            "emg_channel_shift_evaluated": self.emg_channel_shift_evaluated,
            "body_rotation_residual": self.body_rotation_residual,
            "body_rotation_fitted": self.body_rotation_fitted,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "SessionCalibration":
        payload = dict(value)
        if "calibration_version" not in payload:
            payload["calibration_version"] = 1
            payload.setdefault("emg_scale_mode", "per_channel")
        return cls(**payload)


def fit_session_calibration(
    rest_emg: np.ndarray,
    gesture_emg: np.ndarray | Mapping[Gesture | str, np.ndarray],
    rest_accel: np.ndarray,
    rest_gyro: np.ndarray,
    motion_accel: np.ndarray,
    motion_gyro: np.ndarray,
    *,
    observed_direction_vectors: Mapping[Direction, np.ndarray] | None = None,
    reference_emg_profile: np.ndarray | None = None,
    reference_emg_profiles: Mapping[Gesture | str, np.ndarray] | None = None,
    sample_rate_hz: int = 200,
    emg_scale_mode: str = "global",
    shift_confidence_threshold: float = 0.10,
) -> SessionCalibration:
    rest_emg = _samples(rest_emg, 8, "rest_emg")
    if isinstance(gesture_emg, Mapping):
        gesture_blocks = {
            str(key.name if isinstance(key, Gesture) else key).lower(): _samples(
                value, 8, f"gesture_emg[{key}]",
            )
            for key, value in gesture_emg.items()
        }
        if not gesture_blocks:
            raise ValueError("gesture_emg mapping must not be empty")
    else:
        gesture_blocks = {"combined": _samples(gesture_emg, 8, "gesture_emg")}
    rest_accel = _samples(rest_accel, 3, "rest_accel")
    rest_gyro = _samples(rest_gyro, 3, "rest_gyro")
    motion_accel = _samples(motion_accel, 3, "motion_accel")
    motion_gyro = _samples(motion_gyro, 3, "motion_gyro")

    # Calibration and deployment must see the same causal preprocessing.
    from .signal import CausalEMGFilter, line_noise_ratio_db
    notch_50hz = line_noise_ratio_db(rest_emg, sample_rate_hz) >= 6.0
    rest_filter = CausalEMGFilter(sample_rate_hz, notch_50hz=notch_50hz)
    filtered_rest_emg = rest_filter.process(rest_emg)
    filtered_gesture_blocks = {
        key: CausalEMGFilter(sample_rate_hz, notch_50hz=notch_50hz).process(value)
        for key, value in gesture_blocks.items()
    }
    filtered_gesture_emg = np.concatenate(list(filtered_gesture_blocks.values()))
    emg_center = np.median(filtered_rest_emg, axis=0)
    comfortable = np.percentile(np.abs(filtered_gesture_emg - emg_center), 95, axis=0)
    rest_noise = np.percentile(np.abs(filtered_rest_emg - emg_center), 95, axis=0)
    emg_scale = np.maximum(comfortable, np.maximum(rest_noise * 3.0, 1e-6))
    emg_global_scale = max(
        float(np.percentile(np.abs(filtered_gesture_emg - emg_center), 95)),
        float(np.percentile(np.abs(filtered_rest_emg - emg_center), 95)) * 3.0,
        1e-6,
    )
    profiles = {
        key: np.sqrt(np.mean((values - emg_center) ** 2, axis=0))
        for key, values in filtered_gesture_blocks.items()
    }
    shift = 0
    shift_errors = np.zeros(8, dtype=np.float64)
    shift_margin = 0.0
    shift_confident = False
    reference_profiles = {
        str(key.name if isinstance(key, Gesture) else key).lower(): np.asarray(value)
        for key, value in (reference_emg_profiles or {}).items()
    }
    matched_errors: list[np.ndarray] = []
    for key, profile in profiles.items():
        reference = reference_profiles.get(key)
        if reference is not None:
            matched_errors.append(circular_channel_shift_errors(profile, reference))
    if not matched_errors and reference_emg_profile is not None:
        aggregate_profile = np.sqrt(np.mean((filtered_gesture_emg - emg_center) ** 2, axis=0))
        matched_errors.append(circular_channel_shift_errors(aggregate_profile, reference_emg_profile))
    if matched_errors:
        shift_errors = np.mean(np.stack(matched_errors), axis=0)
        order = np.argsort(shift_errors)
        candidate_shift = int(order[0])
        best, second = float(shift_errors[order[0]]), float(shift_errors[order[1]])
        shift_margin = max(0.0, (second - best) / max(second, 1e-12))
        shift_confident = shift_margin >= shift_confidence_threshold
        # An ambiguous rotation is safer left uncorrected than silently rolled
        # to the wrong physical channels.  The candidate remains recoverable as
        # argmin(emg_channel_shift_errors) for diagnostics or guided retry.
        shift = candidate_shift if shift_confident else 0

    gravity = np.median(rest_accel, axis=0)
    gyro_bias = np.median(rest_gyro, axis=0)
    accel_norm = np.linalg.norm(motion_accel - gravity, axis=1)
    gyro_norm = np.linalg.norm(motion_gyro - gyro_bias, axis=1)
    accel_scale = max(float(np.percentile(accel_norm, 95)), 1e-6)
    gyro_scale = max(float(np.percentile(gyro_norm, 95)), 1e-6)
    body_rotation = fit_body_rotation(observed_direction_vectors)
    body_residual = 0.0
    if observed_direction_vectors:
        residuals = []
        for raw_direction, raw_vector in observed_direction_vectors.items():
            direction = Direction(raw_direction)
            vector = np.asarray(raw_vector, dtype=np.float64).reshape(3)
            norm = float(np.linalg.norm(vector))
            if direction in BODY_DIRECTION_VECTORS and norm > 1e-9:
                residuals.append(float(np.linalg.norm(
                    (vector / norm) @ body_rotation.T - BODY_DIRECTION_VECTORS[direction]
                )))
        body_residual = float(np.mean(residuals)) if residuals else 0.0
    return SessionCalibration(
        emg_center=emg_center,
        emg_scale=emg_scale,
        gravity_device=gravity,
        gyro_bias=gyro_bias,
        device_to_body=body_rotation,
        emg_channel_shift=shift,
        accel_motion_scale=accel_scale,
        gyro_motion_scale=gyro_scale,
        sample_rate_hz=int(sample_rate_hz),
        notch_50hz=notch_50hz,
        emg_scale_mode=emg_scale_mode,
        emg_global_scale=emg_global_scale,
        emg_channel_shift_errors=shift_errors,
        emg_channel_shift_margin=shift_margin,
        emg_channel_shift_confident=shift_confident,
        emg_channel_shift_evaluated=bool(matched_errors),
        body_rotation_residual=body_residual,
        body_rotation_fitted=bool(observed_direction_vectors),
    )
