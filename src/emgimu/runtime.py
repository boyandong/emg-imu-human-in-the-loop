from __future__ import annotations

from collections import deque
from typing import Protocol

import numpy as np

from .baseline import HeadPrediction
from .calibration import SessionCalibration
from .consistency import ConditionalConsistencyModel, state_key
from .phases import OpenAwareGestureTracker, PhaseTracker
from .signal import (
    CausalEMGFilter,
    activation_intensity,
    estimate_signal_quality,
    motion_intensity,
)
from .state import (
    Confidence,
    Consistency,
    Direction,
    Gesture,
    HumanState,
    Phase,
    PhasePair,
    QualityFlag,
    SignalQuality,
)


class WindowPredictor(Protocol):
    def predict(self, normalized_emg: np.ndarray, normalized_imu: np.ndarray) -> HeadPrediction: ...


class HumanStateEstimator:
    """Causal 200 Hz input / 25 Hz output human-state estimator."""

    def __init__(
        self,
        predictor: WindowPredictor,
        calibration: SessionCalibration,
        *,
        window_samples: int = 40,
        hop_samples: int = 8,
        quality_threshold: float = 0.60,
        notch_50hz: bool | None = None,
        consistency_model: ConditionalConsistencyModel | None = None,
        open_release_frames: int = 8,
        max_body_rotation_residual: float = 0.35,
    ) -> None:
        if window_samples < 3 or hop_samples < 1 or hop_samples > window_samples:
            raise ValueError("invalid window or hop length")
        self.predictor = predictor
        self.calibration = calibration
        self.window_samples = int(window_samples)
        self.hop_samples = int(hop_samples)
        self.quality_threshold = float(quality_threshold)
        self.consistency_model = consistency_model
        self.max_body_rotation_residual = float(max_body_rotation_residual)
        metadata = getattr(predictor, "metadata", {})
        if isinstance(metadata, dict):
            expected_rate = metadata.get("sample_rate_hz")
            if expected_rate is not None and not np.isclose(
                float(expected_rate), calibration.sample_rate_hz,
            ):
                raise ValueError(
                    f"model expects {expected_rate} Hz but calibration uses "
                    f"{calibration.sample_rate_hz} Hz"
                )
            contract = metadata.get("calibration_contract", {})
            if isinstance(contract, dict):
                modes = set(map(str, contract.get("emg_scale_modes", [])))
                versions = {int(value) for value in contract.get("versions", [])}
                if modes and calibration.emg_scale_mode not in modes:
                    raise ValueError(
                        f"model calibration modes {sorted(modes)} do not include "
                        f"{calibration.emg_scale_mode!r}"
                    )
                if versions and calibration.calibration_version not in versions:
                    raise ValueError(
                        f"model calibration versions {sorted(versions)} do not include "
                        f"{calibration.calibration_version}"
                    )
        use_notch = calibration.notch_50hz if notch_50hz is None else bool(notch_50hz)
        self.filter = CausalEMGFilter(calibration.sample_rate_hz, notch_50hz=use_notch)
        self.arm_phase = PhaseTracker(Direction.NONE, Direction.UNKNOWN)
        self.hand_phase = OpenAwareGestureTracker(
            Gesture.NEUTRAL, Gesture.OPEN, Gesture.UNKNOWN,
            open_release_frames=open_release_frames,
        )
        self.raw_emg: deque[np.ndarray] = deque(maxlen=window_samples)
        self.filtered_emg: deque[np.ndarray] = deque(maxlen=window_samples)
        self.raw_imu: deque[np.ndarray] = deque(maxlen=window_samples)
        self.timestamps: deque[int] = deque(maxlen=window_samples)
        self.sample_quality: deque[float] = deque(maxlen=window_samples)
        self.missing: deque[bool] = deque(maxlen=window_samples)
        self.timestamp_valid: deque[bool] = deque(maxlen=window_samples)
        self.imu_valid: deque[bool] = deque(maxlen=window_samples)
        self.interpolated_imu: deque[bool] = deque(maxlen=window_samples)
        self.channel_quality: deque[np.ndarray] = deque(maxlen=window_samples)
        self.quality_gate_pass: deque[bool] = deque(maxlen=window_samples)
        self.duplicate_packet: deque[bool] = deque(maxlen=window_samples)
        self.out_of_order_packet: deque[bool] = deque(maxlen=window_samples)
        self.samples_since_output = 0
        self.last_timestamp_ms = -1
        self.dropped_samples = False
        self.last_arm_onset_ms: int | None = None
        self.last_hand_onset_ms: int | None = None
        self.current_onset_lag_ms: float | None = None

    def reset(self) -> None:
        self.filter.reset()
        self.arm_phase.reset()
        self.hand_phase.reset()
        self.raw_emg.clear()
        self.filtered_emg.clear()
        self.raw_imu.clear()
        self.timestamps.clear()
        self.sample_quality.clear()
        self.missing.clear()
        self.timestamp_valid.clear()
        self.imu_valid.clear()
        self.interpolated_imu.clear()
        self.channel_quality.clear()
        self.quality_gate_pass.clear()
        self.duplicate_packet.clear()
        self.out_of_order_packet.clear()
        self.samples_since_output = 0
        self.last_timestamp_ms = -1
        self.dropped_samples = False
        self.last_arm_onset_ms = None
        self.last_hand_onset_ms = None
        self.current_onset_lag_ms = None

    def push_sample(
        self,
        timestamp_ms: int,
        emg: np.ndarray,
        accel: np.ndarray,
        gyro: np.ndarray,
        *,
        interpolated_imu: bool = False,
        sample_quality: float = 1.0,
        missing: bool = False,
        channel_quality: np.ndarray | None = None,
        timestamp_valid: bool = True,
        imu_valid: bool = True,
        quality_gate_pass: bool = True,
        duplicate_packet: bool = False,
        out_of_order_packet: bool = False,
    ) -> HumanState | None:
        timestamp_ms = int(timestamp_ms)
        if not timestamp_valid or timestamp_ms <= self.last_timestamp_ms:
            return None
        expected_period = 1000.0 / self.calibration.sample_rate_hz
        if self.last_timestamp_ms >= 0 and timestamp_ms - self.last_timestamp_ms > expected_period * 2.5:
            self.dropped_samples = True
            missing = True
        self.last_timestamp_ms = timestamp_ms

        emg_sample = np.asarray(emg, dtype=np.float64).reshape(8)
        accel_sample = np.asarray(accel, dtype=np.float64).reshape(3)
        gyro_sample = np.asarray(gyro, dtype=np.float64).reshape(3)
        raw_imu_sample = np.concatenate([accel_sample, gyro_sample])
        emg_finite = np.isfinite(emg_sample)
        imu_finite = np.isfinite(raw_imu_sample)
        if not emg_finite.all() or not imu_finite.all():
            missing = True
        if not imu_finite.all():
            imu_valid = False
        safe_emg = np.nan_to_num(emg_sample, nan=0.0, posinf=0.0, neginf=0.0)
        filtered = self.filter.process(safe_emg[None, :])[0]
        self.raw_emg.append(emg_sample)
        self.filtered_emg.append(filtered)
        self.raw_imu.append(raw_imu_sample)
        self.timestamps.append(timestamp_ms)
        sample_quality_value = float(sample_quality)
        if not np.isfinite(sample_quality_value):
            sample_quality_value = 0.0
        self.sample_quality.append(float(np.clip(sample_quality_value, 0.0, 1.0)))
        self.missing.append(bool(missing))
        self.timestamp_valid.append(bool(timestamp_valid))
        self.imu_valid.append(bool(imu_valid))
        self.interpolated_imu.append(bool(interpolated_imu))
        channels = np.ones(8, dtype=np.float64) if channel_quality is None else np.asarray(
            channel_quality, dtype=np.float64,
        ).reshape(8)
        channels = np.where(emg_finite, channels, 0.0)
        self.channel_quality.append(np.clip(
            np.nan_to_num(channels, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0,
        ))
        self.quality_gate_pass.append(bool(quality_gate_pass))
        self.duplicate_packet.append(bool(duplicate_packet))
        self.out_of_order_packet.append(bool(out_of_order_packet))
        self.samples_since_output += 1
        if len(self.timestamps) < self.window_samples or self.samples_since_output < self.hop_samples:
            return None
        self.samples_since_output = 0
        return self._estimate()

    def push_batch(
        self,
        timestamp_ms: np.ndarray,
        emg: np.ndarray,
        accel: np.ndarray,
        gyro: np.ndarray,
        *,
        window_quality: np.ndarray | None = None,
        missing_mask: np.ndarray | None = None,
        channel_quality: np.ndarray | None = None,
        timestamp_valid: np.ndarray | None = None,
        imu_valid: np.ndarray | None = None,
        interpolated_imu: np.ndarray | None = None,
        quality_gate_pass: bool | np.ndarray = True,
    ) -> list[HumanState]:
        timestamps = np.asarray(timestamp_ms).reshape(-1)
        emg = np.asarray(emg)
        accel = np.asarray(accel)
        gyro = np.asarray(gyro)
        n = len(timestamps)
        if emg.shape != (n, 8) or accel.shape != (n, 3) or gyro.shape != (n, 3):
            raise ValueError("batch sensor shapes do not match timestamps")
        states: list[HumanState] = []
        sample_quality_values = np.ones(n) if window_quality is None else np.asarray(window_quality).reshape(n)
        missing_values = np.zeros(n, dtype=bool) if missing_mask is None else np.asarray(missing_mask).reshape(n)
        timestamp_values = np.ones(n, dtype=bool) if timestamp_valid is None else np.asarray(timestamp_valid).reshape(n)
        imu_valid_values = np.ones(n, dtype=bool) if imu_valid is None else np.asarray(imu_valid).reshape(n)
        interpolated_values = (
            np.zeros(n, dtype=bool) if interpolated_imu is None
            else np.asarray(interpolated_imu).reshape(n)
        )
        if channel_quality is None:
            channel_quality_values = np.ones((n, 8), dtype=np.float64)
        else:
            supplied_channel_quality = np.asarray(channel_quality, dtype=np.float64)
            if supplied_channel_quality.shape == (8,):
                channel_quality_values = np.broadcast_to(supplied_channel_quality, (n, 8))
            elif supplied_channel_quality.shape == (n, 8):
                channel_quality_values = supplied_channel_quality
            else:
                raise ValueError("channel_quality must have shape [8] or [samples,8]")
        if np.ndim(quality_gate_pass) == 0:
            quality_gate_values = np.full(n, bool(quality_gate_pass), dtype=bool)
        else:
            quality_gate_values = np.asarray(quality_gate_pass, dtype=bool).reshape(n)
        for index in range(n):
            state = self.push_sample(
                timestamps[index], emg[index], accel[index], gyro[index],
                interpolated_imu=bool(interpolated_values[index]),
                sample_quality=float(sample_quality_values[index]),
                missing=bool(missing_values[index]),
                channel_quality=channel_quality_values[index],
                timestamp_valid=bool(timestamp_values[index]),
                imu_valid=bool(imu_valid_values[index]),
                quality_gate_pass=bool(quality_gate_values[index]),
            )
            if state is not None:
                states.append(state)
        return states

    def _estimate(self) -> HumanState:
        timestamp_ms = self.timestamps[-1]
        raw_emg = np.stack(self.raw_emg)
        filtered_emg = np.stack(self.filtered_emg)
        raw_imu = np.stack(self.raw_imu)
        normalized_emg = self.calibration.transform_emg(filtered_emg)
        safe_imu = np.nan_to_num(raw_imu, nan=0.0, posinf=0.0, neginf=0.0)
        normalized_imu = self.calibration.transform_imu(safe_imu[:, :3], safe_imu[:, 3:])
        emg_quality, imu_quality = estimate_signal_quality(
            raw_emg, raw_imu, expected_samples=self.window_samples,
        )
        sample_quality = float(np.mean(self.sample_quality))
        missing_ratio = float(np.mean(self.missing))
        imu_valid_ratio = float(np.mean(self.imu_valid))
        channel_quality = np.mean(np.stack(self.channel_quality), axis=0)
        valid_channels = int(np.count_nonzero(channel_quality >= 0.5))
        timestamps_valid = bool(all(self.timestamp_valid))
        gate_pass = bool(all(self.quality_gate_pass))
        emg_quality *= sample_quality * (1.0 - missing_ratio) * valid_channels / 8.0
        imu_quality *= sample_quality * (1.0 - missing_ratio) * imu_valid_ratio
        if valid_channels < 6:
            emg_quality = 0.0
        if not timestamps_valid or not gate_pass:
            emg_quality = 0.0
            imu_quality = 0.0
        flags = QualityFlag.CALIBRATED
        if timestamps_valid:
            flags |= QualityFlag.TIMESTAMP_VALID
        if emg_quality >= self.quality_threshold:
            flags |= QualityFlag.EMG_VALID
        if imu_quality >= self.quality_threshold:
            flags |= QualityFlag.IMU_VALID
        if any(self.interpolated_imu):
            flags |= QualityFlag.INTERPOLATED_IMU
        if any(self.duplicate_packet):
            flags |= QualityFlag.DUPLICATE_PACKET
        if any(self.out_of_order_packet):
            flags |= QualityFlag.OUT_OF_ORDER_PACKET
        if not gate_pass:
            flags |= QualityFlag.QUALITY_GATE_FAILED
        if valid_channels < 6:
            flags |= QualityFlag.BAD_CHANNELS
        if self.dropped_samples:
            flags |= QualityFlag.DROPPED_SAMPLES
            self.dropped_samples = False
        quality = SignalQuality(emg_quality, imu_quality, flags)

        prediction = self.predictor.predict(normalized_emg, normalized_imu)
        if prediction.direction_confidence_rejected or prediction.gesture_confidence_rejected:
            flags |= QualityFlag.LOW_CONFIDENCE
        if prediction.direction_drift_rejected or prediction.gesture_drift_rejected:
            flags |= QualityFlag.MODEL_DRIFT
        if prediction.direction_disagreement_rejected or prediction.gesture_disagreement_rejected:
            flags |= QualityFlag.MODEL_DISAGREEMENT
        direction_calibration_valid = not self.calibration.body_rotation_fitted or (
            self.calibration.body_rotation_residual <= self.max_body_rotation_residual
        )
        gesture_calibration_valid = not self.calibration.emg_channel_shift_evaluated or (
            self.calibration.emg_channel_shift_confident
        )
        if not direction_calibration_valid or not gesture_calibration_valid:
            flags |= QualityFlag.CALIBRATION_UNCERTAIN
        quality = SignalQuality(emg_quality, imu_quality, flags)
        raw_direction = (
            prediction.direction
            if imu_quality >= self.quality_threshold and direction_calibration_valid
            else Direction.UNKNOWN
        )
        raw_gesture = (
            prediction.gesture
            if emg_quality >= self.quality_threshold and gesture_calibration_valid
            else Gesture.UNKNOWN
        )
        arm = self.arm_phase.update(raw_direction, prediction.q_direction, valid=raw_direction != Direction.UNKNOWN)
        hand = self.hand_phase.update(raw_gesture, prediction.q_gesture, valid=raw_gesture != Gesture.UNKNOWN)
        direction = Direction(arm.stable_label)
        gesture = Gesture(hand.stable_label)

        if arm.phase == Phase.ONSET:
            self.last_arm_onset_ms = timestamp_ms
        if hand.phase == Phase.ONSET:
            self.last_hand_onset_ms = timestamp_ms
        onset_lag: float | None = None
        if self.last_arm_onset_ms is not None and self.last_hand_onset_ms is not None:
            # Canonical sign: hand onset minus arm onset.  Negative means hand first.
            candidate_lag = float(self.last_hand_onset_ms - self.last_arm_onset_ms)
            if abs(candidate_lag) <= 600.0:
                onset_lag = candidate_lag
        self.current_onset_lag_ms = onset_lag

        activation = activation_intensity(normalized_emg)
        motion = motion_intensity(normalized_imu)
        consistency = Consistency.UNKNOWN
        consistency_score = None
        if self.consistency_model is not None:
            consistency, consistency_score = self.consistency_model.predict(
                state_key(direction, gesture, arm.phase, hand.phase),
                activation, motion, onset_lag,
            )
        arm_q = prediction.q_direction * imu_quality if arm.phase != Phase.UNKNOWN else 0.0
        hand_q = prediction.q_gesture * emg_quality if hand.phase != Phase.UNKNOWN else 0.0
        return HumanState(
            timestamp_ms=timestamp_ms,
            direction=direction,
            gesture=gesture,
            activation=activation,
            phase=PhasePair(arm.phase, hand.phase),
            motion_intensity=motion,
            consistency=consistency,
            confidence=Confidence(
                prediction.q_direction,
                prediction.q_gesture,
                arm_q,
                hand_q,
            ),
            signal_quality=quality,
            consistency_score=consistency_score,
            direction_margin=getattr(prediction, "direction_margin", 0.0),
            gesture_margin=getattr(prediction, "gesture_margin", 0.0),
            direction_drift=getattr(prediction, "direction_drift", 0.0),
            gesture_drift=getattr(prediction, "gesture_drift", 0.0),
        )
