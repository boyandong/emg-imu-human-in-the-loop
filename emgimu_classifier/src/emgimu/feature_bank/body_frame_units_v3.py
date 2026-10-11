"""Unit-checked, wearing-bound body context without changing frozen F6 runs.

Anatomical forward direction and neutral IMU remain explicit inputs.  This
interface validates their contracts; it cannot certify a physical calibration.
All fifteen output coordinates use m/s^2 and rad/s, with standard gravity
9.80665 m/s^2 for the declared acceleration unit g.  No absolute yaw is inferred.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .body_frame_v2 import CalibratedBodyContextV2
from .core import FeatureBatch

STANDARD_GRAVITY = 9.80665
IMU_CHANNELS = ('ax', 'ay', 'az', 'gx', 'gy', 'gz')
ACCELERATION_SCALES = {'m/s^2': 1., 'g': STANDARD_GRAVITY}
ANGULAR_VELOCITY_SCALES = {'rad/s': 1., 'deg/s': np.pi / 180.}


def _identity(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'Explicit nonempty {name} identity required')
    return value


def _provenance(trial_ids, recording_ids, *, rows=None, unique_trials=False):
    if isinstance(trial_ids, (str, bytes)) or isinstance(recording_ids, (str, bytes)):
        raise ValueError('Trial and recording identities must be aligned sequences')
    trials, records = tuple(trial_ids), tuple(recording_ids)
    if not trials or len(trials) != len(records) or (rows is not None and len(trials) != rows):
        raise ValueError('Aligned nonempty trial and recording identities required')
    for trial, record in zip(trials, records):
        _identity(trial, 'trial'); _identity(record, 'recording')
    if unique_trials and len(trials) != len(set(trials)):
        raise ValueError('Independent frame calibration trial identities must be unique')
    if any(len({r for t, r in zip(trials, records) if t == trial}) != 1 for trial in set(trials)):
        raise ValueError('A trial cannot span different recording identities')
    return trials, records


@dataclass(frozen=True)
class ImuContractV3:
    sample_rate_hz: float
    acceleration_unit: str
    angular_velocity_unit: str
    preprocessing_id: str
    channel_order: tuple = IMU_CHANNELS

    def __post_init__(self):
        if (not isinstance(self.sample_rate_hz, (int, float, np.integer, np.floating))
                or isinstance(self.sample_rate_hz, (bool, np.bool_))
                or not np.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0):
            raise ValueError('Finite positive real IMU sampling rate required')
        if not isinstance(self.acceleration_unit, str) or self.acceleration_unit not in ACCELERATION_SCALES:
            raise ValueError('Known acceleration unit must be m/s^2 or g')
        if not isinstance(self.angular_velocity_unit, str) or self.angular_velocity_unit not in ANGULAR_VELOCITY_SCALES:
            raise ValueError('Known angular velocity unit must be rad/s or deg/s')
        _identity(self.preprocessing_id, 'IMU preprocessing')
        if self.channel_order != IMU_CHANNELS:
            raise ValueError('Explicit IMU column order must be ax,ay,az,gx,gy,gz')

    @property
    def physical_contract(self):
        return self.sample_rate_hz, self.preprocessing_id, self.channel_order

    def to_si(self, values):
        data = np.asarray(values, dtype=np.float64)
        if data.ndim not in (2, 3) or data.shape[-1] != 6 or not np.isfinite(data).all():
            raise ValueError('Finite six-column accel/gyro values required')
        scales = np.array([ACCELERATION_SCALES[self.acceleration_unit]] * 3
                          + [ANGULAR_VELOCITY_SCALES[self.angular_velocity_unit]] * 3)
        output = data * scales
        if not np.isfinite(output).all():
            raise ValueError('IMU unit conversion overflow')
        return output


class CalibratedBodyContextUnitsV3(CalibratedBodyContextV2):
    """Explicit SI conversion and per-user/per-wearing frame provenance."""
    family_id = 'F6_calibration_body_context_units_v3'

    def __init__(self, *, imu_sample_rate_hz, gravity_time_constant_s=.5):
        ImuContractV3(imu_sample_rate_hz, 'm/s^2', 'rad/s', 'frame_rate_validation')
        super().__init__(imu_sample_rate_hz=imu_sample_rate_hz,
            acceleration_unit='m/s^2', angular_velocity_unit='rad/s',
            gravity_time_constant_s=gravity_time_constant_s)

    def _check_contract(self, contract, expected=None):
        if not isinstance(contract, ImuContractV3):
            raise ValueError('Explicit checked IMU contract required on every call')
        if contract.sample_rate_hz != self.imu_sample_rate_hz:
            raise ValueError('Actual IMU rate differs from the configured frame rate')
        if expected is not None and contract.physical_contract != expected:
            raise ValueError('IMU rate/preprocessing/column contract differs from frame calibration')

    @staticmethod
    def _converted_batch(batch, contract):
        if batch.imu is None:
            raise ValueError('Real IMU samples required; cannot synthesize missing IMU')
        return FeatureBatch(batch.emg, batch.sample_rate_hz, contract.to_si(batch.imu), batch.posture)

    def fit(self, batch, labels=None, *, imu_contract, neutral_imu_contract,
            neutral_calibration_imu, forward_axis_device, calibration_trial_ids,
            calibration_recording_ids, user_id, wearing_id):
        user = _identity(user_id, 'user'); wearing = _identity(wearing_id, 'wearing')
        trials, records = _provenance(calibration_trial_ids, calibration_recording_ids, unique_trials=True)
        self._check_contract(imu_contract)
        self._check_contract(neutral_imu_contract, imu_contract.physical_contract)
        converted = self._converted_batch(batch, imu_contract)
        neutral = neutral_imu_contract.to_si(neutral_calibration_imu)
        super().fit(converted, labels, neutral_calibration_imu=neutral,
            forward_axis_device=forward_axis_device, calibration_trial_ids=trials)
        self.physical_contract_ = imu_contract.physical_contract
        self.calibration_recording_ids_ = frozenset(records)
        self.user_id_ = user; self.wearing_id_ = wearing
        return self

    def transform(self, batch, *, imu_contract, trial_ids, recording_ids,
                  user_id, wearing_id, evaluation=True):
        self._check()
        if _identity(user_id, 'user') != self.user_id_ or _identity(wearing_id, 'wearing') != self.wearing_id_:
            raise ValueError('Body frame belongs to another user or wearing; recalibration required')
        if not isinstance(evaluation, bool):
            raise ValueError('Evaluation mode must be an explicit boolean')
        trials, records = _provenance(trial_ids, recording_ids, rows=batch.windows)
        if evaluation and (set(trials).intersection(self.calibration_trial_ids_)
                           or set(records).intersection(self.calibration_recording_ids_)):
            raise ValueError('Frame calibration trials/recordings cannot become held-out evaluation')
        self._check_contract(imu_contract, self.physical_contract_)
        return super().transform(self._converted_batch(batch, imu_contract), trial_ids=trials, evaluation=evaluation)

    @property
    def feature_names(self):
        return tuple(name.replace('F6cal.', 'F6cal_units_v3.') for name in super().feature_names)

    @property
    def feature_units(self):
        self._check()
        return ('m/s^2',) * 5 + ('rad/s',) * 5 + ('dimensionless',) * 3 + ('m/s^2', 'rad/s')
