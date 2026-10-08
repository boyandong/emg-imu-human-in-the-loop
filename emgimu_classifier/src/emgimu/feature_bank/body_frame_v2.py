"""Explicit SI conversion and strict provenance for calibration-relative F6."""
import numpy as np
from .body_frame import CalibratedBodyContextFamily
from .core import FeatureBatch


class CalibratedBodyContextV2(CalibratedBodyContextFamily):
    family_id = 'F6_calibration_body_context_si_v2'

    def __init__(self, *, imu_sample_rate_hz, acceleration_unit, angular_velocity_unit,
                 gravity_time_constant_s=.5):
        acceleration = {'m/s^2':1., 'g':9.80665}
        rotation = {'rad/s':1., 'deg/s':np.pi/180.}
        if acceleration_unit not in acceleration or angular_velocity_unit not in rotation:
            raise ValueError('Known acceleration m/s^2 or g and gyro rad/s or deg/s units required')
        self.input_acceleration_unit = acceleration_unit
        self.input_angular_velocity_unit = angular_velocity_unit
        self.input_scale = np.array([acceleration[acceleration_unit]]*3+[rotation[angular_velocity_unit]]*3)
        self.input_scale.setflags(write=False)
        super().__init__(imu_sample_rate_hz=imu_sample_rate_hz, acceleration_unit='m/s^2',
            angular_velocity_unit='rad/s', gravity_time_constant_s=gravity_time_constant_s)

    @staticmethod
    def _ids(values):
        ids = tuple(values)
        if not ids or any(not isinstance(v,str) or not v.strip() for v in ids):
            raise ValueError('Explicit nonempty string trial provenance required')
        return ids

    def _si_batch(self,batch):
        if batch.imu is None:
            raise ValueError('Raw synchronized accel/gyro required')
        return FeatureBatch(batch.emg,batch.sample_rate_hz,
                            np.asarray(batch.imu,dtype=float)*self.input_scale,batch.posture)

    def fit(self,batch,labels=None,*,neutral_calibration_imu,forward_axis_device,calibration_trial_ids):
        ids = self._ids(calibration_trial_ids)
        if len(set(ids)) != len(ids):
            raise ValueError('Unique frame calibration trial identities required')
        neutral = np.asarray(neutral_calibration_imu,dtype=float)
        if neutral.ndim != 2 or neutral.shape[1] != 6:
            raise ValueError('Neutral calibration must contain raw accel/gyro six columns')
        return super().fit(self._si_batch(batch),labels, neutral_calibration_imu=neutral*self.input_scale,
                           forward_axis_device=forward_axis_device,calibration_trial_ids=ids)

    def transform(self,batch,*,trial_ids,evaluation=True):
        ids = self._ids(trial_ids)
        return super().transform(self._si_batch(batch),trial_ids=ids,evaluation=evaluation)
