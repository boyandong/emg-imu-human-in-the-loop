"""Explicit calibration-relative IMU context; no absolute yaw estimate."""
import numpy as np
from .core import FeatureBatch,FeatureFamily


class CalibratedBodyContextFamily(FeatureFamily):
    family_id = 'F6_calibration_body_context'

    def __init__(self, *, imu_sample_rate_hz, acceleration_unit, angular_velocity_unit,
                 gravity_time_constant_s=.5):
        self.imu_sample_rate_hz=float(imu_sample_rate_hz)
        self.gravity_time_constant_s=float(gravity_time_constant_s)
        if not np.isfinite(self.imu_sample_rate_hz) or self.imu_sample_rate_hz<=0:
            raise ValueError('A real finite positive IMU sampling rate is required')
        if not np.isfinite(self.gravity_time_constant_s) or self.gravity_time_constant_s<=0:
            raise ValueError('A positive source-fixed gravity time constant is required')
        if not acceleration_unit or not angular_velocity_unit:
            raise ValueError('Explicit acceleration and angular velocity units are required')
        self.acceleration_unit=str(acceleration_unit)
        self.angular_velocity_unit=str(angular_velocity_unit)

    def _validate(self,batch):
        if batch.imu is None or batch.imu.shape[1]<1 or not np.isfinite(batch.imu).all():
            raise ValueError('Finite real accel/gyro IMU samples are required')
        emg_duration=batch.emg.shape[1]/batch.sample_rate_hz
        imu_duration=batch.imu.shape[1]/self.imu_sample_rate_hz
        if not np.isclose(emg_duration,imu_duration,rtol=0,atol=1e-8):
            raise ValueError('EMG and real-rate IMU windows must have identical durations')

    def fit(self,batch:FeatureBatch,labels=None,*,neutral_calibration_imu,
            forward_axis_device,calibration_trial_ids):
        self._validate(batch)
        neutral=np.asarray(neutral_calibration_imu,dtype=float)
        if neutral.ndim!=2 or neutral.shape[1]!=6 or len(neutral)<self.imu_sample_rate_hz or not np.isfinite(neutral).all():
            raise ValueError('At least one second of explicit neutral calibration IMU is required')
        ids=tuple(str(value) for value in calibration_trial_ids)
        if not ids or any(not value for value in ids):
            raise ValueError('Calibration trial provenance is required')
        axis=np.asarray(forward_axis_device,dtype=float)
        if axis.shape!=(3,) or not np.isfinite(axis).all():
            raise ValueError('A measured/guided forward-axis calibration vector is required')
        gravity=neutral[:,:3].mean(0); magnitude=np.linalg.norm(gravity)
        if magnitude<=1e-10:
            raise ValueError('Neutral calibration cannot establish a gravity direction')
        z=gravity/magnitude
        x=axis-np.dot(axis,z)*z
        if np.linalg.norm(x)<=1e-10:
            raise ValueError('Forward calibration axis cannot be parallel to gravity')
        x/=np.linalg.norm(x); y=np.cross(z,x)
        # Rows are body basis vectors expressed in calibrated device coordinates.
        self.device_to_body_=np.stack([x,y,z])
        self.neutral_gravity_body_=self.device_to_body_@gravity
        self.calibration_trial_ids_=ids
        self.source_rate_=batch.sample_rate_hz
        self.source_shape_=batch.emg.shape[1:]
        self.source_imu_samples_=batch.imu.shape[1]
        self.filter_alpha_=1-np.exp(-1/(self.imu_sample_rate_hz*self.gravity_time_constant_s))
        self.fitted_=True
        return self

    def transform(self,batch,*,trial_ids,evaluation=True):
        self._check();self._validate(batch)
        trials=np.asarray(trial_ids).astype(str)
        if trials.shape!=(batch.windows,) or any(not value for value in trials):
            raise ValueError('Aligned trial identities are required')
        if evaluation and set(trials)&set(self.calibration_trial_ids_):
            raise ValueError('Frame calibration trials cannot become held-out evaluation')
        if batch.sample_rate_hz!=self.source_rate_ or batch.emg.shape[1:]!=self.source_shape_ or batch.imu.shape[1]!=self.source_imu_samples_:
            raise ValueError('Input sensor/window contract differs from source calibration')
        imu=np.asarray(batch.imu,dtype=float)
        accel=imu[:,:,:3]@self.device_to_body_.T
        gyro=imu[:,:,3:]@self.device_to_body_.T
        pieces=[]
        for sensor in (accel,gyro):
            norm=np.linalg.norm(sensor,axis=2)
            pieces.append(np.column_stack([norm.mean(1),norm.std(1),np.sqrt(np.mean(norm**2,axis=1)),norm.max(1),np.ptp(norm,axis=1)]))
        # Each independent window starts from source neutral gravity, not a
        # target-fitted initial state or an unspecified preceding test trial.
        current=np.broadcast_to(self.neutral_gravity_body_,(batch.windows,3)).copy()
        lowpass=np.empty_like(accel)
        for step in range(accel.shape[1]):
            current=(1-self.filter_alpha_)*current+self.filter_alpha_*accel[:,step]
            lowpass[:,step]=current
        mean_gravity=lowpass.mean(1)
        direction=mean_gravity/np.maximum(np.linalg.norm(mean_gravity,axis=1,keepdims=True),1e-10)
        linear=accel-lowpass
        movement=np.column_stack([np.sqrt(np.mean(np.sum(linear**2,axis=2),axis=1)),
                                  np.sqrt(np.mean(np.sum(gyro**2,axis=2),axis=1))])
        return np.concatenate([*pieces,direction,movement],axis=1).astype(np.float32)

    @property
    def feature_names(self):
        self._check()
        return tuple(f'F6cal.{sensor}.{metric}' for sensor in ('accel','gyro')
                     for metric in ('mean_norm','std_norm','rms_norm','max_norm','range_norm')) + (
                     'F6cal.gravity.relative_x','F6cal.gravity.relative_y','F6cal.gravity.relative_z',
                     'F6cal.linear_accel.rms','F6cal.gyro.rms')
