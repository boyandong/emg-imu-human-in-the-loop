import unittest
import pickle
import numpy as np
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.body_frame import CalibratedBodyContextFamily


def model(batch,neutral,axis):
    return CalibratedBodyContextFamily(imu_sample_rate_hz=50,acceleration_unit='m/s^2',
        angular_velocity_unit='rad/s').fit(batch,neutral_calibration_imu=neutral,
        forward_axis_device=axis,calibration_trial_ids=('neutral-source','guided-forward-source'))


class BodyFrameTests(unittest.TestCase):
    def test_device_rotation_preserves_calibrated_body_context(self):
        rng=np.random.default_rng(31)
        imu=rng.normal(size=(3,10,6))*.1;imu[:,:,2]+=9.81
        emg=rng.normal(size=(3,40,8));batch=FeatureBatch(emg,200,imu)
        neutral=np.zeros((50,6));neutral[:,2]=9.81;axis=np.array([1.,0.,0.])
        first=model(batch,neutral,axis)
        q,_=np.linalg.qr(rng.normal(size=(3,3)))
        if np.linalg.det(q)<0:q[:,0]*=-1
        rotated=np.concatenate([imu[:,:,:3]@q.T,imu[:,:,3:]@q.T],axis=2)
        n=np.concatenate([neutral[:,:3]@q.T,neutral[:,3:]@q.T],axis=1)
        second=model(FeatureBatch(emg,200,rotated),n,q@axis)
        before=pickle.dumps((first,second))
        trials=['target1','target2','target3']
        np.testing.assert_allclose(first.transform(batch,trial_ids=trials),second.transform(FeatureBatch(emg,200,rotated),trial_ids=trials),atol=1e-7)
        np.testing.assert_allclose(first.device_to_body_@first.device_to_body_.T,np.eye(3),atol=1e-12)
        self.assertEqual(before,pickle.dumps((first,second)))
        self.assertEqual(len(first.feature_names),15)

    def test_lowpass_linear_acceleration_and_sampling_contract(self):
        imu=np.zeros((2,10,6));imu[:,:,2]=9.81
        emg=np.ones((2,40,8));batch=FeatureBatch(emg,200,imu)
        neutral=np.zeros((50,6));neutral[:,2]=9.81
        fitted=model(batch,neutral,np.array([1,0,0]))
        trials=['target1','target2']
        stationary=fitted.transform(batch,trial_ids=trials)
        np.testing.assert_allclose(stationary[:,10:13],[[0,0,1]]*2,atol=1e-7)
        np.testing.assert_allclose(stationary[:,-2:],0,atol=1e-7)
        step=imu.copy();step[:,:,0]=1
        result=fitted.transform(FeatureBatch(emg,200,step),trial_ids=trials)
        residual=(1-fitted.filter_alpha_)**np.arange(1,11)
        np.testing.assert_allclose(result[:,-2],np.sqrt(np.mean(residual**2)),rtol=1e-6)
        with self.assertRaisesRegex(ValueError,'durations'):
            fitted.transform(FeatureBatch(emg,250,imu),trial_ids=trials)
        with self.assertRaisesRegex(ValueError,'held-out'):
            fitted.transform(batch,trial_ids=['neutral-source','target2'])
        with self.assertRaisesRegex(ValueError,'parallel'):
            model(batch,neutral,np.array([0,0,1]))
        with self.assertRaisesRegex(ValueError,'one second'):
            model(batch,neutral[:10],np.array([1,0,0]))
