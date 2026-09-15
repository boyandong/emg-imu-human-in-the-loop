import pickle
import unittest
import numpy as np
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.force_nested_oof import SubjectWindows
from emgimu.feature_bank.session_pipeline import SessionCalibrationPipeline


def data(prefix,offset=0.,rate=200.):
    rng=np.random.default_rng(17)
    labels=np.repeat(np.arange(5),2)
    signal=rng.normal(size=(10,40,8))*(labels[:,None,None]+1)+offset
    return SubjectWindows(FeatureBatch(signal,rate),labels,np.ones(10,int),
        np.array([f'{prefix}_{h}' for h in labels]))


class SessionPipelineTests(unittest.TestCase):
    def test_rest_and_active_scale_update_without_deleting_long_profile(self):
        source=data('source');cal=data('cal',offset=20.)
        pipeline=SessionCalibrationPipeline(rest_label=2,ring_topology=True).fit_long_term(source)
        before=pickle.dumps(pipeline)
        state=pipeline.calibrate_session(cal)
        center=np.median(cal.batch.emg[cal.labels==2].reshape(-1,8),axis=0)
        scale=np.quantile(np.abs(cal.batch.emg[cal.labels!=2]-center).reshape(-1,8),.95,axis=0)
        np.testing.assert_allclose(state['normalizer'].center_,center)
        np.testing.assert_allclose(state['normalizer'].scale_,scale)
        self.assertEqual(before,pickle.dumps(pipeline))
        self.assertIn('quality_availability',state['descriptor'])
        self.assertIn('session_signatures',state['descriptor'])
        self.assertIn('prototype_beta',state['descriptor'])

    def test_calibration_trials_and_changed_sample_rate_are_rejected(self):
        pipeline=SessionCalibrationPipeline(rest_label=2,ring_topology=True).fit_long_term(data('source'))
        cal=data('cal');state=pipeline.calibrate_session(cal)
        with self.assertRaises(ValueError):pipeline.predict(cal,state)
        with self.assertRaises(ValueError):pipeline.predict(data('source'))
        with self.assertRaises(ValueError):pipeline.predict(data('target',rate=250.))

    def test_channel_count_does_not_prove_ring_geometry(self):
        with self.assertRaises(ValueError):SessionCalibrationPipeline(rest_label=2).fit_long_term(data('source'))


if __name__=='__main__':unittest.main()
