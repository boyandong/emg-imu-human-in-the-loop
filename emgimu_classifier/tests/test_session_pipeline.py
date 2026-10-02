import pickle
import unittest
import numpy as np
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.force_nested_oof import SubjectWindows
from emgimu.feature_bank.session_pipeline import SessionCalibrationPipeline
from emgimu.feature_bank.force_full_fusion import aggregate


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

    def test_unlabeled_predictions_match_independent_trial_aggregation(self):
        pipeline=SessionCalibrationPipeline(rest_label=2,ring_topology=True).fit_long_term(data('source'))
        state=pipeline.calibrate_session(data('cal',offset=20.))
        target=data('target',offset=5.)
        before=pickle.dumps((pipeline,state))
        unlabeled,trials=pipeline.predict_unlabeled(target.batch,target.subjects,target.trials,state)
        scored,truth,offline_trials=pipeline.predict(target,state)
        np.testing.assert_array_equal(trials,offline_trials)
        np.testing.assert_array_equal(truth,np.arange(5))
        for branch in unlabeled:
            for family in unlabeled[branch]:
                np.testing.assert_allclose(unlabeled[branch][family],scored[branch][family],atol=1e-12)
        for branch,normalizer in (('source_model',pipeline.normalizer_),
                                  ('session_model',state['normalizer'])):
            normalized=SubjectWindows(normalizer.transform(target.batch),target.labels,
                                      target.subjects,target.trials)
            family=pipeline.families_['F0']
            values,_,_,expected_trials=aggregate(family.transform(normalized.batch),normalized)
            np.testing.assert_array_equal(trials,expected_trials)
            scaler,model=pipeline.models_['F0']
            expected=model.predict_proba(scaler.transform(values))
            np.testing.assert_allclose(unlabeled[branch]['F0'],expected,atol=1e-12)
        fake_truth=SubjectWindows(target.batch,np.zeros_like(target.labels),
                                  target.subjects,target.trials)
        fake,reported,_=pipeline.predict(fake_truth,state)
        np.testing.assert_array_equal(reported,np.zeros(5,int))
        for branch in unlabeled:
            for family in unlabeled[branch]:
                np.testing.assert_allclose(unlabeled[branch][family],fake[branch][family],atol=1e-12)
        self.assertEqual(before,pickle.dumps((pipeline,state)))

    def test_unlabeled_path_rejects_bad_identity_and_calibration_overlap(self):
        pipeline=SessionCalibrationPipeline(rest_label=2,ring_topology=True).fit_long_term(data('source'))
        state=pipeline.calibrate_session(data('cal'))
        target=data('target')
        with self.assertRaisesRegex(ValueError,'Calibration trials'):
            pipeline.predict_unlabeled(target.batch,target.subjects,data('cal').trials,state)
        with self.assertRaisesRegex(ValueError,'Source-fit trials'):
            pipeline.predict_unlabeled(target.batch,target.subjects,data('source').trials,state)
        blank=target.trials.astype(object);blank[0]=None
        with self.assertRaisesRegex(ValueError,'nonempty native trial'):
            pipeline.predict_unlabeled(target.batch,target.subjects,blank,state)
        with self.assertRaisesRegex(ValueError,'one fitted personal user'):
            pipeline.predict_unlabeled(target.batch,np.full(10,2),target.trials,state)

    def test_session_state_is_bound_to_source_profile(self):
        first=SessionCalibrationPipeline(rest_label=2,ring_topology=True).fit_long_term(data('source'))
        state=first.calibrate_session(data('cal'))
        target=data('target')
        second=SessionCalibrationPipeline(rest_label=2,ring_topology=True).fit_long_term(
            data('other_source',offset=10.))
        self.assertNotEqual(first.profile_id_,second.profile_id_)
        with self.assertRaisesRegex(ValueError,'different long-term profile'):
            second.predict_unlabeled(target.batch,target.subjects,target.trials,state)
        wrong_user=dict(state,user=2)
        with self.assertRaisesRegex(ValueError,'different long-term profile'):
            first.predict_unlabeled(target.batch,target.subjects,target.trials,wrong_user)
        restored=pickle.loads(pickle.dumps(first))
        self.assertEqual(restored.profile_id_,first.profile_id_)
        restored.predict_unlabeled(target.batch,target.subjects,target.trials,state)


if __name__=='__main__':unittest.main()
