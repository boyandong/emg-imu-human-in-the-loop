import unittest
import numpy as np
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.activation_profile import PersonalActivationProfile


class ActivationProfileTests(unittest.TestCase):
    def test_rest_exclusion_and_equal_trial_mass_survive_window_duplication(self):
        x=np.stack([np.full((40,8),v) for v in (100.,1.,2.,8.)])
        labels=np.array([0,1,1,2]);trials=np.array(['rest','a','a','b'])
        a=PersonalActivationProfile().fit_calibration(FeatureBatch(x,200.),labels,trials,rest_label=0).profile_
        repeat=np.array([0,1,2,1,2,3])
        b=PersonalActivationProfile().fit_calibration(FeatureBatch(x[repeat],200.),labels[repeat],trials[repeat],rest_label=0).profile_
        self.assertEqual(a['rest_windows_excluded'],1)
        for key in ('q10','q50','q90','pattern_spread','pattern_mean'):
            np.testing.assert_allclose(a['groups']['ALL_ACTIVE'][key],b['groups']['ALL_ACTIVE'][key])
        self.assertEqual(a['groups']['ALL_ACTIVE']['q90'],8.)

    def test_rest_only_and_mixed_label_trials_are_rejected(self):
        batch=FeatureBatch(np.zeros((2,40,8)),200.)
        with self.assertRaises(ValueError):PersonalActivationProfile().fit_calibration(batch,[0,0],['a','b'],rest_label=0)
        with self.assertRaises(ValueError):PersonalActivationProfile().fit_calibration(batch,[0,1],['a','a'],rest_label=0)

    def test_between_gesture_separation_is_not_reported_as_within_gesture_spread(self):
        x=np.zeros((2,40,8));x[0,:,0]=2.;x[1,:,1]=5.
        profile=PersonalActivationProfile().fit_calibration(FeatureBatch(x,200.),[1,2],['a','b'],rest_label=0).profile_
        self.assertEqual(profile['groups']['ALL_ACTIVE']['pattern_spread'],0.)
        self.assertGreater(profile['groups']['ALL_ACTIVE']['pooled_across_gestures_pattern_spread'],0.)
