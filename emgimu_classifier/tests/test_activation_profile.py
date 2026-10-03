import unittest
import numpy as np
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.activation_profile import PersonalActivationProfile


class ActivationProfileTests(unittest.TestCase):
    def test_document_force_envelope_known_waveforms_and_within_gesture_spread(self):
        # Constant two-channel waveforms make every RMS and pattern hand-computable.
        # The two windows in trial a share one trial's mass; b and c each have one.
        x=np.stack([np.tile(v,(40,1)) for v in
                    ([100.,100.],[1.,0.],[0.,1.],[2.,0.],[0.,4.])])
        profile=PersonalActivationProfile().fit_calibration(
            FeatureBatch(x,200.),[0,1,1,1,2],['rest','a','a','b','c'],rest_label=0).profile_
        pooled=profile['groups']['ALL_ACTIVE']
        np.testing.assert_allclose([pooled[k] for k in ('q10','q50','q90')],
                                   [1/np.sqrt(2),np.sqrt(2),2*np.sqrt(2)])
        np.testing.assert_allclose(profile['groups']['1']['pattern_mean'],
                                   [3*np.sqrt(2)/4,np.sqrt(2)/4])
        self.assertAlmostEqual(profile['groups']['1']['pattern_spread'],.75)
        self.assertAlmostEqual(profile['groups']['2']['pattern_spread'],0.)
        self.assertAlmostEqual(pooled['pattern_spread'],.375)
        self.assertEqual(profile['rest_windows_excluded'],1)
        self.assertEqual(pooled['trials'],3)

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
