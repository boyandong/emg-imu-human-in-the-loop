import pickle
import unittest
import numpy as np
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.session_shift_summary import FamilySessionShiftSummary,affine_covariance_distance
from benchmarks.song_f8_rest_noise_shift import trial_balanced_noise


class SessionShiftTests(unittest.TestCase):
    def test_rest_noise_gives_equal_mass_to_unequal_trial_window_counts(self):
        first=np.asarray([0.,1.,2.,3.])[:,None]
        second=3*first
        batch=FeatureBatch(np.stack([first,first,first,second]),1000.)
        actual=trial_balanced_noise(batch,np.asarray(['neutral']*4),
                                    np.asarray(['a','a','a','b']),'neutral')
        np.testing.assert_allclose(actual,[2.])

    def test_rest_noise_shift_is_numeric_and_source_frozen(self):
        rng=np.random.default_rng(48)
        source=rng.normal(size=(8,200,4))
        labels=np.asarray(['neutral']*4+['fist']*4)
        trials=np.asarray([f'source-{i}' for i in range(8)])
        fitted=FamilySessionShiftSummary().fit_long_term(
            FeatureBatch(source,1000.),labels,trials,rest_label='neutral')
        before=pickle.dumps(fitted)
        calibration=source.copy();calibration[:4]*=2.
        result=fitted.from_calibration(FeatureBatch(calibration,1000.),labels,
                                       np.asarray([f'cal-{i}' for i in range(8)]))
        self.assertTrue(result['rest_noise_shift_available'])
        np.testing.assert_allclose(result['rest_noise_log_ratio_per_channel'],
                                   np.full(4,np.log(2.)),atol=1e-12)
        self.assertAlmostEqual(result['rest_noise_shift_norm'],2*np.log(2.))
        self.assertEqual(before,pickle.dumps(fitted))

    def test_affine_distance_has_known_geometry_and_rejects_non_spd(self):
        self.assertAlmostEqual(affine_covariance_distance(np.eye(2),np.diag([2.,.5])),np.sqrt(2)*np.log(2))
        with self.assertRaises(ValueError):affine_covariance_distance(np.eye(2),np.zeros((2,2)))

    def test_global_gain_is_separated_from_pattern_and_covariance_shift(self):
        rng=np.random.default_rng(47);x=rng.normal(size=(8,200,4));y=[1]*4+[2]*4;trials=[str(i) for i in range(8)]
        profile=FamilySessionShiftSummary().fit_long_term(FeatureBatch(x,1000.),y,trials);before=pickle.dumps(profile)
        result=profile.from_calibration(FeatureBatch(3*x,1000.),y,trials)
        for row in result['classes'].values():
            self.assertAlmostEqual(row['log_global_activation_shift'],np.log(3),places=7)
            self.assertAlmostEqual(row['mean_absolute_log_band_residual'],np.log(9),places=6)
            self.assertLess(row['scale_pattern_residual_norm'],1e-7)
            self.assertLess(row['affine_invariant_trace_covariance_distance'],1e-7)
            self.assertIsNone(row['ring_vector_residual_norm'])
        self.assertFalse(result['rest_noise_shift_available']);self.assertEqual(before,pickle.dumps(profile))
