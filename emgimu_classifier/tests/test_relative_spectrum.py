import pickle
import unittest
import numpy as np
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.relative_spectrum import (
    LogBandEnergyFamily, RelativeSpectrumCoordinates, PersonalSessionSpectralShift,
)


class RelativeSpectrumTests(unittest.TestCase):
    def test_known_gain_has_expected_log_power_shift_with_frozen_reference(self):
        rng=np.random.default_rng(23);x=rng.normal(size=(5,200,8))
        family=LogBandEnergyFamily().fit(FeatureBatch(x,1000.))
        a=family.transform(FeatureBatch(x,1000.));b=family.transform(FeatureBatch(2*x,1000.))
        np.testing.assert_allclose(b-a,np.log(4),atol=1e-6)
        coordinates=RelativeSpectrumCoordinates().fit_calibration(a);before=pickle.dumps(coordinates)
        np.testing.assert_allclose(coordinates.transform(b)-coordinates.transform(a),np.log(4),atol=1e-6)
        query=coordinates.transform(b[:1]);np.testing.assert_array_equal(query,coordinates.transform(b)[:1])
        self.assertEqual(before,pickle.dumps(coordinates));self.assertEqual(len(family.feature_names),32)

    def test_rate_mismatch_and_missing_reference_are_rejected(self):
        batch=FeatureBatch(np.zeros((2,40,4)),200.)
        family=LogBandEnergyFamily().fit(batch)
        with self.assertRaises(ValueError):family.transform(FeatureBatch(batch.emg,500.))
        with self.assertRaises(RuntimeError):RelativeSpectrumCoordinates().transform(np.zeros((2,16)))

    def test_f4d_separates_long_session_and_evaluation_trials(self):
        long=np.array([[1.,2.],[1.,2.],[3.,4.]])
        profile=PersonalSessionSpectralShift().fit_long_term(long,['L1','L1','L2'])
        np.testing.assert_array_equal(profile.long_reference_,[2.,3.])
        before=pickle.dumps(profile)
        with self.assertRaises(ValueError):profile.fit_session_calibration([[5.,6.]],['L1'])
        profile.fit_session_calibration([[5.,6.],[5.,6.]],['S1','S1'])
        with self.assertRaises(ValueError):profile.transform_evaluation([[7.,8.]],['S1'])
        output=profile.transform_evaluation([[7.,8.]],['E1'])
        np.testing.assert_array_equal(output['session_minus_long'],[3.,3.])
        np.testing.assert_array_equal(output['window_minus_long'],[[5.,5.]])
        self.assertNotEqual(before,pickle.dumps(profile))
