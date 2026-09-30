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

    def test_f4d_native_log_bands_match_direct_fourier_and_equal_trial_reference(self):
        rate=200.;samples=64;time=np.arange(samples)/rate
        tones=np.array([25.,62.5,25.,62.5,25.,62.5,25.,62.5])
        channel_signal=np.sin(2*np.pi*time[:,None]*tones[None,:])
        windows=np.stack([channel_signal,2*channel_signal,3*channel_signal])
        batch=FeatureBatch(windows,rate)
        family=LogBandEnergyFamily().fit(batch)
        observed=family.transform(batch)
        frequencies=np.arange(samples//2+1)*rate/samples
        transform=np.exp(-2j*np.pi*np.outer(np.arange(samples//2+1),np.arange(samples))/samples)
        hann=.5-.5*np.cos(2*np.pi*np.arange(samples)/(samples-1))
        expected=[]
        for window in windows:
            detrended=(window-window.mean(0))*hann[:,None]
            power=np.abs(transform@detrended)**2/samples
            bands=[]
            for index,(low,high) in enumerate(family.bands_):
                selected=(frequencies>=low)&(frequencies<=high if index==3 else frequencies<high)
                bands.extend(np.log(power[selected].sum(0)+1e-10))
            expected.append(bands)
        np.testing.assert_allclose(observed,np.asarray(expected),rtol=1e-6,atol=1e-6)
        long=np.array([[1.,3.],[3.,5.],[11.,13.]])
        profile=PersonalSessionSpectralShift().fit_long_term(long,['L0','L0','L1'])
        np.testing.assert_array_equal(profile.long_reference_,[6.5,8.5])
        session=np.array([[2.,4.],[4.,6.],[18.,20.],[20.,22.]])
        profile.fit_session_calibration(session,['S0','S0','S0','S1'])
        np.testing.assert_allclose(profile.session_reference_,[14.,16.])
        before=pickle.dumps(profile)
        result=profile.transform_evaluation([[7.,9.]],['E0'])
        np.testing.assert_allclose(result['window_minus_long'],[[.5,.5]])
        np.testing.assert_allclose(result['session_minus_long'],[7.5,7.5])
        self.assertEqual(before,pickle.dumps(profile))
