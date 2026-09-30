import pickle
import warnings
import unittest
import numpy as np
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.quality_observability import QualityObservabilityFamily,longest_flatline_ratio


class QualityObservabilityTests(unittest.TestCase):
    def test_coarse_grab_frequency_grid_has_no_empty_band_warning(self):
        rng=np.random.default_rng(84)
        batch=FeatureBatch(rng.normal(size=(3,512,8)),2048.)
        family=QualityObservabilityFamily().fit(batch)
        with warnings.catch_warnings():
            warnings.simplefilter('error',RuntimeWarning)
            result=family.transform(batch)
        self.assertFalse(family.availability_['line_noise'])
        for channel in range(1,9):
            self.assertEqual(result[0,family.feature_names.index(f'F9.line_noise_ratio.ch{channel}')],0.)

    def test_unknown_mains_frequency_is_masked_even_when_50hz_is_in_band(self):
        rng=np.random.default_rng(83)
        source=FeatureBatch(rng.normal(size=(5,250,2)),250.)
        family=QualityObservabilityFamily(line_frequency_available=False).fit(source)
        time=np.arange(250)/250.
        target=np.stack((np.sin(2*np.pi*50*time),np.sin(2*np.pi*50*time)),axis=1)[None]
        before=pickle.dumps(family)
        result=family.transform(FeatureBatch(target,250.))
        for channel in (1,2):
            self.assertEqual(result[0,family.feature_names.index(f'F9.line_noise_ratio.ch{channel}')],0.)
        self.assertEqual(result[0,family.feature_names.index('F9v2.available.line_noise')],0.)
        self.assertFalse(family.availability_['line_noise'])
        self.assertEqual(before,pickle.dumps(family))

    def test_longest_run_distinguishes_identical_flatline_fractions(self):
        a=np.array([0.,0.,0.,0.,1.,2.,3.,4.])[None,:,None]
        b=np.array([0.,0.,1.,1.,2.,2.,3.,4.])[None,:,None]
        self.assertEqual(np.mean(np.diff(a,axis=1)==0),np.mean(np.diff(b,axis=1)==0))
        self.assertEqual(longest_flatline_ratio(a,[1e-10]).item(),3/8)
        self.assertEqual(longest_flatline_ratio(b,[1e-10]).item(),1/8)

    def test_unavailable_adc_and_pre_highpass_are_explicit_and_source_is_immutable(self):
        rng=np.random.default_rng(81);batch=FeatureBatch(rng.normal(size=(4,200,8)),1000.)
        family=QualityObservabilityFamily().fit(batch);before=pickle.dumps(family)
        output=family.transform(batch);np.testing.assert_array_equal(output[:,-3:],[ [0.,1.,0.]]*4)
        names=family.feature_names;columns=[i for i,n in enumerate(names) if '.low_frequency_power_ratio.' in n]
        np.testing.assert_array_equal(output[:,columns],np.zeros((4,8)))
        self.assertEqual(before,pickle.dumps(family));self.assertEqual(output.shape[1],len(names))

    def test_pre_highpass_low_frequency_contamination_increases_ratio(self):
        rng=np.random.default_rng(82);x=rng.normal(size=(4,200,8));time=np.arange(200)/1000
        family=QualityObservabilityFamily(pre_highpass_available=True).fit(FeatureBatch(x,1000.))
        names=family.feature_names;columns=[i for i,n in enumerate(names) if '.low_frequency_power_ratio.' in n]
        clean=family.transform(FeatureBatch(x,1000.))[:,columns]
        noisy=family.transform(FeatureBatch(x+5*np.sin(2*np.pi*5*time)[None,:,None],1000.))[:,columns]
        self.assertGreater(noisy.mean(),clean.mean()*10)
