import pickle
import unittest
import numpy as np
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.quality_observability import QualityObservabilityFamily,longest_flatline_ratio


class QualityObservabilityTests(unittest.TestCase):
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
