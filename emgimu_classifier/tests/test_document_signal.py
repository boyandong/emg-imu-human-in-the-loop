import unittest
import pickle
import numpy as np
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.document_signal import RestNoiseLocalDetailFamily,DocumentCspFamily


class DocumentSignalTests(unittest.TestCase):
    def test_active_amplitude_cannot_set_rest_noise_thresholds(self):
        x = np.random.default_rng(20).normal(size=(8,40,8))
        y = np.array([2,2,0,0,1,1,3,3])
        first = RestNoiseLocalDetailFamily(rest_label=2).fit(FeatureBatch(x,200),y)
        changed = x.copy(); changed[y!=2] *= 10000
        second = RestNoiseLocalDetailFamily(rest_label=2).fit(FeatureBatch(changed,200),y)
        np.testing.assert_array_equal(first.thresholds_,second.thresholds_)
        with self.assertRaisesRegex(ValueError,'Rest'):
            RestNoiseLocalDetailFamily(rest_label=5).fit(FeatureBatch(x,200),y)
        before = pickle.dumps(first); first.transform(FeatureBatch(changed,200))
        self.assertEqual(before,pickle.dumps(first))

    def test_uncentered_csp_generalized_eigenproblem_and_variance_oracle(self):
        x = np.random.default_rng(23).normal(size=(16,40,8))+np.arange(8)[None,None,:]*.2
        y = np.arange(16)%4; batch = FeatureBatch(x,200)
        csp = DocumentCspFamily().fit(batch,y)
        cov = np.stack([sample.T@sample/(np.trace(sample.T@sample)+1e-10) for sample in x])
        np.testing.assert_allclose(csp.normalized_second_moments(x),cov,rtol=1e-12,atol=1e-12)
        for i,label in enumerate(csp.classes_):
            positive = cov[y==label].mean(0); total=positive+cov[y!=label].mean(0)+1e-5*np.eye(8)
            w = csp.filters_[:,i*4:(i+1)*4]
            np.testing.assert_allclose(positive@w,(total@w)*csp.eigenvalues_[i],rtol=1e-10,atol=1e-10)
        v = np.stack([np.var(sample@csp.filters_,axis=0) for sample in x])
        expected = np.log(v/(v.sum(1,keepdims=True)+1e-10)+1e-10)
        np.testing.assert_allclose(csp.transform(batch),expected,atol=2e-7,rtol=1e-6)
        self.assertEqual(len(csp.feature_names),16)
