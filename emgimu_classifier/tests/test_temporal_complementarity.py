import unittest
import numpy as np
from emgimu.feature_bank.unibo_temporal_complementarity import complementarity


class WeightedErrorTests(unittest.TestCase):
    def test_weighted_asymmetric_errors(self):
        y=np.array([0,0,1,1]);a=np.eye(2)[[0,1,0,1]];b=np.eye(2)[[1,0,0,1]]
        result=complementarity(y,a,b,np.array([1.,2.,3.,4.]))
        self.assertAlmostEqual(result['a_correct_b_wrong'],.1)
        self.assertAlmostEqual(result['a_wrong_b_correct'],.2)
        self.assertAlmostEqual(result['both_wrong'],.3)
        self.assertAlmostEqual(result['both_correct'],.4)
        self.assertAlmostEqual(result['disagreement_rate'],.3)

    def test_constant_error_correlation_is_unavailable(self):
        y=np.array([0,1]);p=np.eye(2)
        self.assertEqual(complementarity(y,p,p,np.ones(2))['error_correlation'],'N/A')


if __name__=='__main__':unittest.main()
