import unittest
import numpy as np
from emgimu.feature_bank.calibration_study import _anchor_probability


class AnchorProbabilityTests(unittest.TestCase):
    def test_prediction_is_independent_of_other_evaluation_rows(self):
        rng=np.random.default_rng(81)
        x=rng.normal(size=(21,5));y=np.repeat(np.arange(7),3)
        query=rng.normal(size=(1,5))
        alone=_anchor_probability(x,y,query)
        together=_anchor_probability(x,y,np.concatenate((query,np.full((100,5),1e8))))
        np.testing.assert_allclose(alone,together[:1],rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(together.sum(1),1,atol=1e-12)


if __name__=='__main__':unittest.main()
