import unittest
import numpy as np
from emgimu.feature_bank.quality_role_ablation import role_variants


class QualityRemovalTests(unittest.TestCase):
    def test_emg_rejection_does_not_reject_real_imu_provider(self):
        p={'F0':np.array([[.9,.1]]),'F6_IMU':np.array([[.2,.8]]),'F9_Quality':np.array([[.6,.4]])}
        result=role_variants(p,tuple(p),np.zeros(1),np.zeros(1))
        np.testing.assert_allclose(result['full'],[[.4,.6]])

    def test_joint_removal_cannot_depend_on_quality_provider_or_routing(self):
        ids=('F0','F1_X1H','F9_Quality')
        p={'F0':np.array([[.9,.1]]),'F1_X1H':np.array([[.3,.7]]),'F9_Quality':np.array([[.1,.9]])}
        a=role_variants(p,ids,np.array([.2]),np.array([.1]))
        p['F9_Quality']=np.array([[.99,.01]])
        b=role_variants(p,ids,np.array([1.]),np.array([1.]))
        np.testing.assert_allclose(a['without_F9_routing_and_provider'],b['without_F9_routing_and_provider'])
        self.assertFalse(np.allclose(a['full'],b['full']))
