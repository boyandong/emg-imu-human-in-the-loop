import unittest
from types import SimpleNamespace
import numpy as np
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.template_study import trial_paths


class TrialPathTests(unittest.TestCase):
    def test_paths_keep_window_order_and_requested_trial_order(self):
        values=np.broadcast_to(np.arange(1,7)[:,None,None],(6,4,2)).copy()
        data=SimpleNamespace(batch=FeatureBatch(values,200),trials=np.asarray(['a']*3+['b']*3))
        result=trial_paths(data,['b','a'])
        np.testing.assert_array_equal(result.emg[:,:,0],[[4,5,6],[1,2,3]])

    def test_unequal_window_counts_are_rejected(self):
        data=SimpleNamespace(batch=FeatureBatch(np.ones((7,4,2)),200),trials=np.asarray(['a']*3+['b']*4))
        with self.assertRaisesRegex(ValueError,'variable window'):
            trial_paths(data,['a','b'])


if __name__=='__main__':unittest.main()
