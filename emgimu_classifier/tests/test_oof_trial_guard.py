import unittest
import numpy as np
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.force_nested_oof import SubjectWindows, predict


class TrialFoldGuardTests(unittest.TestCase):
    def test_known_user_trial_folds_keep_trial_leakage_guard(self):
        rng=np.random.default_rng(103)
        labels=np.repeat([0,1],4);users=np.ones(8,int)
        train=SubjectWindows(FeatureBatch(rng.normal(size=(8,40,8)),200),labels,users,np.repeat(['a','b'],4))
        held=SubjectWindows(FeatureBatch(rng.normal(size=(8,40,8)),200),labels,users,np.repeat(['c','d'],4))
        with self.assertRaisesRegex(AssertionError,'subject leakage'):predict(train,held,'F0')
        p,_,_,_,_=predict(train,held,'F0',subject_disjoint=False)
        np.testing.assert_allclose(p.sum(1),1)
        held.trials=train.trials.copy()
        with self.assertRaisesRegex(AssertionError,'trial leakage'):predict(train,held,'F0',subject_disjoint=False)
