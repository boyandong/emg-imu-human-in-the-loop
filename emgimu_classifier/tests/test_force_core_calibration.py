import unittest
import numpy as np
from emgimu.feature_bank.force_core_calibration import choose


class WholeTrialBudgetTests(unittest.TestCase):
    def test_budget_selection_is_nested_and_own_user_only(self):
        y=np.tile(np.repeat(np.arange(7),4),2);u=np.repeat([7,8],28)
        trials=np.array([f't{i}' for i in range(56)])
        one=choose(y,u,trials,7,1);two=choose(y,u,trials,7,2)
        self.assertEqual(len(one),7);self.assertEqual(len(two),14)
        self.assertTrue(set(one)<=set(two));self.assertTrue(np.all(u[two]==7))
        self.assertEqual(len(choose(y,u,trials,7,0)),0)
        with self.assertRaisesRegex(ValueError,'supported'):choose(y,u,trials,7,5)

    def test_repeated_calibration_trial_names_are_rejected(self):
        y=np.arange(7);u=np.repeat(7,7);trials=np.repeat('same_trial',7)
        with self.assertRaisesRegex(AssertionError,'Repeated'):
            choose(y,u,trials,7,1)


if __name__=='__main__':unittest.main()
