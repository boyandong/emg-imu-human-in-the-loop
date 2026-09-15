import unittest
import numpy as np
from emgimu.feature_bank.manus_core_calibration import split


class SessionTrialSplitTests(unittest.TestCase):
    def test_own_user_nested_calibration_and_remaining_class_coverage(self):
        y=np.tile(np.repeat(np.arange(6),3),2);u=np.repeat([3,4],18);t=np.array([f't{i}' for i in range(36)])
        one,_=split(y,u,t,3,1);two,ev=split(y,u,t,3,2)
        self.assertTrue(set(one)<=set(two));self.assertFalse(set(two)&set(ev))
        self.assertEqual(len(two),12);self.assertEqual(len(ev),6)
        np.testing.assert_array_equal(np.sort(y[ev]),np.arange(6))
        self.assertTrue(np.all(u[two]==3));self.assertTrue(np.all(u[ev]==3))

    def test_duplicate_trials_and_exhaustive_budget_rejected(self):
        y=np.repeat(np.arange(6),3);u=np.repeat(3,18)
        with self.assertRaisesRegex(AssertionError,'unique whole'):
            split(y,u,np.repeat('same',18),3,1)
        with self.assertRaisesRegex(ValueError,'Unsupported'):
            split(y,u,np.arange(18),3,3)


if __name__=='__main__':unittest.main()
