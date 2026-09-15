import unittest
import numpy as np
from emgimu.feature_bank.force_condition_audit import trial_conditions


class NativeMetadataTests(unittest.TestCase):
    def test_identifier_metadata_and_label_tampering(self):
        ids=np.array(['S10_20P_C1_R1','S9_Hard_C7_R4'])
        np.testing.assert_array_equal(trial_conditions(ids,[0,6],[10,9]),['20P','Hard'])
        with self.assertRaises(ValueError):trial_conditions(ids,[0,5],[10,9])
        with self.assertRaises(ValueError):trial_conditions(ids,[0,6],[9,10])
