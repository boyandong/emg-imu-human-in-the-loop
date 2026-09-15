import unittest
import numpy as np
from emgimu.feature_bank.unibo_sequence_temporal import contiguous_bouts, envelope_path, bout_weights


class SequenceTemporalTests(unittest.TestCase):
    def test_separated_repetitions_are_not_joined(self):
        timeline = np.r_[np.zeros(200,int),np.ones(200,int),np.zeros(200,int)]
        self.assertEqual(contiguous_bouts(timeline),[(0,200,0),(200,400,1),(400,600,0)])

    def test_short_bouts_cannot_be_forced_into_dtw(self):
        self.assertEqual(contiguous_bouts(np.zeros(199,int)),[])
        with self.assertRaises(ValueError):
            envelope_path(np.zeros((40,4)))
        with self.assertRaises(ValueError):
            contiguous_bouts(np.zeros(200,int),minimum_samples=40)

    def test_full_envelope_uses_every_sample(self):
        signal = np.zeros((320,4));signal[-1]=10
        path = envelope_path(signal)
        self.assertEqual(path.shape,(32,4))
        self.assertGreater(float(path[-1].sum()),0)
        np.testing.assert_array_equal(path[:-1],0)

    def test_bout_multiplicity_does_not_change_user_mass(self):
        bouts = [dict(user='a',trial='a1',label=0),dict(user='a',trial='a1',label=0),
            dict(user='a',trial='a1',label=1),dict(user='b',trial='b1',label=0)]
        weights = bout_weights(bouts)
        self.assertAlmostEqual(float(weights[:3].sum()),float(weights[3]))
        self.assertAlmostEqual(float(weights[:2].sum()),float(weights[2]))


if __name__=='__main__':
    unittest.main()
