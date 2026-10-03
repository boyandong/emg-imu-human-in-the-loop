import unittest

import numpy as np

from emgimu.feature_bank.cued_replay import replay_cued_bout


class CuedReplayTests(unittest.TestCase):
    def test_irregular_chunks_reconstruct_exact_native_bout(self):
        rng = np.random.default_rng(145)
        original = rng.standard_normal((507, 4)).astype(np.float32)
        replayed = replay_cued_bout(original, 'source-trial', 183)
        np.testing.assert_array_equal(replayed, original)

    def test_short_sequence_cannot_be_replayed_as_complete(self):
        with self.assertRaisesRegex(ValueError, 'at least one second'):
            replay_cued_bout(np.ones((199, 4)), 'short', 0)
