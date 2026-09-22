import unittest

from benchmarks.epn_shortlist_final_replay import disjoint_trials


class EpnShortlistFinalReplayTests(unittest.TestCase):
    def test_trial_overlap_between_any_phase_is_rejected(self):
        for train, validation, final in ((['a'], ['a'], ['c']),
                                         (['a'], ['b'], ['a']),
                                         (['a'], ['b'], ['b'])):
            with self.subTest(train=train, validation=validation, final=final):
                with self.assertRaisesRegex(ValueError, 'overlap'):
                    disjoint_trials(train, validation, final)

    def test_duplicate_trial_identity_within_phase_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            disjoint_trials(['a', 'a'], ['b'], ['c'])

    def test_valid_three_way_split(self):
        disjoint_trials(['a'], ['b'], ['c'])
