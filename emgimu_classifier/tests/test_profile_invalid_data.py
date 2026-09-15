import pickle
import unittest
import numpy as np
from emgimu.feature_bank.calibration import PersonalAnchor, SessionSignature


class InvalidProfileData(unittest.TestCase):
    def test_bad_anchor_calibration_cannot_replace_valid_profile(self):
        anchor = PersonalAnchor().fit([[0., 1.], [2., 3.]], [0, 1])
        before = pickle.dumps(anchor)
        for x, y in (([[np.nan, 1.], [2., 3.]], [0, 1]),
                     ([[0., 1.], [2., 3.]], [[0], [1]]),
                     ([[0., 1.], [2., 3.]], [0, np.nan])):
            with self.assertRaises(ValueError):
                anchor.fit(x, y)
            self.assertEqual(before, pickle.dumps(anchor))

    def test_nonfinite_query_is_not_silently_replaced_by_zero_features(self):
        anchor = PersonalAnchor().fit([[0., 1.], [2., 3.]], [0, 1])
        before = pickle.dumps(anchor)
        for value in (np.nan, np.inf, -np.inf):
            with self.assertRaises(ValueError):
                anchor.transform([[value, 1.]])
            self.assertEqual(before, pickle.dumps(anchor))

    def test_bad_session_does_not_produce_plausible_signature(self):
        profile = SessionSignature().fit_long_term([[0., 1.], [2., 3.]], [0, 1])
        before = pickle.dumps(profile)
        for x, y in (([[np.nan, 1.], [2., 3.]], [0, 1]),
                     ([[0.], [2.]], [0, 1]),
                     ([[0., 1.], [2., 3.], [4., 5.]], [0, 1, 2]),
                     ([[0., 1.], [2., 3.]], [[0], [1]])):
            with self.assertRaises(ValueError):
                profile.from_session_calibration(x, y)
            self.assertEqual(before, pickle.dumps(profile))

    def test_failed_long_term_fit_preserves_existing_profile(self):
        profile = SessionSignature().fit_long_term([[0., 1.], [2., 3.]], [0, 1])
        before = pickle.dumps(profile)
        with self.assertRaises(ValueError):
            profile.fit_long_term([[np.inf, 1.], [2., 3.]], [0, 1])
        self.assertEqual(before, pickle.dumps(profile))
