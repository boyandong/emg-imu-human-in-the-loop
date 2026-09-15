import pickle
import unittest
import numpy as np
from emgimu.feature_bank.session_anchor import SessionPrototypeAnchor


class SessionAnchorTests(unittest.TestCase):
    def test_budget_blend_preserves_long_profile_and_query_coordinates(self):
        x=np.array([[0.,1.],[2.,1.],[8.,3.],[10.,3.]])
        profile=SessionPrototypeAnchor().fit_long_term(x,[0,0,1,1]);before=pickle.dumps(profile)
        anchor,beta=profile.from_calibration([[4.,2.],[12.,4.]],[0,1])
        np.testing.assert_allclose(beta,[2/3,2/3])
        np.testing.assert_allclose(anchor.prototypes_,[[2.,4/3],[10.,10/3]])
        np.testing.assert_array_equal(anchor.scale_,profile.anchor_.scale_)
        a=anchor.transform([[3.,2.]])
        b=anchor.transform([[3.,2.],[100.,-10.]])
        np.testing.assert_array_equal(a,b[:1]);self.assertEqual(before,pickle.dumps(profile))

    def test_missing_class_is_rejected_and_zero_budget_preserves_profile(self):
        profile=SessionPrototypeAnchor().fit_long_term([[0.],[2.]],[0,1])
        with self.assertRaises(ValueError):profile.from_calibration([[1.]],[0])
        anchor,beta=profile.from_calibration()
        np.testing.assert_array_equal(anchor.prototypes_,profile.anchor_.prototypes_)
        np.testing.assert_array_equal(beta,[1.,1.])
