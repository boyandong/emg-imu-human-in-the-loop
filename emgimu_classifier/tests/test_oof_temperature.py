import unittest
import numpy as np
from emgimu.feature_bank.force_nested_oof import fit_temperature, temperature_probability


class TemperatureTests(unittest.TestCase):
    def test_calibration_softens_confident_errors_without_changing_class_order(self):
        p=np.array([[.99,.01],[.99,.01],[.01,.99],[.01,.99]])
        y=np.array([0,1,0,1]);temp=fit_temperature(p,y);q=temperature_probability(p,temp)
        self.assertGreater(temp,1)
        self.assertLess(-np.log(q[np.arange(4),y]).mean(),-np.log(p[np.arange(4),y]).mean())
        np.testing.assert_array_equal(p.argmax(1),q.argmax(1))
        np.testing.assert_allclose(q.sum(1),1)

    def test_invalid_probability_and_temperature_are_rejected(self):
        for p,t in (([[.2,.2]],1),([[np.nan,.5]],1),([[.5,.5]],0)):
            with self.assertRaises(ValueError):temperature_probability(p,t)
