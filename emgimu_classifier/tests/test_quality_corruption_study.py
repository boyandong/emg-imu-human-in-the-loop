import unittest
import numpy as np
from emgimu.feature_bank.quality_corruption_study import corrupt


class FrozenCorruptionTests(unittest.TestCase):
    def test_source_scaled_line_noise_does_not_depend_on_test_amplitude(self):
        x=np.arange(4000,dtype=float).reshape(5,100,8)/100
        before=x.copy();source=np.arange(1,9,dtype=float)
        a=corrupt(x,1000,'line_50Hz_source_rms',source,source*3)-x
        b=corrupt(x*10,1000,'line_50Hz_source_rms',source,source*3)-x*10
        np.testing.assert_allclose(a,b,atol=1e-12)
        np.testing.assert_array_equal(x,before)

    def test_flatline_and_saturation_keep_unaffected_coordinates(self):
        x=np.arange(800,dtype=float).reshape(1,100,8)
        source=np.ones(8)
        z=corrupt(x,1000,'contiguous_half_flatline_ch3',source,source*3)
        np.testing.assert_array_equal(z[:,:50,2],np.full((1,50),x[0,0,2]))
        np.testing.assert_array_equal(z[:,50:,:],x[:,50:,:])
        np.testing.assert_array_equal(z[:,:,[0,1,3,4,5,6,7]],x[:,:,[0,1,3,4,5,6,7]])
        clipped=corrupt(x,1000,'synthetic_source_q99_saturation',source,source*3)
        self.assertTrue(np.all(np.abs(clipped)<=3))


if __name__=='__main__':unittest.main()
