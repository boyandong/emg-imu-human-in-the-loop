import unittest

import numpy as np

from benchmarks.discovery.scripts.audit_ds2_mat_window_join import trial_mav


class Ds2MatWindowJoinTest(unittest.TestCase):
    def test_published_crop_window_and_hop(self):
        raw = np.zeros((3, 15000))
        raw[0, 3000:3375] = -2
        raw[1, 3075:3450] = 3
        result = trial_mav(raw)
        self.assertEqual(result.shape, (116, 3))
        self.assertAlmostEqual(result[0, 0], 2)
        self.assertAlmostEqual(result[1, 1], 3)
        self.assertAlmostEqual(result[-1, 0], 0)
        with self.assertRaises(ValueError):
            trial_mav(np.zeros((3, 14999)))


if __name__ == "__main__":
    unittest.main()
