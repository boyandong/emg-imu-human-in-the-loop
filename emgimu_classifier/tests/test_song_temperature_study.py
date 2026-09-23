import unittest

import numpy as np

from benchmarks.song_temperature_study import apply_temperature


class SongTemperatureStudyTest(unittest.TestCase):
    def test_temperature_preserves_argmax_and_unit_rows(self):
        rows = np.array([[.2, .6, .1, .1], [.1, .2, .3, .4]])
        for temperature in (.5, .8, 1., 1.5, 2.):
            result = apply_temperature(rows, temperature)
            np.testing.assert_array_equal(result.argmax(axis=1), rows.argmax(axis=1))
            np.testing.assert_allclose(result.sum(axis=1), 1)
        np.testing.assert_array_equal(apply_temperature(rows, 1.), rows)

    def test_invalid_probability_or_temperature_rejected(self):
        with self.assertRaises(ValueError):
            apply_temperature([[.8, .8]], 1.)
        with self.assertRaises(ValueError):
            apply_temperature([[.5, .5]], 0.)


if __name__ == '__main__':
    unittest.main()
