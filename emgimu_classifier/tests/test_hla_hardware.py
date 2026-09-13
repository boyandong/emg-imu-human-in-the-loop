from __future__ import annotations

import unittest

import numpy as np

from emgimu.datasets.hla_hardware import build_multirate_window, channel_metadata_matrix
from emgimu.datasets.hla_schema import ChannelSpec


class HlaHardwareTests(unittest.TestCase):
    def test_stream_availability_never_upsamples_missing_bandwidth(self):
        low = build_multirate_window(np.ones((50, 4)), 200.0)
        middle = build_multirate_window(np.ones((125, 4)), 500.0)
        high = build_multirate_window(np.ones((512, 4)), 2048.0)
        self.assertEqual(tuple(low.streams), (200,))
        self.assertEqual(tuple(middle.streams), (200, 500))
        self.assertEqual(tuple(high.streams), (200, 500, 1000))
        np.testing.assert_array_equal(low.availability, [True, False, False])
        np.testing.assert_array_equal(middle.availability, [True, True, False])
        np.testing.assert_array_equal(high.availability, [True, True, True])

    def test_multirate_lengths_preserve_window_duration(self):
        values = np.arange(512 * 3, dtype=np.float64).reshape(512, 3)
        result = build_multirate_window(values, 2048.0)
        self.assertEqual({rate: len(stream) for rate, stream in result.streams.items()}, {
            200: 50, 500: 125, 1000: 250,
        })

    def test_channel_quality_is_validated(self):
        with self.assertRaisesRegex(ValueError, "one finite value"):
            build_multirate_window(np.ones((50, 4)), 200.0, channel_quality=np.ones(3))
        with self.assertRaisesRegex(ValueError, "lie in"):
            build_multirate_window(
                np.ones((50, 4)), 200.0, channel_quality=np.asarray([1, 1, 1, 2]),
            )

    def test_metadata_encodes_missing_values_and_ring_periodicity(self):
        channels = (
            ChannelSpec("a"),
            ChannelSpec("b", position_xyz=(1, 2, 3), ring_angle_deg=90, anatomical_region="flexor"),
            ChannelSpec("c", anatomical_region="unlisted-region"),
        )
        values = channel_metadata_matrix(channels)
        self.assertEqual(values.shape, (3, 11))
        self.assertEqual(values[0, 3], 0.0)
        self.assertEqual(values[0, 6], 0.0)
        self.assertEqual(values[0, -1], 1.0)
        np.testing.assert_allclose(values[1, :4], [1, 2, 3, 1])
        np.testing.assert_allclose(values[1, 4:7], [1, 0, 1], atol=1e-6)
        self.assertEqual(values[1, 7], 1.0)
        self.assertEqual(values[2, 9], 1.0)


if __name__ == "__main__":
    unittest.main()
