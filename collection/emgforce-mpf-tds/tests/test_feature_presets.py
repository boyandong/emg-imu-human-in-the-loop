import unittest

import torch

from mpf_tds.features import MPFConfig, MultiBandMatrixPowerFeatures


class MPFConfigPresetTest(unittest.TestCase):
    def test_200_hz_preset_produces_400_frames_and_192_features(self) -> None:
        config = MPFConfig.for_sample_rate(200)
        sample_count = (
            config.left_context_samples + 1
            + 399 * config.output_stride_samples
        )
        output = MultiBandMatrixPowerFeatures(config)(
            torch.randn(1, config.channels, sample_count))

        self.assertEqual(config.sample_rate_hz, 200)
        self.assertEqual(config.output_stride_samples, 4)
        self.assertEqual(tuple(output.shape), (1, 400, 192))
        self.assertTrue(torch.isfinite(output).all())

    def test_legacy_2khz_preset_remains_available(self) -> None:
        config = MPFConfig.for_sample_rate(2000)
        self.assertEqual(config.window_samples, 200)
        self.assertEqual(config.output_stride_samples, 40)
        self.assertEqual(config.fft_samples, 64)
        self.assertEqual(config.output_dim, 192)
        self.assertEqual(config.frequency_bins[-1], (687.5, 1000.0))


if __name__ == "__main__":
    unittest.main()
