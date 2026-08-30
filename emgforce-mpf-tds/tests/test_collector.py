import unittest

import numpy as np

from mpf_tds.collector import ActivityDetectorConfig, AdaptiveActivityDetector


class AdaptiveActivityDetectorTest(unittest.TestCase):
    def test_detects_user_paced_bursts_across_chunk_boundaries(self) -> None:
        rng = np.random.default_rng(7)
        config = ActivityDetectorConfig(
            sample_rate_hz=1000,
            channels=8,
            baseline_seconds=0.25,
            envelope_window_ms=10,
            onset_hold_ms=8,
            quiet_hold_ms=30,
            refractory_ms=20,
            onset_sigma=8,
            offset_sigma=3,
            minimum_scale=1e-5,
        )
        detector = AdaptiveActivityDetector(config)
        samples = np.concatenate([
            rng.normal(0, 0.005, (300, 8)),
            rng.normal(0, 0.2, (100, 8)),
            rng.normal(0, 0.005, (180, 8)),
            rng.normal(0, 0.15, (120, 8)),
            rng.normal(0, 0.005, (180, 8)),
        ]).astype(np.float32)
        timestamps = np.arange(len(samples), dtype=np.float64) / 1000.0

        events = []
        boundaries = (0, 137, 421, 700, len(samples))
        for start, end in zip(boundaries, boundaries[1:]):
            events.extend(detector.process(samples[start:end], timestamps[start:end]))

        self.assertTrue(detector.calibrated)
        self.assertEqual(len(events), 2)
        self.assertAlmostEqual(events[0].onset_time, 0.300, places=3)
        self.assertAlmostEqual(events[1].onset_time, 0.580, places=3)
        self.assertGreater(events[0].peak_envelope, events[1].peak_envelope)


if __name__ == "__main__":
    unittest.main()
