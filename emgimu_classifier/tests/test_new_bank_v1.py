import unittest

import numpy as np

from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.new_bank_v1 import (
    CorrelationSpectrumV1, FrequencyDirectionV1, RingLagV1, ScalePatternV1,
    new_bank_v1_registry,
)


class NewBankV1Tests(unittest.TestCase):
    def test_opt_in_registry_contains_only_new_families(self):
        registry = new_bank_v1_registry()
        self.assertEqual(set(registry.family_ids), {
            "new_v1_scale_pattern", "new_v1_ring_lag",
            "new_v1_correlation_spectrum", "new_v1_frequency_direction",
        })
        self.assertIsInstance(registry.create("new_v1_ring_lag"), RingLagV1)

    def test_scale_pattern_removes_global_gain_but_keeps_channel_pattern(self):
        rng = np.random.default_rng(31)
        signal = rng.normal(size=(4, 200, 8)) * np.arange(1, 9)
        family = ScalePatternV1().fit(FeatureBatch(signal, 1000.0))
        reference = family.transform(FeatureBatch(signal, 1000.0))
        np.testing.assert_allclose(reference, family.transform(FeatureBatch(signal * 9, 1000.0)), atol=1e-6)
        self.assertGreater(float(np.mean(reference[:, 7] - reference[:, 0])), 1.0)
        self.assertEqual(reference.shape, (4, 8))

    def test_ring_and_spectrum_have_distinct_permutation_contracts(self):
        rng = np.random.default_rng(32)
        signal = rng.normal(size=(6, 200, 8))
        signal[:, :, 1] = 0.8 * signal[:, :, 0] + 0.2 * rng.normal(size=(6, 200))
        batch = FeatureBatch(signal, 1000.0)
        rotated = FeatureBatch(np.roll(signal, 3, axis=2), 1000.0)
        swapped = FeatureBatch(signal[:, :, [0, 2, 1, 3, 4, 5, 6, 7]], 1000.0)
        ring = RingLagV1().fit(batch)
        spectrum = CorrelationSpectrumV1().fit(batch)
        self.assertEqual(ring.envelope_samples_, 25)
        np.testing.assert_allclose(ring.transform(batch), ring.transform(rotated), atol=1e-6)
        self.assertGreater(float(np.max(np.abs(ring.transform(batch) - ring.transform(swapped)))), 1e-5)
        np.testing.assert_allclose(spectrum.transform(batch), spectrum.transform(swapped), atol=1e-6)
        np.testing.assert_allclose(spectrum.transform(batch).sum(axis=1), 1.0, atol=1e-6)

    def test_frequency_direction_is_gain_invariant_and_band_specific_at_250_hz(self):
        time = np.arange(50) / 250.0
        signal = np.zeros((3, 50, 8))
        signal[:, :, 0] = np.sin(2 * np.pi * 30 * time)
        signal[:, :, 1] = np.sin(2 * np.pi * 60 * time)
        batch = FeatureBatch(signal, 250.0)
        family = FrequencyDirectionV1().fit(batch)
        vector = family.transform(batch)
        self.assertEqual(vector.shape, (3, 32))
        self.assertLess(family.band_edges_hz_[-1], 125.0)
        np.testing.assert_allclose(vector, family.transform(FeatureBatch(signal * 11, 250.0)), atol=1e-6)
        self.assertEqual(int(vector[0, :8].argmax()), 0)
        self.assertEqual(int(vector[0, 8:16].argmax()), 1)

    def test_fit_contract_and_degenerate_windows(self):
        batch = FeatureBatch(np.zeros((2, 200, 8)), 1000.0)
        for family in (ScalePatternV1(), RingLagV1(), CorrelationSpectrumV1(), FrequencyDirectionV1()):
            with self.assertRaises(RuntimeError):
                family.transform(batch)
            value = family.fit_transform(batch)
            self.assertEqual(value.shape[1], len(family.feature_names))
            self.assertTrue(np.isfinite(value).all())
            with self.assertRaisesRegex(ValueError, "sample rate"):
                family.transform(FeatureBatch(np.zeros((2, 200, 8)), 999.0))
            with self.assertRaisesRegex(ValueError, "channel count"):
                family.transform(FeatureBatch(np.zeros((2, 200, 7)), 1000.0))


if __name__ == "__main__":
    unittest.main()
