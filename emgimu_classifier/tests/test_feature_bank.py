from __future__ import annotations

import unittest

import numpy as np

from emgimu.feature_bank import (
    BodyContextFamily,
    CspSpatialFamily,
    FeatureBatch,
    LocalDetailFamily,
    PathSignatureFamily,
    PersonalAnchor,
    PersonalNormalizer,
    QualityFamily,
    ReliabilityWeights,
    RingGeometryFamily,
    ScalePatternFamily,
    SessionSignature,
    SpdTangentFamily,
    SpectralStateFamily,
    TemporalFormFamily,
    TemporalTemplateFamily,
    TraceCovarianceFamily,
    default_registry,
    late_fusion,
)


def make_batch(seed: int = 3, windows: int = 16, samples: int = 48, channels: int = 8) -> FeatureBatch:
    random = np.random.default_rng(seed)
    emg = random.normal(size=(windows, samples, channels))
    imu = random.normal(size=(windows, samples, 6))
    posture = np.asarray(["down", "forward", "up", "forward"] * (windows // 4))
    return FeatureBatch(emg, 200.0, imu, posture)


class FeatureFamilyTests(unittest.TestCase):
    def test_registry_has_explicit_family_candidates(self) -> None:
        ids = default_registry().family_ids
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue({"F0_local_detail", "F1_scale_pattern_x1h", "F2a_trace_covariance", "F2b_csp", "F2c_spd_tangent", "F3_ring_geometry", "F4_spectral_state", "F5_temporal_form", "F5b_dtw_templates", "F5c_path_signature_order2", "F6_body_context", "F9_quality"}.issubset(ids))

    def test_all_window_families_are_finite_and_named(self) -> None:
        batch = make_batch()
        labels = np.arange(batch.windows) % 4
        families = [
            LocalDetailFamily(), ScalePatternFamily(), TraceCovarianceFamily(),
            CspSpatialFamily(), SpdTangentFamily(), RingGeometryFamily(),
            SpectralStateFamily(), TemporalFormFamily(), PathSignatureFamily(),
            BodyContextFamily(), QualityFamily(adc_min=-4, adc_max=4),
        ]
        for family in families:
            output = family.fit_transform(batch, labels)
            self.assertEqual(output.shape, (batch.windows, len(family.feature_names)), family.family_id)
            self.assertTrue(np.all(np.isfinite(output)), family.family_id)

    def test_zero_constant_and_tiny_inputs_remain_finite(self) -> None:
        for magnitude in (0.0, 1e-30, 2.0):
            batch = FeatureBatch(np.full((8, 40, 8), magnitude), 200.0)
            labels = np.arange(8) % 4
            for family in (LocalDetailFamily(), ScalePatternFamily(), TraceCovarianceFamily(), SpdTangentFamily(), RingGeometryFamily(), SpectralStateFamily(), TemporalFormFamily(), QualityFamily()):
                result = family.fit_transform(batch, labels)
                self.assertTrue(np.all(np.isfinite(result)), (family.family_id, magnitude))

    def test_x1h_removes_global_scale(self) -> None:
        batch = make_batch()
        family = ScalePatternFamily().fit(batch)
        scaled = FeatureBatch(batch.emg * 17.0, batch.sample_rate_hz)
        np.testing.assert_allclose(family.transform(batch), family.transform(scaled), atol=1e-6)

    def test_ring_family_is_rotation_invariant_not_permutation_invariant(self) -> None:
        batch = make_batch()
        family = RingGeometryFamily().fit(batch)
        base = family.transform(batch)
        rotated = family.transform(FeatureBatch(np.roll(batch.emg, 2, axis=2), 200.0))
        permuted = family.transform(FeatureBatch(batch.emg[:, :, [0, 2, 1, 3, 4, 5, 6, 7]], 200.0))
        np.testing.assert_allclose(base, rotated, atol=1e-6)
        self.assertGreater(float(np.max(np.abs(base - permuted))), 1e-5)

    def test_validation_transform_does_not_change_fit_state(self) -> None:
        train, validation = make_batch(1), make_batch(2)
        labels = np.arange(train.windows) % 4
        families = [LocalDetailFamily(), CspSpatialFamily(), SpdTangentFamily(), QualityFamily()]
        for family in families:
            family.fit(train, labels)
            before = repr(family.__dict__)
            family.transform(validation)
            self.assertEqual(before, repr(family.__dict__), family.family_id)

    def test_body_context_rejects_unseen_posture(self) -> None:
        train = make_batch()
        family = BodyContextFamily().fit(train)
        invalid = FeatureBatch(train.emg[:1], 200.0, train.imu[:1], np.asarray(["unknown"]))
        with self.assertRaisesRegex(ValueError, "unknown posture"):
            family.transform(invalid)


class CalibrationTests(unittest.TestCase):
    def test_personal_normalizer_uses_only_explicit_calibration(self) -> None:
        batch = make_batch(windows=16)
        labels = np.arange(16) % 4
        normalizer = PersonalNormalizer(rest_label=0).fit(batch, labels)
        center = normalizer.center_.copy()
        scale = normalizer.scale_.copy()
        normalizer.transform(FeatureBatch(batch.emg + 1e6, 200.0))
        np.testing.assert_array_equal(center, normalizer.center_)
        np.testing.assert_array_equal(scale, normalizer.scale_)

    def test_personal_anchor_dimensions_and_no_test_update(self) -> None:
        random = np.random.default_rng(5)
        calibration = random.normal(size=(20, 6))
        labels = np.repeat(np.arange(4), 5)
        anchor = PersonalAnchor().fit(calibration, labels)
        prototypes = anchor.prototypes_.copy()
        transformed = anchor.transform(random.normal(size=(7, 6)))
        self.assertEqual(transformed.shape, (7, 10))
        np.testing.assert_array_equal(prototypes, anchor.prototypes_)

    def test_reliability_shrinkage_and_quality_fusion(self) -> None:
        labels = np.repeat(np.arange(4), 2)
        calibration = {
            "a": (np.eye(8, 4), labels),
            "b": (np.tile(np.arange(8)[:, None], (1, 4)), labels),
        }
        reliability = ReliabilityWeights((0, 1, 2, 3), ("a", "b"), np.asarray([0.5, 0.5]), n0=8)
        weights = reliability.personal(calibration)
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        probabilities = {"a": np.tile([0.7, 0.1, 0.1, 0.1], (3, 1)), "b": np.tile([0.1, 0.1, 0.1, 0.7], (3, 1))}
        fused = late_fusion(probabilities, ("a", "b"), weights, {"a": np.ones(3), "b": np.zeros(3)})
        np.testing.assert_allclose(fused, probabilities["a"])

    def test_session_signature_requires_all_classes(self) -> None:
        random = np.random.default_rng(8)
        labels = np.repeat(np.arange(4), 3)
        signature = SessionSignature().fit_long_term(random.normal(size=(12, 5)), labels)
        vector = signature.from_session_calibration(random.normal(size=(12, 5)), labels)
        self.assertEqual(len(vector), len(signature.feature_names))
        with self.assertRaisesRegex(ValueError, "every"):
            signature.from_session_calibration(random.normal(size=(9, 5)), np.repeat(np.arange(3), 3))

    def test_all_rejected_quality_falls_back_to_population(self) -> None:
        probabilities = {"a": np.asarray([[0.8, 0.2]]), "b": np.asarray([[0.2, 0.8]])}
        result = late_fusion(probabilities, ("a", "b"), np.asarray([0.75, 0.25]),
                             {"a": np.zeros(1), "b": np.zeros(1)})
        np.testing.assert_allclose(result, [[0.65, 0.35]])
        for weights in (np.zeros(2), np.asarray([np.nan, 1.0])):
            with self.assertRaises(ValueError):
                late_fusion(probabilities, ("a", "b"), weights)

    def test_unavailable_family_is_skipped_and_weights_renormalized(self) -> None:
        p = np.asarray([[0.8, 0.2], [0.3, 0.7]])
        np.testing.assert_allclose(late_fusion({'a': p}, ('a', 'b'), np.asarray([0.4, 0.6])), p)
        np.testing.assert_allclose(late_fusion({'a': p}, ('a', 'b'), np.asarray([0., 1.])), p)
        reliability = ReliabilityWeights((0, 1), ('a', 'b'), np.asarray([0.4, 0.6]))
        weights = reliability.personal({'a': (np.asarray([[0., 0.], [1., 1.]]), np.asarray([0, 1]))})
        np.testing.assert_allclose(weights, [1., 0.])
        with self.assertRaisesRegex(ValueError, 'no family'):
            reliability.personal({})

    def test_dtw_templates_use_calibration_only(self) -> None:
        batch = make_batch(windows=8, samples=20, channels=4)
        labels = np.repeat(np.arange(4), 2)
        family = TemporalTemplateFamily().fit(batch, labels)
        templates = [item.copy() for item in family.templates_]
        result = family.transform(FeatureBatch(batch.emg * 2.0, 200.0))
        self.assertEqual(result.shape, (8, 4))
        for before, after in zip(templates, family.templates_):
            np.testing.assert_array_equal(before, after)


if __name__ == "__main__":
    unittest.main()
