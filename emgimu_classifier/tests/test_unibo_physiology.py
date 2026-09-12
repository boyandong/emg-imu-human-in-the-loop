from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from emgimu.datasets.benchmark import BenchmarkTrial
from emgimu.datasets.unibo_baseline import (
    UniBoRawWindows,
    evaluate_unibo_probabilities,
    extract_unibo_emg_features,
    hierarchical_segment_weights,
)
from emgimu.datasets.unibo_physiology import (
    PhysiologyFeatureTransformer,
    chronological_fold,
    load_chronological_raw_windows,
)
from emgimu.state import Gesture


def make_raw_windows(count: int = 24, seed: int = 7) -> UniBoRawWindows:
    rng = np.random.default_rng(seed)
    labels = np.resize(np.asarray([0, 1, 2, 3], dtype=np.int16), count)
    emg = rng.uniform(0.01, 2.0, size=(count, 40, 4)).astype(np.float32)
    emg += labels[:, None, None] * np.asarray([0.1, 0.2, 0.3, 0.4])[None, None, :]
    return UniBoRawWindows(
        emg=emg,
        labels=labels,
        trial_id=np.asarray([f"trial-{index // 2}" for index in range(count)]),
        subject_id=np.asarray([f"u{index % 3 + 1:02d}" for index in range(count)]),
        session_id=np.asarray([f"d{index % 2 + 1:02d}" for index in range(count)]),
        posture=np.resize(np.asarray([1, 2, 3, 4], dtype=np.int16), count),
        timestamp_ms=np.arange(count, dtype=np.float64) * 200.0,
    )


def weights_for(raw: UniBoRawWindows) -> np.ndarray:
    from emgimu.datasets.unibo_baseline import UniBoWindows

    metadata = UniBoWindows(
        features=np.empty((len(raw), 0), dtype=np.float32), labels=raw.labels,
        trial_id=raw.trial_id, subject_id=raw.subject_id, session_id=raw.session_id,
        posture=raw.posture, timestamp_ms=raw.timestamp_ms,
    )
    return hierarchical_segment_weights(metadata)


class UniBoPhysiologyFeatureTests(unittest.TestCase):
    def test_g0_is_exactly_the_legacy_24_columns(self):
        raw = make_raw_windows(8)
        transformer = PhysiologyFeatureTransformer(("G0",)).fit(
            raw.emg, raw.labels, weights_for(raw),
        )
        actual = transformer.transform(raw.emg, raw.posture)
        expected = np.stack([extract_unibo_emg_features(window) for window in raw.emg])
        self.assertEqual(actual.shape, (8, 24))
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(len(transformer.feature_names), 24)

    def test_feature_names_dimensions_and_no_duplicate_columns(self):
        raw = make_raw_windows()
        expected = {"G0": 24, "G1": 8, "G2": 6, "G3": 10, "G4": 3, "G5": 20, "G6": 4}
        for group, dimension in expected.items():
            with self.subTest(group=group):
                transformer = PhysiologyFeatureTransformer(
                    (group,), nmf_components=3 if group == "G4" else None,
                ).fit(raw.emg, raw.labels, weights_for(raw))
                values = transformer.transform(raw.emg, raw.posture)
                self.assertEqual(values.shape, (len(raw), dimension))
                self.assertEqual(len(transformer.feature_names), dimension)
                self.assertEqual(len(set(transformer.feature_names)), dimension)
        combined = PhysiologyFeatureTransformer(
            ("G0", "G1", "G2", "G3", "G4", "G5", "G6"),
            nmf_components=2, temporal_synergy=True,
        ).fit(raw.emg, raw.labels, weights_for(raw))
        self.assertEqual(combined.transform(raw.emg, raw.posture).shape[1], 24 + 8 + 6 + 10 + 8 + 20 + 4)
        self.assertEqual(len(set(combined.feature_names)), len(combined.feature_names))

    def test_zero_constant_and_tiny_inputs_are_finite(self):
        raw = make_raw_windows()
        transformer = PhysiologyFeatureTransformer(("G0", "G1", "G2", "G3", "G5")).fit(
            raw.emg, raw.labels, weights_for(raw),
        )
        special = np.stack([
            np.zeros((40, 4)),
            np.ones((40, 4)),
            np.full((40, 4), 1e-30),
        ])
        values = transformer.transform(special, np.asarray([1, 2, 3]))
        self.assertTrue(np.isfinite(values).all())

    def test_g1_thresholds_are_fit_only_from_training_neutral(self):
        raw = make_raw_windows()
        transformer = PhysiologyFeatureTransformer(("G1",)).fit(
            raw.emg, raw.labels, weights_for(raw),
        )
        before = transformer.neutral_thresholds_.copy()
        validation = make_raw_windows(seed=99)
        validation.emg[:] = 1e6
        transformer.transform(validation.emg, validation.posture)
        np.testing.assert_array_equal(transformer.neutral_thresholds_, before)

    def test_nmf_fit_transform_contract_and_dimensions(self):
        raw = make_raw_windows()
        for components in (2, 3):
            transformer = PhysiologyFeatureTransformer(
                ("G4",), nmf_components=components,
            ).fit(raw.emg, raw.labels, weights_for(raw))
            basis = transformer.nmf_.components_.copy()
            first = transformer.transform(raw.emg[:5], raw.posture[:5])
            second = transformer.transform(raw.emg[5:10], raw.posture[5:10])
            self.assertEqual(first.shape, (5, components))
            self.assertEqual(second.shape, (5, components))
            np.testing.assert_array_equal(transformer.nmf_.components_, basis)

    def test_temporal_features_are_window_local(self):
        raw = make_raw_windows(8)
        transformer = PhysiologyFeatureTransformer(("G5",)).fit(
            raw.emg, raw.labels, weights_for(raw),
        )
        baseline = transformer.transform(raw.emg, raw.posture)
        modified = raw.emg.copy()
        modified[1:] *= 1000.0
        changed = transformer.transform(modified, raw.posture)
        np.testing.assert_array_equal(baseline[0], changed[0])

    def test_posture_mapping_is_fixed_and_unknown_is_rejected(self):
        raw = make_raw_windows(8)
        transformer = PhysiologyFeatureTransformer(("G6",)).fit(
            raw.emg, raw.labels, weights_for(raw),
        )
        values = transformer.transform(raw.emg, raw.posture)
        np.testing.assert_array_equal(values.sum(axis=1), np.ones(len(raw)))
        with self.assertRaisesRegex(ValueError, "unknown UniBo posture"):
            transformer.transform(raw.emg[:1], np.asarray([9]))


class UniBoPhysiologyEvaluationTests(unittest.TestCase):
    def test_active_macro_f1_uses_full_confusion_population(self):
        raw = make_raw_windows(5)
        raw = UniBoRawWindows(
            emg=raw.emg,
            labels=np.asarray([0, 1, 2, 3, 0], dtype=np.int16),
            trial_id=np.asarray([f"t{i}" for i in range(5)]),
            subject_id=np.asarray(["u01"] * 5), session_id=np.asarray(["d04"] * 5),
            posture=np.ones(5, dtype=np.int16), timestamp_ms=np.arange(5),
        )
        predictions = np.asarray([1, 1, 2, 3, 0])
        probabilities = np.full((5, 4), 0.01)
        probabilities[np.arange(5), predictions] = 0.97
        metrics, _ = evaluate_unibo_probabilities("validation", raw, probabilities, 0.0)
        self.assertAlmostEqual(metrics["window_raw"]["active_gesture_macro_f1"], 8.0 / 9.0)

    def test_active_risk_coverage_excludes_neutral_truth(self):
        raw = make_raw_windows(8)
        probabilities = np.full((8, 4), 0.05)
        probabilities[:, 0] = 0.85
        metrics, _ = evaluate_unibo_probabilities("validation", raw, probabilities, 0.0)
        active_rows = metrics["risk_coverage_active"]
        self.assertTrue(active_rows)
        self.assertTrue(all(0.0 <= row["coverage"] <= 1.0 for row in active_rows))
        self.assertGreater(active_rows[0]["risk"], metrics["risk_coverage_all"][0]["risk"])

    def test_chronological_loader_never_opens_unrequested_test_days(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "trials" / "u01" / "d01" / "p01" / "train.npz"
            test_path = root / "trials" / "u01" / "d07" / "p01" / "test.npz"
            train_path.parent.mkdir(parents=True)
            test_path.parent.mkdir(parents=True)
            train_path.touch()
            test_path.touch()
            trial = BenchmarkTrial(
                path=train_path, subject_id="u01", session_id="d01", trial_id="train",
                timestamp_ms=np.arange(80) * 5.0,
                emg=np.ones((80, 4)),
                hand_label=np.zeros(80, dtype=np.int16), posture_label=1,
                source_label=np.ones(80, dtype=np.int16), source_relabel=np.ones(80, dtype=np.int16),
                stable_mask=np.ones(80, dtype=bool), benchmark_eligible=True, source_gesture=1,
            )

            def guarded_loader(path: Path, **_: object) -> BenchmarkTrial:
                if "d07" in str(path):
                    raise AssertionError("test path was opened")
                return trial

            with patch("emgimu.datasets.unibo_physiology.load_benchmark_trial", side_effect=guarded_loader):
                loaded = load_chronological_raw_windows(root, (1,))
            self.assertEqual(len(loaded), 2)

    def test_train_capping_is_fold_specific_and_validation_is_uncapped(self):
        raw = make_raw_windows(40)
        raw = UniBoRawWindows(
            emg=raw.emg, labels=raw.labels,
            trial_id=np.asarray(["shared"] * 20 + ["validation"] * 20),
            subject_id=np.asarray(["u01"] * 40),
            session_id=np.asarray(["d01"] * 20 + ["d02"] * 20),
            posture=raw.posture, timestamp_ms=raw.timestamp_ms,
        )
        train, validation = chronological_fold(raw, (1,), 2, 2)
        self.assertLessEqual(len(train), 8)
        self.assertEqual(len(validation), 20)


if __name__ == "__main__":
    unittest.main()
