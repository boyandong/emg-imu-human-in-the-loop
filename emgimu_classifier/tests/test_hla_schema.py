from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from emgimu.datasets.hla_schema import (
    ChannelSpec,
    DatasetManifestV2,
    EMGTrial,
    OntologyEntry,
    OntologyRelation,
    load_hla_manifest,
    load_hla_trial,
    write_hla_manifest,
    write_hla_trial,
    check_hla_dataset,
)
from emgimu.state import Gesture


def manifest() -> DatasetManifestV2:
    return DatasetManifestV2(
        dataset_id="synthetic",
        dataset_version="1",
        adapter_version="1",
        source_url="https://example.invalid/source",
        license_id="test-only",
        native_sample_rate_hz=1000.0,
        has_imu=False,
        channel_layouts={
            "ring8": tuple(
                ChannelSpec(
                    f"ch{index}", ring_angle_deg=index * 45.0,
                    anatomical_region=None if index % 2 else "unknown",
                )
                for index in range(8)
            )
        },
        ontology=(
            OntologyEntry("rest", "Rest", 0, OntologyRelation.EXACT, Gesture.NEUTRAL),
            OntologyEntry("grip", "Power grip", 1, OntologyRelation.RELATED),
        ),
    )


class HlaSchemaTests(unittest.TestCase):
    def test_manifest_round_trip_preserves_optional_channel_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            write_hla_manifest(path, manifest())
            loaded = load_hla_manifest(path.parent)
            self.assertEqual(loaded, manifest())
            self.assertIsNone(loaded.channel_layouts["ring8"][1].anatomical_region)

    def test_related_label_cannot_enter_common_label_space(self):
        with self.assertRaisesRegex(ValueError, "cannot enter the canonical"):
            OntologyEntry("grip", "Power grip", 1, OntologyRelation.RELATED, Gesture.FIST)

    def test_variable_channel_trial_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trial = EMGTrial(
                path=root / "trial.npz",
                dataset_id="synthetic",
                subject_id="s1",
                session_id="day1",
                trial_id="t1",
                channel_layout_id="ring8",
                sample_rate_hz=1000.0,
                timestamp_ms=np.arange(250, dtype=np.float64),
                emg=np.ones((250, 8), dtype=np.float32),
                source_label=np.asarray(["rest"] * 250),
                task_label=np.zeros(250, dtype=np.int16),
                canonical_label=np.zeros(250, dtype=np.int16),
                stable_mask=np.ones(250, dtype=bool),
            )
            write_hla_trial(trial.path, trial)
            loaded = load_hla_trial(trial.path, manifest())
            self.assertEqual(loaded.emg.shape, (250, 8))
            self.assertEqual(loaded.channel_layout_id, "ring8")
            np.testing.assert_array_equal(loaded.task_label, trial.task_label)

    def test_generic_checker_accepts_v2_and_rejects_false_canonical_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_hla_manifest(root / "manifest.json", manifest())
            n = 250
            path = root / "trials" / "s1" / "trial.npz"
            valid = EMGTrial(
                path, "synthetic", "s1", "day1", "t1", "ring8", 1000.0,
                np.arange(n, dtype=np.float64), np.ones((n, 8)),
                np.asarray(["rest"] * n), np.zeros(n, dtype=np.int16),
                np.zeros(n, dtype=np.int16), np.ones(n, dtype=bool),
            )
            write_hla_trial(path, valid)
            self.assertEqual(check_hla_dataset(root)["status"], "ok")
            invalid = EMGTrial(
                path, "synthetic", "s1", "day1", "t1", "ring8", 1000.0,
                np.arange(n, dtype=np.float64), np.ones((n, 8)),
                np.asarray(["grip"] * n), np.ones(n, dtype=np.int16),
                np.full(n, int(Gesture.FIST), dtype=np.int16), np.ones(n, dtype=bool),
            )
            write_hla_trial(path, invalid)
            report = check_hla_dataset(root)
            self.assertEqual(report["status"], "error")
            self.assertTrue(any("expected -1" in row for row in report["errors"]))

    def test_unibo_v1_upgrade_keeps_power_grip_dataset_specific(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps({
                "schema_version": "1.0",
                "dataset_id": "unibo-inail",
                "adapter_version": "1.0.0",
                "source_url": "https://example.invalid/unibo",
                "license_id": "LGPL-2.1",
                "source_sample_rate_hz": 500.0,
                "target_sample_rate_hz": 200.0,
                "has_imu": False,
                "channel_names": [
                    "extensor_carpi_ulnaris", "extensor_digitorum_communis",
                    "flexor_carpi_radialis", "flexor_carpi_ulnaris",
                ],
            }), encoding="utf-8")
            (root / "label_map.json").write_text(json.dumps({
                "source_labels": {
                    "1": {"name": "rest", "hand_label": 0, "eligible": True},
                    "2": {"name": "power_grip", "hand_label": 2, "eligible": True},
                    "3": {"name": "two_finger_pinch", "hand_label": 1, "eligible": True},
                    "4": {"name": "three_finger_pinch", "hand_label": -1, "eligible": False},
                    "5": {"name": "pointing_index", "hand_label": -1, "eligible": False},
                    "6": {"name": "open_hand", "hand_label": 3, "eligible": True},
                }
            }), encoding="utf-8")
            loaded = load_hla_manifest(root)
            power = loaded.ontology_by_source()["2"]
            self.assertEqual(power.task_label, 2)
            self.assertEqual(power.relation, OntologyRelation.RELATED)
            self.assertIsNone(power.canonical_gesture)

            n = 20
            trial_path = root / "trial.npz"
            np.savez_compressed(
                trial_path,
                subject_id=np.asarray("u01"), session_id=np.asarray("d01"),
                trial_id=np.asarray("legacy"), timestamp_ms=np.arange(n) * 5.0,
                emg=np.ones((n, 4)), hand_label=np.full(n, 2), posture_label=np.asarray(1),
                source_label=np.full(n, 2), source_relabel=np.full(n, 2),
                stable_mask=np.ones(n, dtype=bool), benchmark_eligible=np.asarray(True),
                source_gesture=np.asarray(2),
            )
            upgraded = load_hla_trial(trial_path, loaded)
            self.assertTrue(np.all(upgraded.task_label == 2))
            self.assertTrue(np.all(upgraded.canonical_label == int(Gesture.UNKNOWN)))


if __name__ == "__main__":
    unittest.main()
