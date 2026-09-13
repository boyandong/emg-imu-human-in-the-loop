from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from emgimu.datasets.hla_schema import OntologyRelation, check_hla_dataset, load_hla_manifest, load_hla_trial
from emgimu.datasets.adapters.myo_armband import MyoArmbandEvaluationAdapter
from emgimu.state import Gesture


def write_subject(root: Path, name: str = "Female0") -> None:
    subject = root / name
    for block in ("training0", "Test0", "Test1"):
        destination = subject / block
        destination.mkdir(parents=True)
        for index in range(28):
            values = np.arange(80, dtype=np.int16) + index
            values.tofile(destination / f"classe_{index}.dat")


class MyoArmbandAdapterTests(unittest.TestCase):
    def test_adapts_complete_trials_and_preserves_block_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "EvaluationDataset"
            write_subject(source)
            output = root / "adapted"
            result = MyoArmbandEvaluationAdapter().adapt(source, output)
            self.assertEqual(result.trial_count, 84)
            report = check_hla_dataset(output)
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["trials_by_layout"], {"myo_ring_8": 84})
            manifest = load_hla_manifest(output)
            self.assertEqual(manifest.session_semantics, "within_session_acquisition_block")
            self.assertEqual(len(manifest.channel_layouts["myo_ring_8"]), 8)
            close = manifest.ontology_by_source()["5"]
            self.assertEqual(close.relation, OntologyRelation.EXACT)
            self.assertEqual(close.canonical_gesture, Gesture.FIST)
            radial = manifest.ontology_by_source()["1"]
            self.assertEqual(radial.relation, OntologyRelation.DATASET_ONLY)
            trial_path = next((output / "trials").rglob("*g05-r00.npz"))
            trial = load_hla_trial(trial_path, manifest)
            self.assertEqual(trial.emg.shape, (10, 8))
            self.assertTrue(np.all(trial.task_label == 5))
            self.assertTrue(np.all(trial.canonical_label == int(Gesture.FIST)))

    def test_partial_source_is_reported_without_inventing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            block = root / "Male0" / "Test1"
            block.mkdir(parents=True)
            np.arange(80, dtype=np.int16).tofile(block / "classe_0.dat")
            output = root / "adapted"
            result = MyoArmbandEvaluationAdapter().adapt(root, output)
            self.assertEqual(result.trial_count, 1)
            self.assertTrue(any("missing" in warning for warning in result.warnings))

    def test_refuses_non_interleaved_channel_length_and_is_atomic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            block = root / "Male0" / "Test1"
            block.mkdir(parents=True)
            np.arange(79, dtype=np.int16).tofile(block / "classe_0.dat")
            output = root / "adapted"
            with self.assertRaisesRegex(ValueError, "not divisible"):
                MyoArmbandEvaluationAdapter().adapt(root, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
