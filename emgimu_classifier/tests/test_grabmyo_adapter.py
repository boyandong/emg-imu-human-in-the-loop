from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from emgimu.datasets.adapters.grabmyo import GrabMyoAdapter
from emgimu.datasets.hla_schema import OntologyRelation, check_hla_dataset, load_hla_manifest, load_hla_trial
from emgimu.state import Gesture


def write_record(root: Path, *, gesture: int = 4, samples: int = 16) -> Path:
    folder = root / "Session1" / "session1_participant1"
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"session1_participant1_gesture{gesture}_trial1"
    header = folder / f"{stem}.hea"
    data = folder / f"{stem}.dat"
    names = [*(f"F{i}" for i in range(1, 17)), "U1", *(f"W{i}" for i in range(1, 7)),
             "U2", "U3", *(f"W{i}" for i in range(7, 13)), "U4"]
    digital = np.arange(samples * 32, dtype="<i2").reshape(samples, 32)
    digital.tofile(data)
    lines = [f"{stem} 32 2048 {samples}"]
    lines.extend(
        f"{stem}.dat 16 1000(10)/mV 16 0 0 0 0 {name}" for name in names
    )
    header.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return header


class GrabMyoAdapterTests(unittest.TestCase):
    def test_wfdb_record_creates_forearm_and_wrist_views(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_record(root, gesture=4)
            output = root / "adapted"
            result = GrabMyoAdapter().adapt(root, output)
            self.assertEqual(result.trial_count, 1)
            report = check_hla_dataset(output)
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["trials_by_layout"], {
                "grabmyo_forearm_16": 1, "grabmyo_wrist_12": 1,
            })
            manifest = load_hla_manifest(output)
            opposition = manifest.ontology_by_source()["4"]
            self.assertEqual(opposition.relation, OntologyRelation.RELATED)
            self.assertIsNone(opposition.canonical_gesture)
            forearm_path = next((output / "trials").rglob("grabmyo_forearm_16/*.npz"))
            forearm = load_hla_trial(forearm_path, manifest)
            self.assertEqual(forearm.emg.shape, (16, 16))
            self.assertAlmostEqual(float(forearm.emg[0, 0]), -0.01)
            self.assertTrue(np.all(forearm.canonical_label == int(Gesture.UNKNOWN)))

    def test_hand_open_enters_exact_common_ontology(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_record(root, gesture=15)
            output = root / "adapted"
            GrabMyoAdapter().adapt(root, output)
            manifest = load_hla_manifest(output)
            trial = load_hla_trial(next((output / "trials").rglob("*.npz")), manifest)
            self.assertTrue(np.all(trial.canonical_label == int(Gesture.OPEN)))

    def test_bad_wfdb_length_is_rejected_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            header = write_record(root)
            header.with_suffix(".dat").write_bytes(b"\0\0")
            output = root / "adapted"
            with self.assertRaisesRegex(ValueError, "header declares"):
                GrabMyoAdapter().adapt(root, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
