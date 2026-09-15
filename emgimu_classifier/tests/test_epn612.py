from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from emgimu.datasets.epn612 import load_epn612_windows


def sample(gesture: str) -> dict:
    return {
        "startPointforGestureExecution": 20, "gestureName": gesture,
        "emg": {f"ch{i}": list(range(100)) for i in range(1, 9)},
        "accelerometer": {axis: list(range(25)) for axis in "xyz"},
        "gyroscope": {axis: list(range(25)) for axis in "xyz"},
    }


class Epn612AdapterTests(unittest.TestCase):
    def test_reads_labelled_trials_from_archive_and_weights_trials_equally(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "sample.zip"
            payload = {
                "generalInfo": {"samplingFrequencyInHertz": 200},
                "trainingSamples": {"idx_1": sample("fist"), "idx_2": sample("open")},
            }
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("EMG-EPN612 Dataset/trainingJSON/user1/user1.json", json.dumps(payload))
            loaded = load_epn612_windows(archive, users=(1,), windows_per_trial=2)
            self.assertEqual(loaded.batch.emg.shape, (4, 40, 8))
            self.assertEqual(loaded.batch.imu.shape, (4, 10, 6))
            self.assertEqual(set(loaded.labels), {1, 4})
            totals = [loaded.sample_weight[loaded.trials == trial].sum() for trial in set(loaded.trials)]
            self.assertAlmostEqual(float(totals[0]), float(totals[1]))

    def test_rejects_unlabelled_testing_samples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "sample.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("EMG-EPN612 Dataset/trainingJSON/user1/user1.json", json.dumps({
                    "generalInfo": {"samplingFrequencyInHertz": 200}, "trainingSamples": {"idx_1": sample("bad")},
                }))
            with self.assertRaisesRegex(ValueError, "unknown gesture"):
                load_epn612_windows(archive, users=(1,))


if __name__ == "__main__":
    unittest.main()
