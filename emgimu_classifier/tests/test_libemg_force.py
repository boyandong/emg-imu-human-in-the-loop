from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np

from emgimu.datasets.libemg_force import load_libemg_force_windows


class LibemgForceAdapterTests(unittest.TestCase):
    def test_trial_first_windowing_and_equal_trial_weight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "S1"
            folder.mkdir()
            np.savetxt(folder / "S1_Ramp_C1_R1.csv", np.ones((401, 8)), delimiter=",")
            np.savetxt(folder / "S1_Ramp_C2_R1.csv", np.ones((801, 8)) * 2, delimiter=",")
            loaded = load_libemg_force_windows(root, subjects=(1,), conditions=("Ramp",), maximum_windows_per_trial=8)
            self.assertEqual(loaded.batch.emg.shape, (6, 200, 8))
            first = loaded.sample_weight[loaded.trials == "S1_Ramp_C1_R1"].sum()
            second = loaded.sample_weight[loaded.trials == "S1_Ramp_C2_R1"].sum()
            self.assertAlmostEqual(float(first), float(second))

    def test_unknown_filename_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "S1"
            folder.mkdir()
            np.savetxt(folder / "bad.csv", np.ones((200, 8)), delimiter=",")
            with self.assertRaisesRegex(ValueError, "unrecognized"):
                load_libemg_force_windows(folder.parent, subjects=(1,), conditions=("Ramp",))


if __name__ == "__main__":
    unittest.main()
