from __future__ import annotations

from io import StringIO
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np

from emgimu.datasets.semg_manus import load_semg_manus_windows


class SemgManusAdapterTests(unittest.TestCase):
    def test_reads_schema_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "sample.zip"
            text = StringIO(); np.savetxt(text, np.arange(120 * 42).reshape(120, 42), delimiter=",")
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("data/u_3/s_2/g_flexext_fist/recording_fast_01_01_2024_00_00_00.csv", text.getvalue())
            loaded = load_semg_manus_windows(archive, users=(3,), sessions=(2,), gestures=("flexext_fist",), maximum_windows_per_trial=2)
            self.assertEqual(loaded.batch.emg.shape, (2, 40, 8))
            self.assertEqual(loaded.batch.imu.shape, (2, 40, 6))
            self.assertEqual(set(loaded.speeds), {"fast"})
            self.assertEqual(set(loaded.sessions), {2})


if __name__ == "__main__":
    unittest.main()
