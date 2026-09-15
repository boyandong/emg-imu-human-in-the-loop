from __future__ import annotations

from io import StringIO
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np

from emgimu.datasets.electrode_shift import load_electrode_shift_windows


class ElectrodeShiftAdapterTests(unittest.TestCase):
    def test_reads_domain_and_equal_trial_weights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "sample.zip"; first = StringIO(); second = StringIO()
            np.savetxt(first, np.ones((81, 8)), delimiter=","); np.savetxt(second, np.ones((161, 8)), delimiter=",")
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("CIILData-main/ElectrodeShift/subject0/training/R_0_C_0.csv", first.getvalue())
                handle.writestr("CIILData-main/ElectrodeShift/subject0/training/R_1_C_1.csv", second.getvalue())
            loaded = load_electrode_shift_windows(archive, subjects=(0,), domains=("training",))
            self.assertEqual(loaded.batch.emg.shape, (6, 40, 8))
            totals = [loaded.sample_weight[loaded.trials == trial].sum() for trial in set(loaded.trials)]
            self.assertAlmostEqual(float(totals[0]), float(totals[1]))


if __name__ == "__main__": unittest.main()
