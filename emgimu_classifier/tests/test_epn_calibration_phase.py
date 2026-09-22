import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from emgimu.feature_bank.epn_calibration import run


class EpnCalibrationPhaseGuards(unittest.TestCase):
    def test_phase_method_and_subjects_rejected_before_any_loading(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "emgimu.feature_bank.epn_calibration.load_epn612_windows"
        ) as load:
            archive = Path(directory) / "missing.zip"
            output = Path(directory) / "output"
            cases = (
                ((16, 17, 18), "validaton", "anchor"),
                ((16, 17, 18), "validation", "anchro"),
                ((19, 20, 21), "validation", "anchor"),
            )
            for subjects, phase, method in cases:
                with self.assertRaises(ValueError):
                    run(archive, output, subjects, phase, method)
            load.assert_not_called()
            self.assertFalse(output.exists())
