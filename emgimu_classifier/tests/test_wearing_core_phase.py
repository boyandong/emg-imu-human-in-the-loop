from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from emgimu.feature_bank.wearing_core_ring import run


class WearingCorePhaseGuards(unittest.TestCase):
    def test_invalid_phase_cannot_fall_through_to_final_users(self):
        with tempfile.TemporaryDirectory() as d, patch('emgimu.feature_bank.wearing_core_ring.load') as load:
            output = Path(d) / 'out'
            for phase in (None, '', 'validaton', 'test'):
                with self.assertRaisesRegex(ValueError, 'phase'):
                    run(Path(d) / 'archive.zip', output, phase)
            load.assert_not_called()
            self.assertFalse(output.exists())

    def test_invalid_definition_rejected_before_any_data_or_output(self):
        with tempfile.TemporaryDirectory() as d, patch('emgimu.feature_bank.wearing_core_ring.load') as load:
            output = Path(d) / 'out'
            with self.assertRaisesRegex(ValueError, 'definition'):
                run(Path(d) / 'archive.zip', output, 'validation', 'typo')
            load.assert_not_called()
            self.assertFalse(output.exists())
