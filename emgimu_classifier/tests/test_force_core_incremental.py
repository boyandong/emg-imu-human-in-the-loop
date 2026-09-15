import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from emgimu.feature_bank.force_core_incremental import evaluate


class CoreFitGuards(unittest.TestCase):
    def test_non_source_users_rejected_before_target_load(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'source';source.mkdir()
            (source/'run_manifest.json').write_text(json.dumps({'source_users':[1,2,3,4,5,7],'source_condition':'Ramp'}))
            with patch('emgimu.feature_bank.force_core_incremental.load_libemg_force_windows') as load:
                with self.assertRaisesRegex(AssertionError,'source-only Ramp'):
                    evaluate(Path(d)/'raw',source,Path(d)/'output','final')
                load.assert_not_called()

    def test_training_phase_cannot_be_scored_as_heldout(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'Independent'):
                evaluate(Path(d)/'raw',Path(d)/'source',Path(d)/'output','source_fit')


if __name__=='__main__':unittest.main()
