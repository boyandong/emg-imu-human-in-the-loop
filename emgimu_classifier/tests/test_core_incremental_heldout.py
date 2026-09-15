import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from emgimu.feature_bank.core_incremental_heldout import run


class HeldoutGuards(unittest.TestCase):
    def test_target_user_in_source_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'source';source.mkdir()
            (source/'run_manifest.json').write_text(json.dumps({'dataset':'epn612','phase':'final',
                'source_trials':['trainingJSON:user19:idx_1']}))
            with patch('emgimu.feature_bank.core_incremental_heldout.load_epn612_windows') as load:
                with self.assertRaisesRegex(AssertionError,'Target users'):
                    run(Path(d)/'raw.zip',source,Path(d)/'output')
                load.assert_not_called()
            self.assertFalse((Path(d)/'output').exists())

    def test_source_phase_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'source';source.mkdir()
            (source/'run_manifest.json').write_text(json.dumps({'dataset':'epn612','phase':'source_nested_oof'}))
            with self.assertRaisesRegex(ValueError,'independent'):
                run(Path(d)/'raw.zip',source,Path(d)/'output')


if __name__=='__main__':
    unittest.main()
