import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from emgimu.feature_bank.manus_core_incremental import evaluate, USERS


class SessionCoreGuards(unittest.TestCase):
    def test_target_session_in_source_partition_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'source';source.mkdir()
            (source/'run_manifest.json').write_text(json.dumps({'source_session':3,'source_users':list(USERS)}))
            with patch('emgimu.feature_bank.manus_core_incremental.load_semg_manus_windows') as load:
                with self.assertRaisesRegex(AssertionError,'partition'):
                    evaluate(Path(d)/'raw.zip',source,Path(d)/'out','final')
                load.assert_not_called()

    def test_source_phase_cannot_be_reported_as_heldout(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'Independent'):
                evaluate(Path(d)/'raw.zip',Path(d)/'source',Path(d)/'out','source_fit')


if __name__=='__main__':unittest.main()
