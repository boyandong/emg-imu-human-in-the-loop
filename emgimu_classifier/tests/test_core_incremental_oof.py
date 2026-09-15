import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from emgimu.feature_bank.core_incremental_oof import run


class CoreIncrementalLeakageTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.source=Path(self.tmp.name)/'source';self.source.mkdir();self.output=Path(self.tmp.name)/'output'
        (self.source/'run_manifest.json').write_text(json.dumps({'dataset':'epn612','source_users':[1,2]}))

    def test_overlapping_training_and_heldout_trial_is_rejected(self):
        (self.source/'split_trial_ids.json').write_text(json.dumps([{'fold':0,'train':['a'],'validation':['a'],'inner':[]}]))
        with self.assertRaises(AssertionError):run(self.source,self.output)
        self.assertFalse(self.output.exists())

    def test_target_users_cannot_replace_source_oof_users(self):
        splits=[{'fold':0,'train':['b'],'validation':['a'],'inner':[]},
                {'fold':1,'train':['a'],'validation':['b'],'inner':[]}]
        (self.source/'split_trial_ids.json').write_text(json.dumps(splits))
        np.savez(self.source/'oof_predictions.npz',labels=[0,1],users=[16,17],trials=['a','b'],folds=[0,1])
        with self.assertRaises(AssertionError):run(self.source,self.output)
        self.assertFalse(self.output.exists())
