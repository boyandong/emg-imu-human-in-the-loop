import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from emgimu.feature_bank.reliability_selection import load_policy


class ReliabilityPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.path=self.root/'policy.json'
        (self.root/'oof_predictions.npz').write_bytes(b'source-only OOF fixture')
        self.policy={'dataset':'epn612','families':['F0','F1'],'source_trials':['source-A'],
            'target_data_opened':False,'population_weights':[.3,.7],
            'selected':{'n0':2.,'reliability_temperature':.5},
            'source_oof_sha256':hashlib.sha256((self.root/'oof_predictions.npz').read_bytes()).hexdigest()}

    def load(self):
        self.path.write_text(json.dumps(self.policy))
        return load_policy(self.path,'epn612',['source-A'],('F0','F1'),self.root)

    def test_selected_parameters_and_nonuniform_prior_are_preserved(self):
        w,n0,tau,_=self.load();np.testing.assert_array_equal(w,[.3,.7])
        self.assertEqual((n0,tau),(2.,.5))

    def test_wrong_source_target_access_and_modified_oof_are_rejected(self):
        for field,value in [('source_trials',['target-B']),('target_data_opened',True),
                            ('source_oof_sha256','bad-hash'),('families',['F1','F0']),('dataset','semg_manus')]:
            original=self.policy[field];self.policy[field]=value
            with self.assertRaises(AssertionError):self.load()
            self.policy[field]=original
