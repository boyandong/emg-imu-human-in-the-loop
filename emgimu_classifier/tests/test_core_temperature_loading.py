import json
from pathlib import Path
import tempfile
import unittest
from emgimu.feature_bank.core_probability_oof import load_temperatures, apply_temperature
import numpy as np


class CalibrationBindingTests(unittest.TestCase):
    def test_target_users_rejected_before_oof_probabilities_are_read(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'source';source.mkdir();cal=Path(d)/'cal';cal.mkdir()
            (source/'run_manifest.json').write_text(json.dumps({'dataset':'semg_manus','source_users':[3,4,5,6,7,8]}))
            (cal/'run_manifest.json').write_text(json.dumps({'dataset':'semg_manus','source_users':[3,4,5,6,7,9],'target_data_opened':False}))
            with self.assertRaisesRegex(AssertionError,'source users'):
                load_temperatures(source,cal,{})

    def test_temperature_preserves_population_decision_and_legacy_identity(self):
        p=np.array([[.7,.2,.1],[.05,.15,.8]])
        self.assertIs(apply_temperature(p,'Core',{}),p)
        result=apply_temperature(p,'Core',{'probability_calibration':{'temperatures':{'Core':4}}})
        np.testing.assert_array_equal(result.argmax(1),p.argmax(1))
        np.testing.assert_allclose(result.sum(1),1)
        self.assertTrue(np.all(result.max(1)<p.max(1)))


if __name__=='__main__':unittest.main()
