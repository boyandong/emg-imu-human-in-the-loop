import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from emgimu.datasets.libemg_force import ForceWindows
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.core_probability_oof import run, take


class SourceProbabilityGuards(unittest.TestCase):
    def test_final_user_in_source_manifest_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'source';source.mkdir()
            (source/'run_manifest.json').write_text(json.dumps({'dataset':'libemg_contraction_intensity',
                'source_users':[1,2,3,4,5,9],'source_condition':'Ramp'}))
            with patch('emgimu.datasets.libemg_force.load_libemg_force_windows') as load:
                with self.assertRaisesRegex(AssertionError,'Only source'):
                    run(Path(d)/'raw',source,Path(d)/'output')
                load.assert_not_called()

    def test_fold_take_preserves_metadata_and_excludes_held_windows(self):
        x=np.arange(6*4*8,dtype=float).reshape(6,4,8)
        data=ForceWindows(FeatureBatch(x,1000),np.array([0,0,1,1,2,2]),np.array([1,1,2,2,3,3]),
            np.repeat('Ramp',6),np.array(['a','a','b','b','c','c']),np.ones(6,int),np.ones(6))
        training=take(data,np.array([0,1,2,3]));held=take(data,np.array([4,5]))
        self.assertFalse(set(training.trials)&set(held.trials))
        np.testing.assert_array_equal(training.batch.emg,x[:4])
        np.testing.assert_array_equal(held.subjects,[3,3])
        self.assertEqual(data.batch.windows,6)


if __name__=='__main__':unittest.main()
