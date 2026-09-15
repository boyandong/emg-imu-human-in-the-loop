from __future__ import annotations

from io import StringIO
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np

from emgimu.datasets.emg_fmg import load_emg_fmg_windows


class EmgFmgAdapterTests(unittest.TestCase):
    def test_selects_only_published_emg_channels(self)->None:
        with tempfile.TemporaryDirectory() as directory:
            archive=Path(directory)/"sample.zip";text=StringIO();text.write(','.join(f'Channel {i}' for i in range(1,17))+'\n');np.savetxt(text,np.tile(np.arange(16),(36000,1)),delimiter=',')
            with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as handle:handle.writestr('Data/Par 1/Key/0/Key 0 1.csv',text.getvalue())
            loaded=load_emg_fmg_windows(archive,subjects=(1,),loads=(0,),positions=(1,),maximum_windows_per_trial=2)
            self.assertEqual(loaded.batch.emg.shape,(2,400,8));self.assertTrue(np.all(loaded.batch.emg[0,0]==np.arange(8,16)))


if __name__=='__main__':unittest.main()
