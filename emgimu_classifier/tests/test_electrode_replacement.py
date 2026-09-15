import unittest
import tempfile
from pathlib import Path
import numpy as np
from emgimu.datasets.electrode_replacement import parse_recording_name,load_electrode_replacement_recording


class ReplacementRecordingTests(unittest.TestCase):
    def test_native_names_preserve_intent_and_position(self):
        self.assertEqual(parse_recording_name('ID10_3F_P2.txt'),(10,'3F','P2'))
        for name in ('ID11_PS_P1.txt','ID1_OPEN_P1.txt','ID1_PP_P4.txt'):
            with self.assertRaises(ValueError):parse_recording_name(name)

    def test_raw_recording_is_not_given_fabricated_trial_or_rest_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'ID1_PS_P1.txt'
            values=np.arange(80,dtype=float).reshape(10,8)
            np.savetxt(path,values)
            recording=load_electrode_replacement_recording(path)
            np.testing.assert_array_equal(recording.emg,values)
            self.assertEqual(recording.sample_rate_hz,1000)
            self.assertIsNone(recording.interval_labels)
            self.assertIsNone(recording.repetition_boundaries)
            np.savetxt(path,np.ones((10,9)))
            with self.assertRaisesRegex(ValueError,'eight-channel'):load_electrode_replacement_recording(path)
