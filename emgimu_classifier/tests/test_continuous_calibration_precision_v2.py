"""A native converter parity fixture that distinguishes float32 resampling."""
import numpy as np
from scipy.io import savemat
from emgimu.datasets.adapters.unibo_inail import _load_source
from emgimu.signal import polyphase_resample


def test_reserved_reconstruction_matches_converter_precision_without_tolerance(tmp_path):
    emg = np.random.default_rng(912).normal(0, 3000, (2500, 4)).astype(np.float32)
    counter = np.r_[np.zeros(100), np.ones(2300), np.zeros(100)]
    label = np.r_[np.ones(200), np.full(2100, 2), np.ones(200)]
    path = tmp_path/'source.mat'
    savemat(path, dict(emg=emg, gestureCounter=counter[:, None], label=label[:, None], relabel=label[:, None]))
    converted, _, _, _ = _load_source(path)
    assert converted.dtype == np.float64
    expected = polyphase_resample(converted[100:2400], 500., 200.).astype(np.float32)
    reconstructed = polyphase_resample(np.asarray(emg[100:2400], dtype=np.float64), 500., 200.).astype(np.float32)
    np.testing.assert_array_equal(reconstructed, expected)
    assert np.any(polyphase_resample(emg[100:2400], 500., 200.).astype(np.float32) != expected)
