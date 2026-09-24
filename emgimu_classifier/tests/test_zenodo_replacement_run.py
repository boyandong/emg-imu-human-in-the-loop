import numpy as np
import pytest

from benchmarks.new_bank_v1.zenodo_replacement_run import windows


def test_file_level_window_selection_covers_recording_without_overlap():
    signal = np.repeat(np.arange(30_000, dtype=float)[:, None], 8, axis=1)
    selected = windows(signal)
    assert selected.emg.shape == (100, 200, 8)
    starts = selected.emg[:, 0, 0].astype(int)
    assert starts[0] == 0
    assert starts[-1] == len(signal) - 200
    assert np.all(np.diff(starts) >= 200)
    for index, start in enumerate(starts):
        np.testing.assert_array_equal(selected.emg[index], signal[start:start + 200])


def test_short_file_is_rejected_instead_of_creating_a_partial_window():
    with pytest.raises(ValueError, match="shorter than one window"):
        windows(np.zeros((199, 8)))
