import numpy as np
import pytest

from emgimu.feature_bank.core import FeatureBatch
from benchmarks.new_bank_v1.zenodo_replacement_run import descriptors, windows


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


def test_target_position_cannot_change_source_fitted_descriptors():
    rng = np.random.default_rng(6)
    source = {("PS", "P1"): FeatureBatch(rng.normal(size=(4, 200, 8)), 1000.0),
              ("FL", "P1"): FeatureBatch(rng.normal(size=(4, 200, 8)), 1000.0)}
    target = FeatureBatch(rng.normal(size=(4, 200, 8)), 1000.0)
    original = descriptors({**source, ("PS", "P2"): target}, ["PS", "FL"])
    changed = descriptors({**source, ("PS", "P2"):
                           FeatureBatch(target.emg * 40 + 100, 1000.0)}, ["PS", "FL"])
    for key in source:
        for family in original[key]:
            np.testing.assert_array_equal(original[key][family], changed[key][family])
