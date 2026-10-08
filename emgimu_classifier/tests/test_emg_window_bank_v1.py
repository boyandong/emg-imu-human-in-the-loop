import numpy as np
import pytest
from benchmarks.new_bank_v3.emg_window_bank_v1 import compositions, concatenate, GROUPS


def test_group_removal_and_addition_geometry_is_exact():
    c = compositions()
    assert len(c) == 13 and c['window_bank'] == GROUPS
    features = {g: np.full((3, i + 1), i, dtype=np.float32) for i, g in enumerate(GROUPS)}
    for group in GROUPS:
        remaining = c['window_bank_minus_' + group]
        assert group not in remaining and set(remaining) == set(GROUPS) - {group}
        x = concatenate(features, remaining)
        assert x.shape[1] == sum(features[g].shape[1] for g in remaining)
        assert not np.any(x == GROUPS.index(group))
    for group in GROUPS[1:]: assert c['F0_plus_' + group] == ('F0', group)


def test_feature_join_rejects_unaligned_invalid_or_duplicate_groups():
    with pytest.raises(ValueError): concatenate({'F0': np.ones((3, 1)), 'F1': np.ones((2, 1))}, ('F0', 'F1'))
    with pytest.raises(ValueError): concatenate({'F0': np.ones((3, 1))}, ('F0', 'F0'))
    with pytest.raises(ValueError): concatenate({'F0': np.full((3, 1), np.nan)}, ('F0',))
    with pytest.raises(ValueError): concatenate({}, ())
