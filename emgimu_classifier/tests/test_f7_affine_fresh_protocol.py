"""Guard the precommitted fresh-cohort decision and paired probability contract."""

import numpy as np
import pytest

from benchmarks.new_bank_v3.f7_affine_fresh_epn import (
    ARMS, FRESH_USERS, paired_arms, primary_guard,
)


def test_paired_arms_fixed_weight_and_reject_mismatch():
    core = np.eye(6)[[0, 1]] * .8 + .2 / 6
    f7 = np.eye(6)[[1, 0]] * .7 + .3 / 6
    arms = paired_arms(core, f7)
    np.testing.assert_allclose(arms['Core_plus_Affine_F7'], (core + f7) / 2)
    np.testing.assert_allclose(arms['Core_plus_Uniform'], (core + 1 / 6) / 2)
    with pytest.raises(ValueError):
        paired_arms(core, f7[:1])


def test_primary_guard_keeps_a_failed_criterion_visible():
    rows = []
    for subject in ('ALL', *(str(u) for u in FRESH_USERS)):
        for arm in ARMS:
            loss = {'Core': .8, 'Core_plus_Uniform': .7,
                    'Affine_F7': .6, 'Core_plus_Affine_F7': .65}[arm]
            rows.append(dict(subject=subject, arm=arm, shots_per_class=5,
                             log_loss=loss, macro_f1=.6 if arm == 'Core' else .59,
                             brier=.3 if arm == 'Core' else .2))
    result = primary_guard(rows)
    assert result['criteria']['pooled_logloss_lower']
    assert result['criteria']['logloss_better_than_uniform']
    assert result['user_logloss_wins'] == 10
    assert not result['criteria']['pooled_macro_f1_nonworse']
    assert not result['conjunction_passed']
    with pytest.raises(ValueError):
        primary_guard(rows[:-1])
