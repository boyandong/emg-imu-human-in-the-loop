import numpy as np
import pytest
from benchmarks.new_bank_v3.emg_f0_f7_bank_v1 import paired_arms, select_trials


def test_nested_independent_trials_and_fixed_evaluation_disjointness():
    ids = np.array([f'class{c}_trial{i}' for c in range(6) for i in range(8)])
    labels = np.repeat(range(6),8)
    selected,evaluation = select_trials(ids,labels,'fixed')
    assert len(evaluation)==18 and len(set(evaluation))==18
    assert not set(evaluation)&{i for indices in selected.values() for i in indices}
    for c,indices in selected.items():
        assert len(indices)==5 and np.all(labels[indices]==c)
    order = np.arange(len(ids))[::-1]
    other,held = select_trials(ids[order],labels[order],'fixed')
    assert set(ids[evaluation])==set(ids[order][held])
    assert all(list(ids[selected[c]])==list(ids[order][other[c]]) for c in range(6))
    with pytest.raises(ValueError): select_trials(np.repeat('duplicate',48),labels,'fixed')
    with pytest.raises(ValueError): select_trials(ids[:40],labels[:40],'fixed')


def test_provider_removal_and_uniform_control_are_exact_and_nonmutating():
    a=np.array([[.7,.1,.05,.05,.05,.05]]);b=a[:,::-1].copy()
    original=(a.copy(),b.copy());arms=paired_arms(a,b)
    np.testing.assert_array_equal(arms['F0'],a)
    np.testing.assert_array_equal(arms['F7'],b)
    np.testing.assert_allclose(arms['F0_F7'],(a+b)*.5)
    np.testing.assert_allclose(arms['F0_uniform'],a*.5+np.ones_like(a)/12)
    arms['F0'][0,0]=0
    np.testing.assert_array_equal(a,original[0]);np.testing.assert_array_equal(b,original[1])
    for wrong in (b[:,:5],b*2,np.full_like(b,np.nan)):
        with pytest.raises(ValueError): paired_arms(a,wrong)
