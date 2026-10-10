"""Study coverage, fixed-weight removal and metric denominator checks."""
import numpy as np
import pytest
from benchmarks.new_bank_v3.roam_native_joint_v1 import derive,score,select,merge,CLASSES,GROUPS
from emgimu.datasets.roam_cued_intervals_v1 import RoamCuedIntervalsV1,CHANNELS,PREPROCESSING
from emgimu.feature_bank.personal_temporal_bouts_v1 import TemporalBoutBatchV1


def fixture_data(prefix):
    ids=tuple(f'{prefix}:{i}' for i in range(9));classes=('relax','open','close','open','relax','open','relax','close','relax')
    b=TemporalBoutBatchV1(tuple(np.full((200+i,8),i,dtype=np.float32) for i in range(9)),ids,
        (prefix,)*9,tuple(210*i for i in range(9)),200.,CHANNELS,PREPROCESSING,'complete_cued')
    return RoamCuedIntervalsV1(b,dict(zip(ids,classes)),{})


def test_native_calibration_selection_uses_distinct_cues_and_whole_recording_query_exclusion():
    a,b=fixture_data('long'),fixture_data('query')
    selected=select(a,2)
    assert len(selected.batch.trial_ids)==6 and list(selected.labels.values())==['close','close','open','open','relax','relax']
    assert set(selected.batch.recording_ids).isdisjoint(b.batch.recording_ids)
    assert len(merge([a,b]).batch.trial_ids)==18
    with pytest.raises(ValueError):select(a,3)
    with pytest.raises(ValueError):merge([a,a])


def test_native_derived_arms_preserve_axis_weights_and_whole_F5_semantics():
    rng=np.random.default_rng(714);q=rng.dirichlet([1,2,3],(7,4));weights=np.arange(1,8,dtype=float);weights/=weights.sum()
    p={g:q[i] for i,g in enumerate(GROUPS)};window=dict(trial_ids=('c','a','d','b'),weights=weights,
        decision_provider_probabilities=p,probabilities=sum(weights[i]*q[i] for i in range(7)))
    order=[1,3,0,2];dtw=rng.dirichlet([2,1,3],4);sig=rng.dirichlet([3,2,1],4)
    temporal=dict(DTW_blended=dtw,signature_blended=sig,DTW_long=sig,signature_long=dtw)
    joint=dict(window=window,trial_ids=('a','b','c','d'),temporal=dict(arms=temporal),
        probabilities=.75*window['probabilities'][order]+.125*dtw+.125*sig)
    other={n:window for n in ('window_reliability','window_F7','window_F8')}
    arms=derive(joint,other,{'population':window['probabilities'][order]})
    keep=[i for i,g in enumerate(GROUPS) if g not in ('F2ac','F2b')]
    omitted=sum(weights[i]*q[i] for i in keep)/sum(weights[keep])
    np.testing.assert_allclose(arms['minus_family_F2'],.75*omitted[order]+.125*dtw+.125*sig,rtol=0,atol=1e-14)
    keep=[i for i,g in enumerate(GROUPS) if g!='F5window']
    np.testing.assert_allclose(arms['minus_family_F5'],(sum(weights[i]*q[i] for i in keep)/sum(weights[keep]))[order],rtol=0,atol=1e-14)
    np.testing.assert_array_equal(arms['joint_minus_temporal'],window['probabilities'][order])


def test_native_metrics_count_every_cue_once_with_three_class_brier():
    q=np.array([[.7,.2,.1],[.2,.6,.2],[.3,.3,.4]])
    r=score(CLASSES,q)
    assert r['trials']==3 and r['accuracy']==1. and r['macro_f1']==1.
    assert r['log_loss']==pytest.approx(-np.log([.7,.6,.4]).mean())
    assert r['brier']==pytest.approx(np.mean((q-np.eye(3))**2))
    with pytest.raises(ValueError):score(CLASSES,q[:,::-1]*2)
