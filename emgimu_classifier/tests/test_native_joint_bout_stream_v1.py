"""Native streaming parity, leakage isolation and censoring, not efficacy."""
import inspect
import pickle
import numpy as np
import pytest
from emgimu.feature_bank.native_joint_bout_stream_v1 import NativeJointBoutStreamV1
from test_native_joint_bout_workflow_v2 import workflow, bouts
from benchmarks.new_bank_v3.verify_calibration_rest_continuous_unibo_v1 import independent_intervals


def stream(w,p,**changes):
    kwargs=dict(personal=p,user_id='fixture',session_id='query',source_recording_ids=('source',),
        sample_rate_hz=w.rate,channel_ids=w.window.channels,preprocessing_id=w.window.preprocessing_id)
    kwargs.update(changes)
    return NativeJointBoutStreamV1(w,**kwargs)


def test_native_stream_matches_direct_intervals_irregular_chunks_and_preserves_source(workflow):
    w=workflow;b,y=bouts(w,'stream_cal');p=w.enroll(b,y,user_id='fixture',session_id='long')
    before=pickle.dumps((w,p));rate=int(w.rate)
    x=np.concatenate([np.zeros((rate//2,8)),np.ones((rate*2,8))*5,np.zeros((rate//2,8))])
    s=stream(w,p);actual=[];pos=0
    for end in (13,41,rate+7,len(x)):
        actual.extend(s.feed(x[pos:end],pos,recording_id='query'));pos=end
    assert not s.finish() and len(actual)==1
    other=stream(w,p);direct=other.feed(x,0,recording_id='query');assert direct==actual
    bounds,censored=independent_intervals(x,p.detector.on_,p.detector.off_,w.rate)
    assert [[v['start'],v['end']] for v in actual]==bounds and not censored
    e=actual[0]
    assert e['boundary_kind']=='estimated' and e['available_at_sample_index']==e['end']
    assert set(e['probabilities'])=={'source_window','window_full','joint_full'}
    assert all(np.isclose(sum(v),1.) for v in e['probabilities'].values())
    assert pickle.dumps((w,p))==before
    assert 'labels' not in inspect.signature(s.feed).parameters


def test_stream_gaps_forbidden_recordings_and_eof_discard_candidates(workflow):
    w=workflow;b,y=bouts(w,'stream_guard');p=w.enroll(b,y,user_id='fixture',session_id='long')
    s=stream(w,p);rate=int(w.rate);quiet=np.zeros((rate,8));active=np.ones((rate*2,8))*5
    assert s.feed(quiet,0,recording_id='query')==[]
    assert s.feed(active,rate,recording_id='query')==[] and s.finish()
    assert s.feed(quiet,0,recording_id='query')==[]
    with pytest.raises(ValueError,match='gap'):s.feed(quiet,rate+1,recording_id='query')
    assert s.recording is None and s.detector.next_ is None
    for name in ('source','stream_guard'):
        with pytest.raises(ValueError,match='Disjoint'):s.feed(quiet,0,recording_id=name)
    s.feed(quiet,0,recording_id='one')
    with pytest.raises(ValueError,match='Finish'):s.feed(quiet,rate,recording_id='two')
    with pytest.raises(ValueError,match='rate'):stream(w,p,sample_rate_hz=w.rate+1)
    with pytest.raises(ValueError,match='identities'):stream(w,p,source_recording_ids=())
    with pytest.raises(ValueError,match='channel'):stream(w,p,channel_ids=w.window.channels[::-1])
