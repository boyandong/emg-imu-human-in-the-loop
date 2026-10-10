import pickle
import numpy as np
import pytest
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.native_class_transition_stream_v1 import NativeClassTransitionStreamV1,window_scores
from emgimu.feature_bank.class_transition_bouts_v1 import ConfirmedClassBoutDetectorV1
from test_native_joint_bout_workflow_v2 import workflow


def make(w,**changes):
    kwargs=dict(source_recording_ids=('source',),forbidden_recording_ids=('cal',),user_id='fixture',rest_label='relax',
        sample_rate_hz=w.rate,channel_ids=w.window.channels,source_channel_ids=w.window.channels,
        preprocessing_id=w.window.preprocessing_id,source_preprocessing_id=w.window.preprocessing_id)
    kwargs.update(changes);return NativeClassTransitionStreamV1(w.window.bank,**kwargs)


def test_online_window_scores_equal_batch_source_decisions_and_do_not_mutate_bank(workflow,monkeypatch):
    import emgimu.feature_bank.native_class_transition_stream_v1 as module
    w=workflow;rng=np.random.default_rng(45);x=rng.normal(size=(int(5*w.rate),8)).astype(np.float32)
    x[:int(w.rate)]*=.01;x[int(3*w.rate):]*=.01
    bank=w.window.bank;before=pickle.dumps(bank);ends=np.arange(w.samples,len(x)+1,w.hop)
    windows=np.stack([x[e-w.samples:e] for e in ends])
    expected=window_scores(bank,windows,ends,recording_id='query',user_id='fixture')
    observed=[];original=module.window_scores
    def capture(*args,**kwargs):
        q=original(*args,**kwargs);observed.append(q);return q
    monkeypatch.setattr(module,'window_scores',capture)
    stream=make(w);events=[];pos=0
    for stop in (13,41,79,251,613,len(x)):
        events+=stream.feed(x[pos:stop],pos,recording_id='query');pos=stop
    np.testing.assert_allclose(np.concatenate(observed),expected,rtol=0,atol=1e-14)
    d=ConfirmedClassBoutDetectorV1(bank_id=bank.bank_id_,class_names=bank.classes_,rest_label='relax',sample_rate_hz=w.rate)
    direct=d.feed(x,0,recording_id='query',probabilities=expected,window_end_indices=ends,bank_id=bank.bank_id_)
    assert [(e['start'],e['end'],e['estimated_class']) for e in events]==[(e['start'],e['end'],e['estimated_class']) for e in direct]
    assert pickle.dumps(bank)==before


def test_native_stream_provenance_rate_and_gap_contracts(workflow):
    w=workflow;s=make(w);x=np.zeros((100,8),np.float32)
    for recording in ('source','cal'):
        with pytest.raises(ValueError,match='Disjoint'):s.feed(x,0,recording_id=recording)
    s.feed(x,0,recording_id='query')
    with pytest.raises(ValueError):s.feed(x,101,recording_id='query')
    assert s.next is None and s.detector.next_sample is None
    with pytest.raises(ValueError,match='sensor'):make(w,sample_rate_hz=w.rate+1)
    with pytest.raises(ValueError,match='sensor'):make(w,channel_ids=w.window.channels[::-1])
