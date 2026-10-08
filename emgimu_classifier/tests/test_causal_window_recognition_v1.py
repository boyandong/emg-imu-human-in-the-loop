import pickle
import numpy as np
import pytest
from scipy.special import expit
from emgimu.feature_bank.causal_window_recognition_v1 import CausalWindowRecognizerV1, transition_hold_diagnostics


class MeanFamily:
    samples_ = 40
    sample_rate_hz_ = 200.
    def _check(self): pass
    def transform(self, batch): return batch.emg.mean(1)[:, :1]


class FixedModel:
    classes_ = np.array([0, 1])
    def predict_proba(self, x):
        p = expit(x[:, 0]); return np.column_stack([1-p, p])


def test_chunk_parity_exact_trailing_windows_no_future_and_warmup():
    x = np.broadcast_to(np.sin(np.arange(173)[:, None]/19), (173,8)).copy()
    family, model = MeanFamily(), FixedModel()
    frozen = pickle.dumps((family,model))
    outputs = []
    for size in (1, 7, 37, 173):
        stream = CausalWindowRecognizerV1(family,model,sample_rate_hz=200.)
        emitted, probabilities = [], []
        for start in range(0,len(x),size):
            ends,p = stream.push(x[start:start+size]); emitted.extend(ends); probabilities.extend(p)
            assert all(e < min(start+size,len(x)) for e in ends)
            assert len(stream.buffer) <= 39
        outputs.append((np.asarray(emitted),np.asarray(probabilities)))
    expected_ends = np.arange(39,len(x),10)
    direct = expit(np.array([sum(x[e-39:e+1,0])/40 for e in expected_ends]))
    for ends,p in outputs:
        np.testing.assert_array_equal(ends,expected_ends)
        np.testing.assert_allclose(p[:,1],direct,rtol=0,atol=1e-12)
        np.testing.assert_allclose(p,outputs[0][1],rtol=0,atol=1e-12)
    altered = x.copy(); altered[100:] = 10000
    a=CausalWindowRecognizerV1(family,model,sample_rate_hz=200.)
    b=CausalWindowRecognizerV1(family,model,sample_rate_hz=200.)
    ends,p=a.push(x); _,other=b.push(altered)
    np.testing.assert_array_equal(p[ends<100],other[ends<100])
    assert pickle.dumps((family,model))==frozen


def test_stream_contract_failures_preserve_buffer_state():
    stream=CausalWindowRecognizerV1(MeanFamily(),FixedModel(),sample_rate_hz=200.)
    assert not len(stream.push(np.zeros((39,8)))[0])
    state=pickle.dumps(stream)
    for bad in (np.zeros((2,7)),np.full((1,8),np.nan)):
        with pytest.raises(ValueError): stream.push(bad)
        assert pickle.dumps(stream)==state
    with pytest.raises(ValueError,match='rate'): CausalWindowRecognizerV1(MeanFamily(),FixedModel(),sample_rate_hz=250.)


def test_transition_holds_penalize_delay_third_class_flicker_and_unknown():
    truth=np.repeat([0,1,2],100).astype(int)
    perfect=transition_hold_diagnostics(truth,truth,rate_hz=200,half_buffer_samples=10)
    assert perfect['correct_transitions']==perfect['eligible_transitions']==2
    assert perfect['transition_hold_accuracy']==1
    delayed=truth.copy();delayed[100:112]=0
    assert transition_hold_diagnostics(truth,delayed,rate_hz=200,half_buffer_samples=10)['correct_transitions']==1
    third=truth.copy();third[100]=2
    r=transition_hold_diagnostics(truth,third,rate_hz=200,half_buffer_samples=10)
    assert r['events'][0]['reasons']==['other_label_in_reaction']
    flicker=truth.copy();flicker[150]=0
    r=transition_hold_diagnostics(truth,flicker,rate_hz=200,half_buffer_samples=10)
    assert r['correct_transitions']==1 and r['maintenance_switches']==2
    unknown=np.full_like(truth,-1)
    assert transition_hold_diagnostics(truth,unknown,rate_hz=200,half_buffer_samples=10)['correct_transitions']==0
    assert transition_hold_diagnostics(np.zeros(20,dtype=int),np.zeros(20,dtype=int),rate_hz=200,half_buffer_samples=10)['transition_hold_accuracy'] is None
    close=np.repeat([0,1,2,0],5).astype(int)
    assert transition_hold_diagnostics(close,close,rate_hz=200,half_buffer_samples=4)['eligible_transitions']==0
