import pickle
from pathlib import Path
import numpy as np
import pytest
from emgimu.feature_bank.song_f0_stream_v1 import SongF0StreamV1
from emgimu.feature_bank.core import FeatureBatch

ROOT=Path(__file__).resolve().parents[1]


def bank():return pickle.loads((ROOT/'feature_bank/models/song_f0_250hz_v1.pkl').read_bytes())


@pytest.mark.parametrize('arm',['pooled_source_threshold','rest_only_threshold'])
def test_chunk_causality_probabilities_and_confirmation_match_independent_one_pass(arm):
    runtime=bank();before=pickle.dumps(runtime);raw=np.random.default_rng(83).normal(size=(411,8))
    stream=SongF0StreamV1(runtime,arm=arm,recording_id='fixture')
    records=[]
    for start,stop in zip([0,1,17,48,49,50,68,100,310],[1,17,48,49,50,68,100,310,411]):
        out=stream.push(raw[start:stop],first_sample_index=start,sample_rate_hz=250.)
        assert all(r['end_sample']<stop for r in out);records+=out
    ends=np.arange(49,411,10);assert [r['end_sample'] for r in records]==ends.tolist()
    filtered=runtime.new_filter().process(raw)
    windows=np.stack([filtered[e-49:e+1] for e in ends]);family,model=runtime.models_[arm]
    columns=[list(model[-1].classes_).index(c) for c in runtime.classes_]
    q=model.predict_proba(family.transform(FeatureBatch(windows,250.)))[:,columns]
    np.testing.assert_allclose(np.stack([r['probabilities'] for r in records]),q,atol=1e-12,rtol=0)
    # Independent explicit two-consecutive-label confirmation recurrence.
    stable=-1;candidate=None;count=0;expected=[]
    for label in q.argmax(1):
        if label==stable:candidate=None;count=0
        else:
            count=count+1 if label==candidate else 1;candidate=label
            if count>=2:stable=int(label);candidate=None;count=0
        expected.append(stable)
    assert [r['confirmed_class_index'] for r in records]==expected and expected[0]==-1
    assert records[0]['available_at_nominal_seconds']==.2
    assert pickle.dumps(runtime)==before and len(stream.buffer)<=49
    # Future samples cannot change previously emitted probabilities.
    np.testing.assert_allclose(records[0]['probabilities'],q[0],atol=1e-12,rtol=0)


def test_explicit_reset_restarts_filter_window_and_unknown_confirmation():
    r=bank();raw=np.random.default_rng(9).normal(size=(70,8))
    s=SongF0StreamV1(r,arm='pooled_source_threshold',recording_id='fixture')
    s.push(raw,first_sample_index=0,sample_rate_hz=250.)
    s.reset(first_sample_index=200)
    output=s.push(raw,first_sample_index=200,sample_rate_hz=250.)
    assert [v['end_sample'] for v in output]==[249,259,269]
    assert output[0]['confirmed_class_index']==-1
    fresh=SongF0StreamV1(r,arm='pooled_source_threshold',recording_id='fixture').push(raw,first_sample_index=0,sample_rate_hz=250.)
    np.testing.assert_allclose(np.stack([v['probabilities'] for v in output]),np.stack([v['probabilities'] for v in fresh]),atol=1e-12,rtol=0)


@pytest.mark.parametrize('case',['gap','duplicate','rate','channels','nan'])
def test_invalid_chunks_fail_before_mutating_stream(case):
    s=SongF0StreamV1(bank(),arm='rest_only_threshold',recording_id='fixture')
    raw=np.ones((12,8));s.push(raw,first_sample_index=0,sample_rate_hz=250.)
    before=pickle.dumps(s);first=12;rate=250.
    if case=='gap':first=13
    elif case=='duplicate':first=0
    elif case=='rate':rate=200.
    elif case=='channels':raw=raw[:,:7]
    else:raw=raw.copy();raw[0,0]=np.nan
    with pytest.raises(ValueError):s.push(raw,first_sample_index=first,sample_rate_hz=rate)
    assert pickle.dumps(s)==before
