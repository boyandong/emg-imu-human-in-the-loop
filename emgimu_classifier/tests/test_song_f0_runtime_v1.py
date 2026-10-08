import hashlib
import inspect
import json
import pickle
from pathlib import Path
import numpy as np
import pytest
from scipy.signal import butter,iirnotch,sosfilt,tf2sos
from emgimu.feature_bank.core import FeatureBatch

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def runtime():
    d=json.loads((ROOT/'feature_bank/SONG_F0_RUNTIME_ACCEPTANCE_V1.json').read_text(encoding='utf8'))
    packed=(ROOT/d['package_path']).read_bytes()
    assert hashlib.sha256(packed).hexdigest()==d['package_sha256']
    return pickle.loads(packed)


def test_song_runtime_source_hashes_native_trial_coverage_and_boundaries(runtime):
    d=json.loads((ROOT/'feature_bank/SONG_F0_RUNTIME_ACCEPTANCE_V1.json').read_text(encoding='utf8'))
    for path,digest in d['source_sha256'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
    assert d['source_state_exactly_recovered'] and d['source_state_immutable'] and d['source_windows']==849
    assert d['prediction_rows']==568 and len(d['records'])==4
    assert {(r['session'],r['arm'],r['trials']) for r in d['records']}=={(s,a,n) for s,n in [('S03',140),('S04',144)] for a in runtime.models_}
    assert all(r['probability_max_error']<1e-12 for r in d['records'])
    assert not any(d[k] for k in ['physical_validation_proven','default_promoted','completion_proven'])


def test_stream_filter_matches_independent_fixed_causal_signal_and_chunk_boundaries(runtime):
    rng=np.random.default_rng(73);raw=rng.normal(size=(703,8))
    reference=sosfilt(butter(4,40.,btype='highpass',fs=250.,output='sos'),raw,axis=0)
    for hz in [50.,100.]:reference=sosfilt(tf2sos(*iirnotch(hz,Q=30.,fs=250.)),reference,axis=0)
    reference=reference.astype(np.float32)
    f=runtime.new_filter();pieces=[];starts=[0,1,17,100,303,703]
    before=pickle.dumps(runtime)
    for a,b in zip(starts,starts[1:]):pieces.append(f.process(raw[a:b]))
    np.testing.assert_array_equal(np.concatenate(pieces),reference)
    assert f.samples==len(raw)
    f.reset();assert f.samples==0
    np.testing.assert_array_equal(f.process(raw),reference)
    assert pickle.dumps(runtime)==before
    # Resetting at every window is demonstrably a different preprocessing policy.
    wrong=np.concatenate([runtime.new_filter().process(raw[a:a+50]) for a in range(0,700,50)])
    assert np.max(abs(wrong-reference[:700]))>.01


def test_raw_recording_and_probability_mean_match_window_api_without_labels(runtime):
    raw=np.random.default_rng(91).normal(size=(210,8));starts=np.array([3,59,120]);ids=np.array(['eval_b','eval_a','eval_a']);offsets=np.array([0,1,0])
    before=pickle.dumps(runtime)
    assert 'labels' not in inspect.signature(runtime.predict_recording).parameters
    output=runtime.predict_recording(raw,starts,ids,window_offsets=offsets,sample_rate_hz=250.)
    filtered=runtime.new_filter().process(raw);windows=np.stack([filtered[s:s+50] for s in starts]);batch=FeatureBatch(windows,250.)
    assert output['trial_ids']==('eval_a','eval_b') and not output['default_promoted']
    for arm,(family,model) in runtime.models_.items():
        cols=[list(model[-1].classes_).index(c) for c in runtime.classes_]
        q=model.predict_proba(family.transform(batch))[:,cols]
        expected=np.stack([q[[2,1]].mean(0),q[0]])
        np.testing.assert_array_equal(output['arms'][arm]['probabilities'],expected)
        order=np.array([2,0,1])
        permuted=runtime.predict_windows(batch.take(order),ids[order],window_offsets=offsets[order],preprocessing_id=runtime.preprocessing_id)
        np.testing.assert_allclose(output['arms'][arm]['probabilities'],permuted['arms'][arm]['probabilities'],atol=1e-12,rtol=0)
    assert pickle.dumps(runtime)==before


@pytest.mark.parametrize('case',['rate','samples','channels','raw_preprocessing','source_overlap','blank_ids','offset_duplicates','offset_fractional','offset_shape'])
def test_window_contract_and_source_identity_reject_incompatible_calls(runtime,case):
    b=FeatureBatch(np.ones((2,50,8)),250.);ids=np.array(['eval','eval'],dtype=object);kw={'window_offsets':np.arange(2),'preprocessing_id':runtime.preprocessing_id}
    if case=='rate':b=FeatureBatch(b.emg,200.)
    elif case=='samples':b=FeatureBatch(b.emg[:,:40],250.)
    elif case=='channels':b=FeatureBatch(b.emg[:,:,:7],250.)
    elif case=='raw_preprocessing':kw['preprocessing_id']='unfiltered'
    elif case=='source_overlap':ids[:]=runtime.source_trial_ids_[0]
    elif case=='blank_ids':ids[:]=''
    elif case=='offset_duplicates':kw['window_offsets']=np.array([0,0])
    elif case=='offset_fractional':kw['window_offsets']=np.array([0.,1.])
    else:kw['window_offsets']=np.array([0])
    with pytest.raises(ValueError):runtime.predict_windows(b,ids,**kw)


@pytest.mark.parametrize('case',['rate','outside','negative','fractional','nonfinite','channels'])
def test_recording_filter_rejects_invalid_calls_without_mutating_model(runtime,case):
    raw=np.ones((100,8));starts=np.array([0]);rate=250.
    if case=='rate':rate=200.
    elif case=='outside':starts=np.array([51])
    elif case=='negative':starts=np.array([-1])
    elif case=='fractional':starts=np.array([.5])
    elif case=='nonfinite':raw[0,0]=np.nan
    else:raw=raw[:,:7]
    before=pickle.dumps(runtime)
    with pytest.raises(ValueError):runtime.predict_recording(raw,starts,['eval'],window_offsets=np.array([0]),sample_rate_hz=rate)
    assert pickle.dumps(runtime)==before
