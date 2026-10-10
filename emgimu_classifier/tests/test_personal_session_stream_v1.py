"""Independent filter, trial-pooling, persistence and stream-state checks."""
from pathlib import Path
import pickle
import numpy as np
import pytest
from scipy.signal import sosfilt
from emgimu.feature_bank.personal_session_cli_v1 import load_workflow
from emgimu.feature_bank.personal_session_stream_v1 import PersonalSessionStreamV1
from emgimu.feature_bank.core import FeatureBatch
from benchmarks.song_real8_study import _filter_emg

ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'benchmarks/song_real8/song_personal_session_v1/source_bank.pkl'
ACCEPTANCE=ROOT/'feature_bank/SONG_PERSONAL_SESSION_ACCEPTANCE_V1.json'


def service(session='fixture1',channels=None):
    w=load_workflow(PACKAGE,ACCEPTANCE)
    return PersonalSessionStreamV1(w,user_id='fixture',session_id=session,channel_ids=channels or w.channels)


def capture(s):
    s.command(dict(op='begin',kind='personal',shots=1,settle_samples=30,hold_samples=80))
    values=np.random.default_rng(19).normal(size=(440,8))*100
    for i in range(4):
        s.command(dict(op='trial'))
        state=s.ingest(values[i*110:(i+1)*110],np.arange(i*110,(i+1)*110))
        assert state['capture']['completed']==i+1
    return s.capture


def test_stream_filter_single_window_probabilities_and_chunking_are_exact():
    raw=np.random.default_rng(51).integers(-2000,2000,size=(300,8))
    s=service();source=pickle.dumps(s.workflow.bank)
    s.command(dict(op='recognize'))
    result=s.ingest(raw,np.arange(len(raw)))
    ends=np.arange(49,300,10);filtered=_filter_emg(raw,'causal')
    batch=FeatureBatch(np.stack([filtered[end-49:end+1] for end in ends]),250.)
    ids=np.array([f'fixture:{i:04}' for i in range(len(ends))])
    expected=s.workflow.bank.predict(batch,ids,window_offsets=np.zeros(len(ids),int),user_id='fixture')
    np.testing.assert_allclose(result['probabilities'],expected['probabilities'],rtol=0,atol=1e-12)
    other=service();other.command(dict(op='recognize'));q=[];labels=[]
    for start in range(0,300,13):
        r=other.ingest(raw[start:start+13],np.arange(start,min(start+13,300)))
        q.extend(r.get('probabilities',[]));labels.extend(r.get('confirmed_labels',[]))
    np.testing.assert_allclose(q,result['probabilities'],rtol=0,atol=1e-12)
    assert labels==result['confirmed_labels'] and labels[0] is None
    assert pickle.dumps(s.workflow.bank)==source


def test_guided_calibration_pools_by_trial_and_roundtrips(tmp_path):
    s=service();data=capture(s)
    assert len(data['windows'])==16 and len(data['labels'])==4 # Four windows per trial, not 16 shots.
    profile=s.workflow.enroll_user(FeatureBatch(np.stack(data['windows']),250.),data['ids'],data['labels'],
        window_offsets=data['offsets'],**s.kwargs)
    path=tmp_path/'personal.zip';r=s.command(dict(op='save',path=str(path)))
    assert r['personal_trials']==4 and r['personal_profile_id']==profile.profile_id
    from emgimu.feature_bank.frozen_emg_bank_cli_v1 import load_windows
    restored,ids,offsets=load_windows(r['calibration_windows_path'])
    np.testing.assert_array_equal(restored.emg,np.stack(data['windows']))
    assert ids.tolist()==data['ids'] and offsets.tolist()==data['offsets']
    assert Path(r['calibration_labels_path']).is_file() and Path(r['calibration_audit_path']).is_file()
    fresh=service('fixture2');fresh.command(dict(op='profiles',personal=str(path)))
    np.testing.assert_allclose(fresh.personal.fusion_state.fusion_state.weights,
                               profile.fusion_state.fusion_state.weights,rtol=0,atol=0)
    fresh.command(dict(op='begin',kind='session',shots=1,settle_samples=0,hold_samples=50))
    for i in range(4):
        fresh.command(dict(op='trial'))
        fresh.ingest(np.random.default_rng(i).normal(size=(50,8))*100,np.arange(i*50,(i+1)*50))
    current=tmp_path/'session.zip';fresh.command(dict(op='save',path=str(current)))
    reloaded=service('fixture2');reloaded.command(dict(op='profiles',personal=str(path),session=str(current)))
    assert reloaded.info()['session_trials']==4
    raw=np.random.default_rng(4).normal(size=(100,8))*100
    for item in (fresh,reloaded):item.command(dict(op='recognize'))
    a=fresh.ingest(raw,np.arange(100));b=reloaded.ingest(raw,np.arange(100))
    np.testing.assert_allclose(a['probabilities'],b['probabilities'],rtol=0,atol=0)
    assert a['confirmed_labels']==b['confirmed_labels']
    with pytest.raises(ValueError):service('wrong').command(dict(op='profiles',personal=str(path),session=str(current)))


def test_contract_errors_do_not_replace_profile_and_gap_cancels_capture(tmp_path):
    s=service();capture(s);path=tmp_path/'personal.zip';s.command(dict(op='save',path=str(path)))
    old=s.personal
    with pytest.raises(ValueError,match='requires a personal'):s.command(dict(op='begin',kind='session'))
    with pytest.raises(ValueError):s.command(dict(op='profiles',session=str(path)))
    assert s.personal is old
    s.command(dict(op='begin',kind='personal'));s.command(dict(op='trial'))
    s.ingest(np.zeros((10,8)),np.arange(10))
    r=s.ingest(np.zeros((10,8)),np.arange(11,21))
    assert r['mode']=='idle' and r['capture'] is None and 'reset_reason' in r
    assert s.personal is old
    with pytest.raises(ValueError):s.command(dict(op='save',path=str(path)))
    s.command(dict(op='recognize'));s.ingest(np.zeros((60,8)),np.arange(60))
    assert s.active is not None
    s.command(dict(op='gap'));assert s.active is None
    s.command(dict(op='recognize'))
    assert 'probabilities' not in s.ingest(np.zeros((49,8)),np.arange(100,149))


def test_explicit_channel_order_and_invalid_inputs():
    order=[f'CH{i}' for i in range(8,0,-1)]
    raw=np.random.default_rng(99).normal(size=(80,8))*100
    a=service();b=service(channels=order)
    for s in (a,b):s.command(dict(op='recognize'))
    np.testing.assert_allclose(a.ingest(raw,np.arange(80))['probabilities'],
                              b.ingest(raw[:,::-1],np.arange(80))['probabilities'],rtol=0,atol=1e-12)
    for x,ids in [(np.zeros((3,7)),np.arange(3)),(np.full((3,8),np.nan),np.arange(3)),
                  (np.zeros((3,8)),np.arange(3,dtype=float))]:
        with pytest.raises(ValueError):a.ingest(x,ids)
    with pytest.raises(ValueError):service(channels=['CH1']*8)
