"""Quality/recognition/capture integration without changing the frozen V1 route."""
from pathlib import Path
import numpy as np
import pytest
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.personal_session_cli_v1 import load_workflow
from emgimu.feature_bank.personal_session_stream_v1 import PersonalSessionStreamV1
from emgimu.feature_bank.personal_session_stream_v2 import PersonalSessionStreamV2
from emgimu.feature_bank.source_quality_gate_v1 import SourceQualityGateV1

ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'benchmarks/song_real8/song_personal_session_v1/source_bank.pkl'
ACCEPTANCE=ROOT/'feature_bank/SONG_PERSONAL_SESSION_ACCEPTANCE_V1.json'


def service():
    w=load_workflow(PACKAGE,ACCEPTANCE)
    x=np.random.default_rng(71).integers(-1000,1000,size=(40,50,8))
    gate=SourceQualityGateV1(channel_ids=w.channels,adc_range=(-8388608,8388607),adc_range_provenance='signed24 fixture').fit(
        FeatureBatch(x,250.),[f'source_q{i}' for i in range(40)])
    return PersonalSessionStreamV2(w,gate,user_id='fixture',session_id='recording',channel_ids=w.channels)


def test_off_matches_v1_and_quality_requires_raw_replay(tmp_path):
    s=service();old=PersonalSessionStreamV1(s.workflow,user_id=s.user,session_id=s.session_id,channel_ids=s.channels)
    for item in (s,old):item.command(dict(op='recognize'))
    raw=np.random.default_rng(15).integers(-1000,1000,size=(150,8))
    for start in range(0,150,17):
        a=s.ingest(raw[start:start+17],np.arange(start,min(start+17,150)))
        b=old.ingest(raw[start:start+17],np.arange(start,min(start+17,150)))
        if 'probabilities' in b:
            np.testing.assert_allclose(a['probabilities'],b['probabilities'],rtol=0,atol=1e-12)
            assert a['confirmed_labels']==b['confirmed_labels']
    s.command(dict(op='quality',mode='structural'))
    assert s.info()['mode']=='idle' and s.quality_active is None
    with pytest.raises(ValueError,match='separate raw'):s.command(dict(op='replay',path=str(tmp_path/'windows.npz')))


def test_rejected_windows_clear_gesture_and_require_two_new_valid_frames(monkeypatch):
    s=service();s.command(dict(op='quality',mode='structural'))
    def predictable(batch,ids,offsets):
        axis=tuple(sorted(ids));q=np.tile([.91,.03,.03,.03],(len(axis),1))
        s.provider_readout=dict(probabilities={g:q for g in s.workflow.bank.providers_},trial_ids=axis)
        return dict(probabilities=q,class_names=s.workflow.bank.class_names_,trial_ids=axis,
                    weights=s.workflow.bank.policy_.population,provider_names=s.workflow.bank.providers_)
    monkeypatch.setattr(s,'_predict',predictable)
    s.command(dict(op='recognize'))
    raw=np.random.default_rng(21).integers(-1000,1000,size=(270,8));raw[100:170,2]=0
    a=s.ingest(raw[:100],np.arange(100));assert a['confirmed_labels'][-1]=='fist'
    b=s.ingest(raw[100:170],np.arange(100,170));assert b['quality_rejected'][-1] and b['confirmed_labels'][-1] is None
    c=s.ingest(raw[170:],np.arange(170,270));valid=np.flatnonzero(~c['quality_rejected'])
    assert c['confirmed_labels'][valid[0]] is None and c['confirmed_labels'][valid[1]]=='fist'
    s.command(dict(op='gap'));assert s.quality_active is None and s.mode=='idle'


def test_bad_guided_trial_is_not_counted_and_raw_companion_is_persisted(tmp_path):
    s=service();s.command(dict(op='quality',mode='structural'))
    s.command(dict(op='begin',kind='personal',shots=1,settle_samples=0,hold_samples=50));s.command(dict(op='trial'))
    raw=np.random.default_rng(5).integers(-1000,1000,size=(50,8));bad=raw.copy();bad[:,2]=0
    r=s.ingest(bad,np.arange(50));assert r['calibration_rejected'] and r['capture']['completed']==0
    assert not s.capture['labels'] and not s.capture['windows']
    for i in range(4):
        s.command(dict(op='trial'));s.ingest(raw,np.arange(50+i*50,100+i*50))
    assert s.info()['capture']['completed']==4
    path=tmp_path/'personal.zip';r=s.command(dict(op='save',path=str(path)))
    with np.load(r['calibration_raw_windows_path'],allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['emg'],np.stack([raw]*4))
    assert s.personal is not None and s.info()['personal_trials']==4
    s.command(dict(op='begin',kind='personal'))
    with pytest.raises(ValueError):s.command(dict(op='quality',mode='off'))
