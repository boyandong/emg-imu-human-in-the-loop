"""Whole native action capture, frozen window parity and causal automatic release."""
import copy
import json
from pathlib import Path
import pickle
import numpy as np
import pytest
from test_personal_session_stream_v4 import workflow
from emgimu.feature_bank.personal_session_stream_v4 import PersonalSessionStreamV4
from emgimu.feature_bank.personal_session_stream_v5 import PersonalSessionStreamV5
from emgimu.feature_bank.personal_temporal_bouts_v1 import TemporalBoutBatchV1


def service(session='long',channels=None):
    return PersonalSessionStreamV5(workflow(),user_id='fixture',session_id=session,
        channel_ids=tuple(f'CH{i+1}' for i in range(8)) if channels is None else channels)


def signal(label,n=500):
    rng=np.random.default_rng(sum(map(ord,label)))
    if label=='neutral':return rng.normal(0,2,(n,8))
    index=['fist','index_pinch','open_hand'].index(label)
    a=np.roll(np.arange(1,9),2*index)*100
    envelope=np.sin(np.linspace(0,np.pi,n))**2
    return rng.normal(size=(n,8))*a*envelope[:,None]+rng.normal(0,2,(n,8))


def enroll(s,path,kind='personal',offset=0):
    s.command(dict(op='temporal_begin',kind=kind,shots=1));i=offset
    for _ in range(4):
        label=s.info()['temporal']['capture']['next_label'];s.command(dict(op='temporal_trial_start'))
        raw=signal(label)
        s.command(dict(op='emg',raw=raw,indices=np.arange(i,i+len(raw))));i+=len(raw)
        assert not s.command(dict(op='temporal_trial_end'))['calibration_rejected']
    s.command(dict(op='temporal_save',path=str(path)))


def test_full_action_lifecycle_manual_parity_quality_and_saved_inputs(tmp_path):
    s=service();bank=pickle.dumps(s.workflow.bank)
    pp=tmp_path/'long.zip';enroll(s,pp)
    assert s.info()['temporal']['personal_profile_id']
    with np.load(str(pp)+'.bouts.npz',allow_pickle=False) as z:
        assert z['raw'].shape==z['filtered'].shape==(2000,8)
        assert z['offsets'].tolist()==[0,500,1000,1500,2000]
        assert str(z['boundary_kind'])=='complete_cued'
    assert json.loads(Path(str(pp)+'.capture.json').read_text())['settle_samples_discarded']==0
    s.command(dict(op='temporal_mode',mode='manual'));s.command(dict(op='recognize'))
    s.command(dict(op='temporal_action_start'))
    raw=signal('fist',513);s.command(dict(op='emg',raw=raw,indices=np.arange(3000,3513)))
    result=s.command(dict(op='temporal_action_end'))['temporal_events'][0]
    assert result['start']==3000 and result['end']==3513 and result['available_at_sample_index']==3512
    assert result['boundary_kind']=='complete_cued' and result['window_unrepresented_tail_samples']==3
    np.testing.assert_allclose(result['probabilities'],.75*result['base_probabilities']+.125*(result['DTW_probabilities']+result['signature_probabilities']),rtol=0,atol=1e-14)
    assert not s.info()['temporal']['action_open'] and pickle.dumps(s.workflow.bank)==bank
    s.command(dict(op='quality',mode='structural'));s.command(dict(op='recognize'));s.command(dict(op='temporal_action_start'))
    bad=raw.copy();bad[:,2]=0;s.command(dict(op='emg',raw=bad,indices=np.arange(4000,4513)))
    event=s.command(dict(op='temporal_action_end'))['temporal_events'][0]
    assert event['rejected'] and event['label']=='Unknown'
    with pytest.raises(ValueError,match='never accepts'):s.command(dict(op='temporal_action_start',labels={'a':'fist'}))
    t=service('current');t.command(dict(op='temporal_profiles',personal=str(pp)))
    sp=tmp_path/'current.zip';enroll(t,sp,'session')
    restored=service('current');restored.command(dict(op='temporal_profiles',personal=str(pp),session=str(sp)))
    assert restored.info()['temporal']['session_profile_id']==t.info()['temporal']['session_profile_id']
    before=restored.info()['temporal'].copy()
    with pytest.raises(ValueError,match='parent/session'):restored.command(dict(op='temporal_profiles',personal=str(pp),session=str(pp)))
    assert restored.info()['temporal']==before


def test_window_off_identity_auto_chunk_parity_and_direct_interval_oracle(tmp_path):
    s=service();old=PersonalSessionStreamV4(workflow(),user_id='fixture',session_id='long',channel_ids=s.channels)
    for v in (s,old):v.command(dict(op='recognize'))
    raw=signal('fist',513)
    a=s.command(dict(op='emg',raw=raw,indices=np.arange(513)))
    b=old.command(dict(op='emg',raw=raw,indices=np.arange(513)))
    np.testing.assert_array_equal(a['probabilities'],b['probabilities'])
    assert a['confirmed_labels']==b['confirmed_labels'] and a['output_sample_indices']==b['output_sample_indices']
    s.command(dict(op='pause'));pp=tmp_path/'p.zip';enroll(s,pp)
    # Rest arms the detector, followed by a complete action and confirmed release.
    stream=np.concatenate((signal('neutral',400),signal('open_hand',500),signal('neutral',200)))
    outputs=[]
    for chunks in ((1100,),(7,41,1,250,97,314,390)):
        v=service('current');v.command(dict(op='temporal_profiles',personal=str(pp)))
        v.command(dict(op='temporal_mode',mode='auto'));v.command(dict(op='recognize'))
        immutable=pickle.dumps((v.temporal_service.personal,v.temporal_service.long_detector))
        events=[];offset=0
        for n in chunks:
            result=v.command(dict(op='emg',raw=stream[offset:offset+n],indices=np.arange(offset,offset+n)))
            events.extend(result.get('temporal_events',[]));offset+=n
        assert offset==1100 and len(events)==1
        event=events[0];assert event['boundary_kind']=='estimated' and event['decision_after_interval']
        assert event['available_at_sample_index']>=event['end']-1
        assert pickle.dumps((v.temporal_service.personal,v.temporal_service.long_detector))==immutable
        outputs.append(event)
    assert (outputs[0]['start'],outputs[0]['end'])==(outputs[1]['start'],outputs[1]['end'])
    np.testing.assert_allclose(outputs[0]['probabilities'],outputs[1]['probabilities'],rtol=0,atol=1e-12)
    # Reconstruct the detector interval using the exact causal filtering contract.
    from scipy.signal import sosfilt
    x=stream.copy()
    for sos in s.filters:x=sosfilt(sos,x,axis=0)
    e=outputs[0];ids=(e['trial_id'],);records=(e['recording_id'],);start,end=e['start'],e['end']
    batch=TemporalBoutBatchV1((x.astype(np.float32)[start:end],),ids,records,(start,),250.,s.channels,
        s.workflow.preprocessing_id,'estimated')
    from dataclasses import replace
    raw_batch=replace(batch,sequences=(stream[start:end],),preprocessing_id='raw_pre_software_highpass')
    direct=s.temporal_service.adapter.predict(batch,temporal_personal=s.temporal_service.personal,user_id='fixture',
        session_id='current',raw_batch=raw_batch,quality_mode='off')
    np.testing.assert_allclose(e['probabilities'],direct['temporal']['arms']['base_full'][0],rtol=0,atol=1e-12)


def test_gap_overflow_capture_conflicts_and_no_false_completion(tmp_path):
    s=service();s.command(dict(op='temporal_begin',kind='personal',shots=1));s.command(dict(op='temporal_trial_start'))
    s.command(dict(op='emg',raw=signal('neutral',300),indices=np.arange(300)))
    with pytest.raises(ValueError,match='Save/cancel'):s.command(dict(op='decision',use_anchor=False,use_session_routing=False))
    result=s.command(dict(op='emg',raw=signal('neutral',300),indices=np.arange(301,601)))
    assert result['mode']=='idle' and result['temporal']['capture'] is None
    pp=tmp_path/'p.zip';enroll(s,pp)
    s.command(dict(op='temporal_mode',mode='manual'));s.command(dict(op='recognize'));s.command(dict(op='temporal_action_start'))
    with pytest.raises(ValueError,match='exceeded30'):s.command(dict(op='emg',raw=signal('fist',7501),indices=np.arange(7501)))
    assert s.mode=='idle' and not s.info()['temporal']['action_open']
    s.command(dict(op='temporal_mode',mode='auto'));s.command(dict(op='recognize'))
    # A stream beginning in ongoing activity has no observed onset.
    raw=np.random.default_rng(10).normal(0,500,(600,8))
    assert not s.command(dict(op='emg',raw=raw,indices=np.arange(600))).get('temporal_events')
    s.command(dict(op='gap'));assert s.mode=='idle' and s.temporal_service.detector.start_ is None


def test_calibration_quality_no_overwrite_bundle_checksum_and_channel_order(tmp_path):
    s=service();s.command(dict(op='quality',mode='structural'));s.command(dict(op='temporal_begin',kind='personal',shots=1))
    s.command(dict(op='temporal_trial_start'));bad=signal('neutral');bad[:,4]=0
    s.command(dict(op='emg',raw=bad,indices=np.arange(500)))
    assert s.command(dict(op='temporal_trial_end'))['calibration_rejected']
    assert s.info()['temporal']['capture']['completed']==0
    s.command(dict(op='cancel'));pp=tmp_path/'p.zip';enroll(s,pp)
    saved=pp.read_bytes();enroll(s,tmp_path/'other.zip')
    s.command(dict(op='temporal_begin',kind='personal',shots=1))
    for i in range(4):
        label=s.info()['temporal']['capture']['next_label'];s.command(dict(op='temporal_trial_start'))
        s.command(dict(op='emg',raw=signal(label),indices=np.arange(i*500,(i+1)*500)))
        s.command(dict(op='temporal_trial_end'))
    with pytest.raises(FileExistsError):s.command(dict(op='temporal_save',path=str(pp)))
    assert pp.read_bytes()==saved and s.info()['temporal']['capture']['completed']==4
    import zipfile
    damaged=tmp_path/'damaged.zip'
    with zipfile.ZipFile(pp) as source,zipfile.ZipFile(damaged,'w') as out:
        for name in source.namelist():
            payload=source.read(name)
            out.writestr(name,payload+b'corrupt' if name=='detector.pkl' else payload)
    restored=service('current');restored.command(dict(op='temporal_profiles',personal=str(pp)))
    identity=restored.temporal_service.personal.profile_id
    with pytest.raises(ValueError,match='checksum'):restored.command(dict(op='temporal_profiles',personal=str(damaged)))
    assert restored.temporal_service.personal.profile_id==identity
    reversed_service=service('current',tuple(reversed(restored.channels)))
    reversed_service.command(dict(op='temporal_profiles',personal=str(pp)))
    for v in (restored,reversed_service):
        v.command(dict(op='temporal_mode',mode='manual'));v.command(dict(op='recognize'));v.command(dict(op='temporal_action_start'))
    raw=signal('fist',513)
    restored.command(dict(op='emg',raw=raw,indices=np.arange(513)))
    reversed_service.command(dict(op='emg',raw=raw[:,::-1],indices=np.arange(513)))
    a=restored.command(dict(op='temporal_action_end'))['temporal_events'][0]
    b=reversed_service.command(dict(op='temporal_action_end'))['temporal_events'][0]
    np.testing.assert_allclose(a['probabilities'],b['probabilities'],rtol=0,atol=1e-12)
