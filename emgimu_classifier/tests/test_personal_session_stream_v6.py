"""Shared native capture, persistent bound branches and direct action oracles."""
import copy
from dataclasses import replace
import pickle
import numpy as np
import pytest
from scipy.signal import sosfilt

from test_personal_session_stream_v4 import workflow
from test_personal_session_stream_v5 import signal,enroll
from emgimu.feature_bank.personal_session_stream_v3 import _LifecycleProfile
from emgimu.feature_bank.personal_session_stream_v4 import PersonalSessionStreamV4
from emgimu.feature_bank.personal_session_stream_v6 import PersonalSessionStreamV6
from emgimu.feature_bank.personal_temporal_bouts_v1 import TemporalBoutBatchV1


def service(session='long',channels=None):
    return PersonalSessionStreamV6(workflow(),user_id='fixture',session_id=session,
        channel_ids=tuple(f'CH{i+1}' for i in range(8)) if channels is None else channels)


def test_shared_capture_saves_every_sample_once_and_binds_both_branches(tmp_path):
    s=service();source=pickle.dumps(s.workflow.bank)
    path=tmp_path/'long.zip';enroll(s,path)
    p=s.temporal_service.personal;info=s.info();cost=p.calibration_cost
    assert info['schema']=='song_personal_session_stream_v6' and info['profile_schema']=='joint_bout_profile_v1'
    assert info['personal_trials']==cost['unique_native_calibration_trials']==4
    assert info['personal_profile_id']==p.window.profile_id and info['temporal']['personal_profile_id']==p.profile_id
    assert cost['native_signal_samples']==2000 and cost['native_signal_seconds']==8.
    assert cost['counted_once'] and cost['quality_mode']=='off' and cost['begin_to_save_elapsed_seconds']>0
    assert p.detector_receipt['settle_samples_discarded']==0 and p.detector_receipt['raw_and_filtered_samples_saved']
    assert all(x.shape==(500,8) for x in p.raw_calibration.sequences+p.calibration.sequences)
    restored=service('current');restored.command(dict(op='temporal_profiles',personal=str(path)))
    assert restored.personal.profile_id==s.personal.profile_id
    sp=tmp_path/'session.zip';enroll(restored,sp,'session')
    local=restored.temporal_service.session
    assert local.parent_profile_id==p.profile_id and restored.session.profile_id==local.window.profile_id
    assert restored.info()['session_trials']==4 and restored.info()['temporal']['session_calibration_cost']['counted_once']
    assert pickle.dumps(s.workflow.bank)==source


@pytest.mark.parametrize('anchor,routing',[(False,False),(False,True),(True,True)])
def test_bound_window_stream_and_manual_joint_action_match_independent_calls(tmp_path,anchor,routing):
    s=service();path=tmp_path/'p.zip';enroll(s,path)
    s=service('current');s.command(dict(op='temporal_profiles',personal=str(path)))
    sp=tmp_path/'s.zip';enroll(s,sp,'session')
    p,local=s.temporal_service.personal,s.temporal_service.session
    s.command(dict(op='decision',use_anchor=anchor,use_session_routing=routing))
    old=PersonalSessionStreamV4(workflow(),user_id='fixture',session_id='current',channel_ids=s.channels)
    old.personal=_LifecycleProfile(p.window);old.session=_LifecycleProfile(local.window)
    old.command(dict(op='decision',use_anchor=anchor,use_session_routing=routing))
    s.command(dict(op='temporal_mode',mode='manual'))
    for v in (s,old):v.command(dict(op='recognize'))
    s.command(dict(op='temporal_action_start'));raw=signal('fist',513)
    before=pickle.dumps((p,local,s.workflow.bank),protocol=4)
    a=s.command(dict(op='emg',raw=raw,indices=np.arange(3000,3513)))
    b=old.command(dict(op='emg',raw=raw,indices=np.arange(3000,3513)))
    np.testing.assert_array_equal(a['probabilities'],b['probabilities'])
    assert a['confirmed_labels']==b['confirmed_labels']
    event=s.command(dict(op='temporal_action_end'))['temporal_events'][0]
    x=raw.copy()
    for sos in s.filters:x=sosfilt(sos,x,axis=0)
    bout=TemporalBoutBatchV1((x.astype(np.float32),),(event['trial_id'],),(event['recording_id'],),(3000,),
        250.,s.channels,s.workflow.preprocessing_id,'complete_cued')
    direct=s.temporal_service.joint.predict(bout,personal=p,session=local,user_id='fixture',session_id='current',
        raw_batch=replace(bout,sequences=(raw,),preprocessing_id='raw_pre_software_highpass'),
        use_anchor=anchor,use_session_routing=routing)
    np.testing.assert_allclose(event['probabilities'],direct['probabilities'][0],rtol=0,atol=1e-14)
    np.testing.assert_allclose(event['probabilities'],.75*event['base_probabilities']+
        .125*(event['DTW_probabilities']+event['signature_probabilities']),rtol=0,atol=1e-14)
    assert event['shared_registration'] and event['window_unrepresented_tail_samples']==3
    assert event['joint_personal_profile_id']==p.profile_id and event['joint_session_profile_id']==local.profile_id
    assert pickle.dumps((p,local,s.workflow.bank),protocol=4)==before


def test_estimated_auto_chunk_parity_reversed_channels_and_quality_unknown(tmp_path):
    initial=service();path=tmp_path/'p.zip';enroll(initial,path)
    raw=np.concatenate((signal('neutral',400),signal('open_hand'),signal('neutral',200)))
    events=[]
    for reverse,chunks in ((False,(1100,)),(True,(7,41,1,250,97,314,390))):
        s=service('current',tuple(reversed(initial.channels)) if reverse else initial.channels)
        s.command(dict(op='temporal_profiles',personal=str(path)))
        s.command(dict(op='temporal_mode',mode='auto'));s.command(dict(op='recognize'))
        offset=0;result=[]
        for size in chunks:
            part=raw[offset:offset+size];part=part[:,::-1] if reverse else part
            r=s.command(dict(op='emg',raw=part,indices=np.arange(offset,offset+size)));offset+=size
            result.extend(r.get('temporal_events',[]))
        assert len(result)==1 and result[0]['boundary_kind']=='estimated'
        assert result[0]['available_at_sample_index']>=result[0]['end']-1
        events.append(result[0])
    assert (events[0]['start'],events[0]['end'])==(events[1]['start'],events[1]['end'])
    np.testing.assert_allclose(events[0]['probabilities'],events[1]['probabilities'],rtol=0,atol=1e-12)
    s=service('current');s.command(dict(op='temporal_profiles',personal=str(path)))
    s.command(dict(op='quality',mode='structural'));s.command(dict(op='temporal_mode',mode='manual'))
    s.command(dict(op='recognize'));s.command(dict(op='temporal_action_start'))
    bad=signal('fist');bad[:,2]=0;s.command(dict(op='emg',raw=bad,indices=np.arange(500)))
    event=s.command(dict(op='temporal_action_end'))['temporal_events'][0]
    assert event['rejected'] and event['label']=='Unknown'


def test_failed_save_load_capture_conflicts_gap_and_overflow_are_atomic(tmp_path):
    s=service();path=tmp_path/'p.zip';enroll(s,path)
    original=path.read_bytes();p=s.temporal_service.personal
    s.command(dict(op='temporal_begin',kind='personal',shots=1))
    for i in range(4):
        label=s.info()['temporal']['capture']['next_label'];s.command(dict(op='temporal_trial_start'))
        s.command(dict(op='emg',raw=signal(label),indices=np.arange(i*500,(i+1)*500)));s.command(dict(op='temporal_trial_end'))
    with pytest.raises(FileExistsError):s.command(dict(op='temporal_save',path=str(path)))
    assert path.read_bytes()==original and s.info()['temporal']['capture']['completed']==4
    assert s.temporal_service.personal is p and s.personal.profile_id==p.window.profile_id
    with pytest.raises(ValueError,match='Save/cancel'):s.command(dict(op='decision',use_anchor=True,use_session_routing=True))
    s.command(dict(op='cancel'));assert s.info()['temporal']['capture'] is None
    s.command(dict(op='recognize'))
    with pytest.raises(FileNotFoundError):s.command(dict(op='temporal_profiles',personal=str(tmp_path/'missing.zip')))
    assert s.mode=='idle' and s.temporal_service.personal is p and s.personal.profile_id==p.window.profile_id
    for op in ('profiles','begin','trial','save'):
        with pytest.raises(ValueError,match='Shared backend'):s.command(dict(op=op))
    with pytest.raises(ValueError,match='never accepts'):s.command(dict(op='recognize',labels={}))
    s.command(dict(op='temporal_begin',kind='personal',shots=1));s.command(dict(op='temporal_trial_start'))
    s.command(dict(op='emg',raw=signal('neutral',300),indices=np.arange(300)))
    r=s.command(dict(op='emg',raw=signal('neutral',300),indices=np.arange(301,601)))
    assert r['reset_reason'] and s.mode=='idle' and s.info()['temporal']['capture'] is None
    s.command(dict(op='temporal_mode',mode='manual'));s.command(dict(op='recognize'));s.command(dict(op='temporal_action_start'))
    with pytest.raises(ValueError,match='exceeded30'):s.command(dict(op='emg',raw=signal('fist',7501),indices=np.arange(7501)))
    assert s.mode=='idle' and not s.info()['temporal']['action_open']
