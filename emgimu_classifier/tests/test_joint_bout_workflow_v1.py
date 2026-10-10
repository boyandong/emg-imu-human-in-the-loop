"""Joint lifecycle on the actual frozen Song source; complete bouts are fixtures.

These tests do not treat Song stable-only excerpts as full actions and do not
estimate native accuracy. Probability oracles use a separately written sum.
"""
import copy
from dataclasses import replace
import json
from pathlib import Path
import pickle
import zipfile

import numpy as np
import pytest

from emgimu.feature_bank.extended_window_cli_v1 import load_extended_workflow
from emgimu.feature_bank.joint_bout_workflow_v1 import JointBoutWorkflowV1, profile_digest
from emgimu.feature_bank.personal_temporal_bouts_v1 import TemporalBoutBatchV1

ROOT=Path(__file__).resolve().parents[1]
SONG=ROOT/'benchmarks/song_real8'


def native_fixture(w,prefix,shots=2):
    rng=np.random.default_rng(120+sum(prefix.encode()))
    sequences=[];ids=[];labels={}
    for i,c in enumerate(w.temporal.class_names):
        for j in range(shots):
            n=250+13*j
            envelope=(1+.4*np.sin(np.linspace(0,2*np.pi,n)))[:,None]
            amplitude=.1 if c==w.window.rest_label else 1+i
            x=rng.normal(size=(n,8))*envelope*amplitude*np.roll(np.arange(1,9),i)
            t=f'{prefix}:{c}:{j}';ids.append(t);labels[t]=c;sequences.append(x.astype(np.float32))
    b=TemporalBoutBatchV1(tuple(sequences),tuple(ids),(prefix,)*len(ids),(0,)*len(ids),250.,
        w.window.channels,w.window.preprocessing_id,'complete_cued')
    raw=replace(b,sequences=tuple((x*1000).astype(np.float32) for x in sequences),
        preprocessing_id='raw_pre_software_highpass')
    return b,labels,raw


@pytest.fixture(scope='module')
def registered():
    source=load_extended_workflow(SONG/'song_extended_window_v1/source_bank.pkl',
        SONG/'song_extended_window_v1/policy.json',SONG/'SONG_EXTENDED_WINDOW_V1_RESULTS.json',
        SONG/'song_raw_quality_v1/source_gate.pkl',SONG/'SONG_RAW_QUALITY_V1_RESULTS.json')
    w=JointBoutWorkflowV1(source)
    b,y,raw=native_fixture(w,'long');p=w.enroll(b,y,user_id='fixture',session_id='long',raw_batch=raw)
    b,y,raw=native_fixture(w,'current',shots=1)
    s=w.enroll(b,y,user_id='fixture',session_id='current',personal=p,raw_batch=raw)
    q,_,qr=native_fixture(w,'query',shots=1)
    return w,p,s,q,qr


def test_shared_registration_native_samples_cost_and_neutral_only(registered):
    w,p,s,_,_=registered
    for profile in (p,s):
        b=profile.calibration;cost=profile.calibration_cost
        assert set(profile.window.base.calibration_trials)==set(b.trial_ids)==set(profile.temporal.trial_ids)
        assert profile.window.base.user_id==profile.temporal.user_id==profile.user_id
        assert profile.window.base.session_id==profile.temporal.session_id==profile.session_id
        assert cost['unique_native_calibration_trials']==len(b.trial_ids)
        assert cost['native_signal_samples']==sum(map(len,b.sequences))
        assert cost['native_signal_seconds']==sum(map(len,b.sequences))/250
        assert cost['window_rows']==sum(1+(len(x)-50)//10 for x in b.sequences)
        assert cost['window_unrepresented_tail_samples']==[len(x)-(50+10*((len(x)-50)//10)) for x in b.sequences]
        neutral=tuple(t for t,c in profile.calibration_labels if c==w.window.rest_label)
        assert tuple(cost['detector_neutral_trials'])==neutral==profile.detector.source_trial_ids_
        assert cost['counted_once'] and cost['wall_clock_seconds'] is None
        assert not cost['physical_validation_proven']
    assert s.parent_profile_id==p.profile_id
    # Independently compute the detector's neutral-only quantiles and MAD.
    labels=dict(p.calibration_labels)
    rest=np.concatenate([x for t,x in zip(p.calibration.trial_ids,p.calibration.sequences)
                         if labels[t]==w.window.rest_label]).astype(float)
    power=np.mean(rest*rest,axis=1);width=round(250*.025)
    energy=np.sqrt(np.array([power[i:i+width].mean() for i in range(len(power)-width+1)]))
    median=np.median(energy);mad=np.median(abs(energy-median));scale=max(1.4826*mad,.05*median,1e-10)
    off=max(np.quantile(energy,.95),median+3*scale)
    on=max(np.quantile(energy,.995),median+6*scale,1.2*off)
    np.testing.assert_allclose([p.detector.off_,p.detector.on_],[off,on],rtol=0,atol=1e-14)


@pytest.mark.parametrize('has_session',[False,True])
@pytest.mark.parametrize('anchor,routing',[(False,False),(False,True),(True,True)])
@pytest.mark.parametrize('quality',['off','structural','soft'])
def test_joint_branch_matrix_independent_mixture_and_immutable_state(registered,has_session,anchor,routing,quality):
    w,p,s,q,raw=registered;s=s if has_session else None
    before=pickle.dumps((w,p,s),protocol=4)
    r=w.predict(q,personal=p,session=s,user_id='fixture',session_id='current',raw_batch=raw,
        quality_mode=quality,use_anchor=anchor,use_session_routing=routing)
    arms=r['temporal']['arms']
    expected=np.array([[.75*arms['base'][i,j]+.125*arms['DTW_blended'][i,j]
        +.125*arms['signature_blended'][i,j] for j in range(4)] for i in range(len(q.trial_ids))])
    np.testing.assert_allclose(r['probabilities'],expected,rtol=0,atol=2e-15)
    np.testing.assert_allclose(r['probabilities'],arms['base_full'],rtol=0,atol=2e-15)
    assert tuple(r['predicted_labels'])==r['labels']
    assert r['composition']['evaluation_trials']==q.trial_ids
    assert r['joint_calibration_trials']==len(p.calibration.trial_ids)+(0 if s is None else len(s.calibration.trial_ids))
    assert r['temporal_uses_all_native_samples'] and r['decision_available_after_interval']
    assert not r['physical_latency_proven'] and not r['default_promoted']
    assert pickle.dumps((w,p,s),protocol=4)==before


def test_persistence_two_profiles_and_caller_array_ownership(registered,tmp_path):
    w,p,s,q,raw=registered
    pp,sp=tmp_path/'personal.zip',tmp_path/'session.zip'
    w.save_profile(p,pp);w.save_profile(s,sp)
    lp=w.load_profile(pp,user_id='fixture')
    ls=w.load_profile(sp,user_id='fixture',session_id='current',personal=lp)
    assert (lp.profile_id,ls.profile_id)==(p.profile_id,s.profile_id)
    for a,b in zip(p.raw_calibration.sequences,lp.raw_calibration.sequences):np.testing.assert_array_equal(a,b)
    a=w.predict(q,personal=p,session=s,user_id='fixture',session_id='current',raw_batch=raw)
    b=w.predict(q,personal=lp,session=ls,user_id='fixture',session_id='current',raw_batch=raw)
    np.testing.assert_array_equal(a['probabilities'],b['probabilities'])
    assert profile_digest(p)==profile_digest(copy.deepcopy(p))==profile_digest(pickle.loads(pickle.dumps(p)))
    original=pp.read_bytes()
    with pytest.raises(FileExistsError):w.save_profile(p,pp)
    assert pp.read_bytes()==original
    b,y,r=native_fixture(w,'owned',shots=1)
    owned=w.enroll(b,y,user_id='fixture',session_id='owned',raw_batch=r)
    old=owned.calibration.sequences[0].copy();old_raw=owned.raw_calibration.sequences[0].copy()
    b.sequences[0][:]=123.;r.sequences[0][:]=456.;y.clear()
    np.testing.assert_array_equal(old,owned.calibration.sequences[0])
    np.testing.assert_array_equal(old_raw,owned.raw_calibration.sequences[0])
    w._profile(owned,'fixture')


@pytest.mark.parametrize('kind',['trial','record','source','forbidden_trial','forbidden_record','estimated','raw_axes','missing_raw'])
def test_registration_and_query_provenance_rejections(registered,kind):
    w,p,s,q,raw=registered
    before=pickle.dumps((w,p,s),protocol=4)
    if kind in ('trial','record'):
        bad=replace(q,trial_ids=p.calibration.trial_ids[:4]) if kind=='trial' else replace(q,recording_ids=p.calibration.recording_ids[:4])
        with pytest.raises(ValueError,match='overlaps'):w.predict(bad,personal=p,user_id='fixture',session_id='query')
    else:
        b,y,r=native_fixture(w,'invalid',shots=1);kwargs={}
        if kind=='source':
            ids=(w.window.bank.policy_.source_trials[0],)+b.trial_ids[1:]
            y={t:c for t,c in zip(ids,y.values())};b=replace(b,trial_ids=ids)
        elif kind=='forbidden_trial':kwargs['forbidden_trial_ids']=(b.trial_ids[0],)
        elif kind=='forbidden_record':kwargs['forbidden_recording_ids']=(b.recording_ids[0],)
        elif kind=='estimated':b=replace(b,boundary_kind='estimated')
        elif kind=='raw_axes':kwargs['raw_batch']=replace(r,starts=(1,)*4)
        elif kind=='missing_raw':kwargs['quality_mode']='structural'
        with pytest.raises(ValueError):w.enroll(b,y,user_id='fixture',session_id='invalid',**kwargs)
    assert pickle.dumps((w,p,s),protocol=4)==before


def test_parent_rejection_and_failure_isolation(registered,monkeypatch):
    w,p,s,_,_=registered
    b,y,raw=native_fixture(w,'new',shots=1)
    with pytest.raises(ValueError,match='disjoint'):
        w.enroll(replace(b,recording_ids=p.calibration.recording_ids[:4]),y,
            user_id='fixture',session_id='new',personal=p)
    before=pickle.dumps((w,p,s),protocol=4)
    def fail(*args,**kwargs):raise RuntimeError('injected window registration failure')
    with monkeypatch.context() as patch:
        patch.setattr(type(w.window),'calibrate_session',fail)
        with pytest.raises(RuntimeError,match='injected'):
            w.enroll(b,y,user_id='fixture',session_id='new',personal=p,raw_batch=raw)
    assert pickle.dumps((w,p,s),protocol=4)==before


def test_severe_raw_rejection_cannot_be_rescued_by_temporal_branch(registered):
    w,p,s,q,raw=registered
    damaged=tuple(x.copy() for x in raw.sequences)
    for x in damaged:x[:,0]=0
    fault=replace(raw,sequences=damaged)
    r=w.predict(q,personal=p,session=s,user_id='fixture',session_id='current',raw_batch=fault,quality_mode='structural')
    assert r['labels']==('Unknown',)*4 and r['rejected'].all()
    assert r['composition']['rejection_reason']==('all_quality_rejected',)*4
    np.testing.assert_allclose(r['probabilities'],r['temporal']['arms']['base_full'],rtol=0,atol=2e-15)
    with pytest.raises(ValueError,match='quality rejected'):
        w.enroll(q,{t:c for t,c in zip(q.trial_ids,w.temporal.class_names)},
            user_id='fixture',session_id='fault',raw_batch=fault,quality_mode='structural')


def test_estimated_query_and_trial_order(registered):
    w,p,s,q,raw=registered
    r=w.predict(q,personal=p,session=s,user_id='fixture',session_id='current')
    order=[3,1,0,2]
    shuffled=replace(q,**{k:tuple(getattr(q,k)[i] for i in order) for k in ('sequences','trial_ids','recording_ids','starts')})
    sr=w.predict(shuffled,personal=p,session=s,user_id='fixture',session_id='current')
    np.testing.assert_allclose(sr['probabilities'],r['probabilities'][order],rtol=0,atol=1e-15)
    er=w.predict(replace(q,boundary_kind='estimated'),personal=p,user_id='fixture',session_id='query')
    assert er['temporal']['boundary_kind']=='estimated' and not er['temporal']['certified_full_coverage']


@pytest.mark.parametrize('problem',['wrong_user','wrong_session','missing_parent','wrong_parent','payload','manifest','members','tampered'])
def test_saved_identity_and_corruption_rejected(registered,tmp_path,problem):
    w,p,s,_,_=registered;path=tmp_path/'profile.zip'
    w.save_profile(s if problem in ('wrong_session','missing_parent','wrong_parent') else p,path)
    kwargs=dict(user_id='fixture')
    if problem=='wrong_user':kwargs['user_id']='other'
    elif problem=='wrong_session':kwargs.update(personal=p,session_id='other')
    elif problem=='wrong_parent':kwargs.update(personal=replace(p,profile_id='bad'),session_id='current')
    elif problem in ('payload','manifest','members'):
        with zipfile.ZipFile(path) as z:content={n:z.read(n) for n in z.namelist()}
        if problem=='payload':content['profile.pkl']+=b'corrupt'
        elif problem=='manifest':
            m=json.loads(content['manifest.json']);m['profile_id']='bad';content['manifest.json']=json.dumps(m).encode()
        else:content['extra']=b'extra'
        with zipfile.ZipFile(path,'w') as z:
            for name,data in content.items():z.writestr(name,data)
    elif problem=='tampered':
        damaged=copy.deepcopy(p);damaged.calibration.sequences[0][0,0]+=1
        with pytest.raises(ValueError,match='checksum'):w.save_profile(damaged,tmp_path/'tampered.zip')
        assert not (tmp_path/'tampered.zip').exists();return
    with pytest.raises(ValueError):w.load_profile(path,**kwargs)


def test_save_failure_cleans_partial_file_without_touching_existing(registered,tmp_path,monkeypatch):
    w,p,_,_,_=registered
    path=tmp_path/'failed.zip'
    def fail(*args,**kwargs):raise OSError('injected archive failure')
    with monkeypatch.context() as patch:
        patch.setattr(zipfile.ZipFile,'writestr',fail)
        with pytest.raises(OSError,match='injected'):w.save_profile(p,path)
    assert not path.exists()
    path.write_bytes(b'existing')
    with pytest.raises(FileExistsError):w.save_profile(p,path)
    assert path.read_bytes()==b'existing'


@pytest.mark.parametrize('mode',['long_term','local','blended'])
def test_explicit_anchor_modes_survive_joint_lifecycle(registered,mode):
    w,p,s,q,_=registered
    r=w.predict(q,personal=p,session=s,user_id='fixture',session_id='current',anchor_mode=mode)
    assert np.isfinite(r['probabilities']).all()
    np.testing.assert_allclose(r['probabilities'].sum(1),1.,rtol=0,atol=2e-15)


def test_detector_stream_states_are_separate_and_profile_content_hash_is_layout_independent(registered):
    w,p,s,q,_=registered
    before=pickle.dumps((w,p,s),protocol=4)
    a=w.make_detector(s,user_id='fixture',personal=p,session_id='current')
    b=w.make_detector(s,user_id='fixture',personal=p,session_id='current')
    a.feed(q.sequences[0][:100],0,trial_id=q.trial_ids[0])
    assert a.next_==100 and b.next_ is None and s.detector.next_ is None
    assert pickle.dumps((w,p,s),protocol=4)==before
    equivalent=replace(p,calibration=replace(p.calibration,
        sequences=tuple(np.asfortranarray(x) for x in p.calibration.sequences)),
        calibration_cost=dict(reversed(list(p.calibration_cost.items()))))
    assert profile_digest(equivalent)==p.profile_id
    w._profile(equivalent,'fixture')
