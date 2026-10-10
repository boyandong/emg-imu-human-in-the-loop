"""Native-rate joint lifecycle and frozen250Hz parity; no accuracy claims."""
from dataclasses import replace
from pathlib import Path
import pickle
import numpy as np
import pytest
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.document_window_composition_v1 import GROUPS
from emgimu.feature_bank.native_document_window_v2 import fit_native_document_source,NativeDocumentWindowDecisionV2
from emgimu.feature_bank.native_joint_bout_workflow_v2 import NativeJointBoutWorkflowV2
from emgimu.feature_bank.frozen_emg_provider_bank_v1 import FrozenEmgProviderBankV1
from emgimu.feature_bank.personal_session_workflow_v1 import PersonalSessionWorkflowV1
from emgimu.feature_bank.personal_temporal_bouts_v1 import TemporalBoutBatchV1
from emgimu.feature_bank.extended_window_cli_v1 import load_extended_workflow
from emgimu.feature_bank.joint_bout_workflow_v1 import JointBoutWorkflowV1
from test_joint_bout_workflow_v1 import native_fixture

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module',params=[200.,250.])
def workflow(request):
    rate=request.param;size=round(.2*rate)
    classes=('close','open','relax');ids=np.array([f'source:{c}:{i}' for c in classes for i in range(8)])
    y=np.array([c for c in classes for i in range(8)])
    values=np.random.default_rng(782).normal(size=(len(ids),size,8)).astype(np.float32)
    values[y=='relax']*=.1
    b=FeatureBatch(values,rate)
    families,models,metadata=fit_native_document_source(b,y,ids,rest_label='relax')
    assert [metadata[g]['dimension'] for g in GROUPS]==[48,8,72,8,72,57,12]
    bank=FrozenEmgProviderBankV1(families,models,{g:1. for g in GROUPS},classes=classes,class_names=classes,
        source_trial_ids=tuple(ids),source_policy_id='native_fixture',sample_rate_hz=rate,window_samples=size,channels=8,
        population=np.ones(7)/7,n0=4.,reliability_temperature=.5)
    base=PersonalSessionWorkflowV1(bank,channel_ids=tuple(f'EMG{i}' for i in range(8)),
        preprocessing_id='native_fixture',rest_label='relax',quality_options={})
    return NativeJointBoutWorkflowV2(NativeDocumentWindowDecisionV2(base))


def bouts(w,prefix):
    rng=np.random.default_rng(420+sum(prefix.encode()));sequences=[];ids=[];labels={}
    for i,c in enumerate(w.temporal.class_names):
        for j in range(2):
            n=int(w.rate)+17+11*j
            x=rng.normal(size=(n,8))*(.1 if c=='relax' else 2+i)
            ids.append(f'{prefix}:{c}:{j}');labels[ids[-1]]=c;sequences.append(x.astype(np.float32))
    batch=TemporalBoutBatchV1(tuple(sequences),tuple(ids),(prefix,)*len(ids),tuple(9*i for i in range(len(ids))),
        w.rate,w.window.channels,w.window.preprocessing_id,'complete_cued')
    return batch,labels


def test_native_rate_windows_seconds_class_axis_and_shared_profiles(workflow,tmp_path):
    w=workflow;b,y=bouts(w,'long')
    p=w.enroll(b,y,user_id='fixture',session_id='long')
    cb,cy=bouts(w,'current');s=w.enroll(cb,cy,user_id='fixture',session_id='current',personal=p)
    query,_=bouts(w,'query');before=pickle.dumps((w,p,s))
    r=w.predict(query,personal=p,session=s,user_id='fixture',session_id='current')
    cost=p.calibration_cost
    assert cost['native_sample_rate_hz']==w.rate and cost['native_signal_seconds']==sum(map(len,b.sequences))/w.rate
    assert cost['unique_native_calibration_trials']==6 and cost['counted_once'] and not cost['resampled']
    assert cost['window_rows']==sum(1+(len(x)-w.samples)//w.hop for x in b.sequences)
    assert cost['window_unrepresented_tail_samples']==[(len(x)-w.samples)%w.hop for x in b.sequences]
    assert p.detector.rate_==w.rate
    assert r['probabilities'].shape==(6,3) and r['temporal']['class_names']==('close','open','relax')
    np.testing.assert_allclose(r['probabilities'].sum(1),1.,rtol=0,atol=1e-14)
    pp,sp=tmp_path/'p.zip',tmp_path/'s.zip';w.save_profile(p,pp);w.save_profile(s,sp)
    rp=w.load_profile(pp,user_id='fixture');rs=w.load_profile(sp,user_id='fixture',session_id='current',personal=rp)
    replay=w.predict(query,personal=rp,session=rs,user_id='fixture',session_id='current')
    np.testing.assert_array_equal(r['probabilities'],replay['probabilities'])
    assert pickle.dumps((w,p,s))==before


def test_native_adapter_mixture_matches_direct_window_and_path_arithmetic(workflow):
    w=workflow;b,y=bouts(w,'anchor');p=w.enroll(b,y,user_id='fixture',session_id='long')
    q,_=bouts(w,'query');windows,ids,offsets,tails=w._windows(q)
    before=pickle.dumps((w,p))
    actual=w.predict(q,personal=p,user_id='fixture',session_id='query')
    direct=w.window.predict(windows,ids,window_offsets=offsets,personal=p.window,user_id='fixture',session_id='query',
        observed_channel_ids=q.channel_ids,preprocessing_id=q.preprocessing_id)
    order=[direct['trial_ids'].index(t) for t in q.trial_ids]
    temporal=w.temporal.predict(q,personal=p.temporal,user_id='fixture',session_id='query',
        base_probabilities=direct['probabilities'][order],base_trial_ids=q.trial_ids,base_class_names=w.temporal.class_names)
    expected=.75*direct['probabilities'][order]+.125*temporal['arms']['DTW_blended']+.125*temporal['arms']['signature_blended']
    expected/=expected.sum(1,keepdims=True)
    np.testing.assert_allclose(actual['probabilities'],expected,rtol=0,atol=1e-14)
    assert actual['window_unrepresented_tail_samples']==tails and pickle.dumps((w,p))==before


def test_native_query_and_metadata_contracts_reject_implicit_substitutions(workflow):
    w=workflow;b,y=bouts(w,'cal');p=w.enroll(b,y,user_id='fixture',session_id='long')
    q,_=bouts(w,'query')
    with pytest.raises(ValueError,match='overlaps'):w.predict(b,personal=p,user_id='fixture',session_id='query')
    with pytest.raises(ValueError,match='quality'):w.predict(q,personal=p,user_id='fixture',session_id='query',quality_mode='structural')
    with pytest.raises(ValueError):w.predict(replace(q,sample_rate_hz=250. if w.rate==200. else 200.),
        personal=p,user_id='fixture',session_id='query')
    with pytest.raises(ValueError,match='complete cued'):w.enroll(replace(q,boundary_kind='estimated'),y,user_id='fixture',session_id='new')


def test_new_native250_lifecycle_preserves_frozen_joint_probabilities(tmp_path):
    h=ROOT/'benchmarks/song_real8'
    source=load_extended_workflow(h/'song_extended_window_v1/source_bank.pkl',h/'song_extended_window_v1/policy.json',
        h/'SONG_EXTENDED_WINDOW_V1_RESULTS.json',h/'song_raw_quality_v1/source_gate.pkl',h/'SONG_RAW_QUALITY_V1_RESULTS.json')
    old=JointBoutWorkflowV1(source);new=NativeJointBoutWorkflowV2(source)
    b,y,raw=native_fixture(old,'parity_long',shots=1)
    op=old.enroll(b,y,user_id='fixture',session_id='long',raw_batch=raw)
    np_=new.enroll(b,y,user_id='fixture',session_id='long',raw_batch=raw)
    query,_,raw=native_fixture(old,'parity_query',shots=1)
    for quality in ('off','structural','soft'):
        kwargs=dict(user_id='fixture',session_id='query',raw_batch=raw,quality_mode=quality)
        a=old.predict(query,personal=op,**kwargs);b=new.predict(query,personal=np_,**kwargs)
        np.testing.assert_array_equal(a['probabilities'],b['probabilities'])
        np.testing.assert_array_equal(a['rejected'],b['rejected'])
    path=tmp_path/'v1.zip';old.save_profile(op,path)
    with pytest.raises(ValueError,match='source'):new.load_profile(path,user_id='fixture')


def test_native_joint_window_family_removal_preserves_templates_and_frozen_state(workflow):
    w=workflow;b,y=bouts(w,'lofo_long');p=w.enroll(b,y,user_id='fixture',session_id='long')
    query,_=bouts(w,'lofo_query');before=pickle.dumps((w,p))
    full=w.predict(query,personal=p,user_id='fixture',session_id='query')
    active=tuple(g for g in GROUPS if g not in ('F2ac','F2b'))
    removed=w.predict(query,personal=p,user_id='fixture',session_id='query',available_window_providers=active)
    a=removed['window'];order=[a['trial_ids'].index(t) for t in query.trial_ids]
    np.testing.assert_allclose(a['probabilities'],sum(a['weights'][i]*a['decision_provider_probabilities'][g]
        for i,g in enumerate(active)),rtol=0,atol=1e-14)
    for key in ('DTW_blended','signature_blended'):
        np.testing.assert_array_equal(full['temporal']['arms'][key],removed['temporal']['arms'][key])
    expected=.75*a['probabilities'][order]+.125*full['temporal']['arms']['DTW_blended']+.125*full['temporal']['arms']['signature_blended']
    np.testing.assert_allclose(removed['probabilities'],expected,rtol=0,atol=1e-14)
    assert set(a['decision_provider_probabilities'])==set(active) and pickle.dumps((w,p))==before
    with pytest.raises(ValueError,match='available providers'):
        w.predict(query,personal=p,user_id='fixture',session_id='query',available_window_providers=())


def test_independent_native_study_oracles_match_frozen_classifier_and_full_paths(workflow):
    from benchmarks.new_bank_v3.verify_roam_native_joint_v1 import manual_source,temporal_oracle
    w=workflow;b,y=bouts(w,'oracle_long');p=w.enroll(b,y,user_id='fixture',session_id='long')
    query,_=bouts(w,'oracle_query');r=w.predict(query,personal=p,user_id='fixture',session_id='query')
    qb,qi,qo,_=w._windows(query)
    axis,q=manual_source(w.window.bank.families_,w.window.bank.models_,qb,qi,w.window.bank.temperatures_)
    direct=w.window.bank.predict_providers(qb,qi,window_offsets=qo,user_id='fixture')
    assert axis==direct['trial_ids']
    for g in GROUPS:np.testing.assert_allclose(q[g],direct['probabilities'][g],rtol=0,atol=1e-12)
    dtw,signature=temporal_oracle(query,p.temporal)
    np.testing.assert_allclose(dtw,r['temporal']['arms']['DTW_long'],rtol=0,atol=1e-12)
    np.testing.assert_allclose(signature,r['temporal']['arms']['signature_long'],rtol=0,atol=1e-12)
