"""Independent F7 geometry, F8 weights, branch omissions and persistence checks."""
from dataclasses import replace
import inspect
import pickle
from pathlib import Path
import numpy as np
import pytest
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.affine_spd_anchor import affine_spd_distance
from emgimu.feature_bank.personal_session_cli_v1 import load_workflow
from emgimu.feature_bank.personal_session_decision_v1 import PersonalSessionDecisionV1,anchor_probability,spd_distances
from emgimu.feature_bank.personal_session_stream_v2 import load_gate

ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'benchmarks/song_real8'
CHANNELS=tuple(f'CH{i+1}' for i in range(8))


def kwargs(session='new'):
    return dict(user_id='fixture',session_id=session,observed_channel_ids=CHANNELS,
                preprocessing_id='song250_causal_hp40_order4_notch50_100_Q30_zero_session_initial_v1')


def windows(prefix,shots=2):
    classes=('fist','index_pinch','neutral','open_hand');rng=np.random.default_rng(17+len(prefix))
    ids=np.array([f'{prefix}:{c}:{i}' for c in classes for i in range(shots) for _ in range(2)])
    y=np.repeat(classes,shots*2);x=rng.normal(size=(len(ids),50,8))
    for i,c in enumerate(classes):x[y==c]*=(1+np.arange(8)*(.3+i*.1))*(i+1)
    return FeatureBatch(x.astype(np.float32),250.),ids,np.tile([0,1],len(ids)//2),dict(zip(ids,y))


@pytest.fixture
def workflow():
    base=load_workflow(HERE/'song_personal_session_v1/source_bank.pkl',ROOT/'feature_bank/SONG_PERSONAL_SESSION_ACCEPTANCE_V1.json')
    gate=load_gate(HERE/'song_raw_quality_v1/source_gate.pkl',HERE/'SONG_RAW_QUALITY_V1_RESULTS.json')
    return PersonalSessionDecisionV1(base,gate=gate)


def profiles(w):
    b,ids,o,y=windows('long',3);p=w.enroll_user(b,ids,y,window_offsets=o,**kwargs('long'))
    b,ids,o,y=windows('current',1);s=w.calibrate_session(b,ids,y,window_offsets=o,personal=p,**kwargs())
    return p,s


def test_anchor_probabilities_use_calibration_geometry_and_exact_affine_metric():
    x=np.array([[0.,0.],[1.,1.]]);p=np.array([[0.,0.],[2.,0.],[0.,2.]])
    q,d,t=anchor_probability(x,p)
    expected=np.stack([np.sqrt(((x-v)**2).sum(1)) for v in p],1)
    assert t==(2+2+np.sqrt(8))/3
    np.testing.assert_array_equal(d,expected)
    logits=-expected/t;logits-=logits.max(1,keepdims=True);e=np.exp(logits);e/=e.sum(1,keepdims=True)
    np.testing.assert_array_equal(q,e)
    q,_,t=anchor_probability(x,np.zeros((3,2)));assert t==0
    np.testing.assert_array_equal(q,np.full((2,3),1/3))
    rng=np.random.default_rng(13);a=rng.normal(size=(7,8,8));a=a@a.transpose(0,2,1)+np.eye(8)
    actual=spd_distances(a[:4],a[4:]);oracle=np.array([[affine_spd_distance(v,p) for p in a[4:]] for v in a[:4]])
    np.testing.assert_allclose(actual,oracle,rtol=0,atol=1e-12)


def test_integrated_probabilities_and_removals_follow_explicit_arithmetic(workflow):
    w=workflow;p,s=profiles(w);b,ids,o,_=windows('held');before=pickle.dumps((w,p,s))
    out=w.predict(b,ids,window_offsets=o,personal=p,session=s,**kwargs())
    base=w.base.predict(b,ids,window_offsets=o,personal=p.base,session=s.base,**kwargs())
    source=w.bank.predict_providers(b,ids,window_offsets=o,user_id='fixture')['probabilities']
    rawweights=np.array(base['weights'])*np.maximum(np.exp(-np.clip(s.routing['risks'],0,5)),.05)
    rawweights/=rawweights.sum()
    np.testing.assert_allclose(out['weights'],rawweights,rtol=0,atol=1e-12)
    expected=sum(rawweights[i]*(.5*source[n]+.5*out['anchor_probabilities'][n]) for i,n in enumerate(w.bank.providers_))
    np.testing.assert_allclose(out['probabilities'],expected,rtol=0,atol=1e-12)
    beta=.75
    np.testing.assert_allclose(s.spd_blended,beta*p.spd_prototypes+(1-beta)*s.spd_local,rtol=0,atol=1e-12)
    off=w.predict(b,ids,window_offsets=o,personal=p,session=s,use_anchor=False,use_session_routing=False,**kwargs())
    np.testing.assert_allclose(off['probabilities'],base['probabilities'],rtol=0,atol=1e-12)
    available=('F0','F1');removed=w.predict(b,ids,window_offsets=o,personal=p,session=s,available=available,**kwargs())
    assert tuple(removed['decision_provider_probabilities'])==available
    expected=sum(float(removed['weights'][i])*removed['decision_provider_probabilities'][n] for i,n in enumerate(available))
    np.testing.assert_allclose(removed['probabilities'],expected,rtol=0,atol=1e-12)
    long=w.predict(b,ids,window_offsets=o,personal=p,**kwargs())
    assert long['anchor_mode']=='long_term' and not long['session_routing_enabled']
    assert 'labels' not in inspect.signature(w.predict).parameters
    assert pickle.dumps((w,p,s))==before


def test_profile_persistence_leakage_identity_and_raw_quality(workflow,tmp_path):
    w=workflow;p,s=profiles(w);b,ids,o,_=windows('held')
    pp=tmp_path/'personal.zip';sp=tmp_path/'session.zip';w.save_profile(p,pp);w.save_profile(s,sp)
    restored=w.load_profile(pp,user_id='fixture');session=w.load_profile(sp,user_id='fixture',session_id='new',personal=restored)
    a=w.predict(b,ids,window_offsets=o,personal=p,session=s,**kwargs())
    z=w.predict(b,ids,window_offsets=o,personal=restored,session=session,**kwargs())
    np.testing.assert_array_equal(a['probabilities'],z['probabilities'])
    for call in [lambda:w.load_profile(pp,user_id='other'),lambda:w.load_profile(sp,user_id='fixture',session_id='other',personal=p),
                 lambda:w.predict(b,ids,window_offsets=o,personal=replace(p,policy_id='other'),**kwargs()),
                 lambda:w.predict(b,ids,window_offsets=o,personal=p,session=s,quality_mode='structural',**kwargs())]:
        with pytest.raises(ValueError):call()
    raw=b.emg.copy();raw[:,:,2]=0
    result=w.predict(b,ids,window_offsets=o,personal=p,session=s,quality_mode='structural',raw_batch=FeatureBatch(raw,250.),**kwargs())
    assert set(result['predicted_labels'])=={'Unknown'} and result['quality_decision']['rejected'].all()
    leaked=ids.copy();leaked[:2]=p.base.calibration_trials[0]
    with pytest.raises(ValueError):w.predict(b,leaked,window_offsets=o,personal=p,session=s,**kwargs())
    with pytest.raises(FileExistsError):w.save_profile(p,pp)
