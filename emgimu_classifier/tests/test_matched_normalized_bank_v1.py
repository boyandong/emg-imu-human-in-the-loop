"""Independent normalization oracle, source/domain guards and lifecycle replay."""
import pickle
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.new_bank_v2 import RestNoiseDetailV2
from emgimu.feature_bank.calibration import DocumentPersonalNormalizerV2
from emgimu.feature_bank.matched_normalized_bank_v1 import MatchedNormalizedProviderBankV1,MatchedNormalizedWorkflowV1
from emgimu.feature_bank.frozen_emg_provider_bank_v1 import FrozenEmgProviderBankV1

CLASSES=('fist','index_pinch','neutral','open_hand')
CHANNELS=tuple(f'CH{i+1}' for i in range(8))
CONFIG=dict(channel_ids=CHANNELS,preprocessing_id='fixture_causal',rest_label='neutral')


def data(prefix,seed):
    labels=np.repeat(CLASSES,4)
    rng=np.random.default_rng(seed)
    x=rng.normal(size=(16,50,8))*np.repeat([80,40,2,120],4)[:,None,None]+np.arange(8)[None,None,:]
    ids=np.array([prefix+str(i) for i in range(16)])
    return FeatureBatch(x,250.),ids,dict(zip(ids,labels)),labels


def workflow():
    b,ids,labels,y=data('source',12);cal=np.array([0,4,8,12]);fit=np.setdiff1d(np.arange(16),cal)
    normalizer=DocumentPersonalNormalizerV2(rest_label='neutral').fit(b.take(cal),y[cal])
    normalized=normalizer.transform(b.take(fit));family=RestNoiseDetailV2(rest_label='neutral').fit(normalized,y[fit])
    x=family.transform(normalized);scaler=StandardScaler().fit(x)
    model=LogisticRegression(max_iter=1000).fit(scaler.transform(x),y[fit])
    kwargs=dict(classes=CLASSES,class_names=CLASSES,source_trial_ids=ids.tolist(),source_policy_id='fixture_frozen',
        sample_rate_hz=250.,window_samples=50,channels=8,population=(1.,),n0=4.,reliability_temperature=.5)
    bank=MatchedNormalizedProviderBankV1({'F0':[family]},{'F0':(scaler,model)},{'F0':1.3},
        source_model_fit_trials=ids[fit].tolist(),source_normalization_trials={'source_recording':ids[cal].tolist()},**kwargs)
    return MatchedNormalizedWorkflowV1(bank,**CONFIG)


def args(ids,session):
    return dict(window_offsets=np.zeros(len(ids),int),user_id='fixture',session_id=session,
                observed_channel_ids=CHANNELS,preprocessing_id='fixture_causal')


def test_classifier_uses_exact_raw_calibration_normalizer_once_and_never_refits():
    w=workflow();b,ids,labels,y=data('personal',3)
    profile=w.enroll_user(b,ids,labels,**args(ids,'long'))
    rest=b.emg[y=='neutral'].reshape(-1,8);center=np.median(rest,axis=0)
    scale=np.quantile(np.abs(b.emg[y!='neutral']-center).reshape(-1,8),.95,axis=0)
    np.testing.assert_array_equal(profile.normalizer.center_,center)
    np.testing.assert_array_equal(profile.normalizer.scale_,scale)
    held,hi,_,_=data('held',9);expected=(held.emg-center)/(scale+1e-10)
    np.testing.assert_array_equal(w.classifier_input(held,personal=profile,**args(hi,'new')).emg,expected)
    source=pickle.dumps(w.bank);personal=pickle.dumps(profile)
    r=w.predict(held,hi,personal=profile,**args(hi,'new'))
    oracle=w.bank.predict(FeatureBatch(expected,250.),hi,window_offsets=np.zeros(len(hi),int),user_id='fixture',
                          user_state=profile.fusion_state)
    np.testing.assert_allclose(r['probabilities'],oracle['probabilities'],rtol=0,atol=1e-12)
    assert r['normalization_used_by_classifier'] and r['quality_observations'] is None and r['session_descriptor'] is None
    assert pickle.dumps(w.bank)==source and pickle.dumps(profile)==personal
    with pytest.raises(ValueError):w.predict(held,hi,personal=None,**args(hi,'new'))
    with pytest.raises(ValueError):w.predict(b,ids,personal=profile,**args(ids,'new'))


def test_new_session_updates_scale_without_overwriting_long_term_and_roundtrips(tmp_path):
    w=workflow();b,ids,labels,_=data('personal',3);p=w.enroll_user(b,ids,labels,**args(ids,'long'));before=pickle.dumps(p)
    current,ci,cy,_=data('current',4);s=w.calibrate_session(current,ci,cy,personal=p,**args(ci,'new'))
    assert pickle.dumps(p)==before and not np.array_equal(p.normalizer.scale_,s.normalizer.scale_)
    held,hi,_,_=data('held',5);original=w.predict(held,hi,personal=p,session=s,**args(hi,'new'))
    pp=tmp_path/'personal.zip';sp=tmp_path/'session.zip';w.save_profile(p,pp);w.save_profile(s,sp)
    restored_p=w.load_profile(pp,user_id='fixture');restored_s=w.load_profile(sp,user_id='fixture',session_id='new',personal=restored_p)
    replay=w.predict(held,hi,personal=restored_p,session=restored_s,**args(hi,'new'))
    np.testing.assert_array_equal(original['probabilities'],replay['probabilities'])
    reversed_args=args(hi,'new');reversed_args['observed_channel_ids']=CHANNELS[::-1]
    reorder=w.predict(FeatureBatch(held.emg[:,:,::-1],250.),hi,personal=p,session=s,**reversed_args)
    np.testing.assert_array_equal(original['probabilities'],reorder['probabilities'])
    for bad in (dict(user_id='other'),dict(session_id='other'),dict(preprocessing_id='wrong')):
        with pytest.raises(ValueError):w.predict(held,hi,personal=p,session=s,**dict(args(hi,'new'),**bad))


def test_raw_source_models_and_normalization_trial_leakage_are_rejected():
    w=workflow();bank=w.bank
    raw=FrozenEmgProviderBankV1(bank.families_,bank.models_,bank.temperatures_,classes=CLASSES,class_names=CLASSES,
        source_trial_ids=bank.policy_.source_trials,source_policy_id='raw_fixture',sample_rate_hz=250.,window_samples=50,channels=8,
        population=(1.,),n0=4.,reliability_temperature=.5)
    with pytest.raises(ValueError,match='explicitly trained'):MatchedNormalizedWorkflowV1(raw,**CONFIG)
    b,ids,labels,_=data('personal',7);p=w.enroll_user(b,ids,labels,**args(ids,'long'))
    held,hi,_,_=data('held',8);bad=hi.astype(object);bad[0]=bank.source_normalization_trials_['source_recording'][0]
    with pytest.raises(ValueError,match='Source trials'):w.predict(held,bad,personal=p,**args(bad,'new'))
    changed=FeatureBatch(b.emg*2,250.);other=w.enroll_user(changed,ids,labels,**args(ids,'long'))
    assert other.profile_id!=p.profile_id # Calibration identity binds the raw counts too.
