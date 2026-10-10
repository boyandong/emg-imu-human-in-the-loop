"""Software lifecycle fixtures; these are not measured device sessions."""
from dataclasses import replace
import inspect
import json
from pathlib import Path
import pickle
import zipfile
import numpy as np
import pytest
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.document_reliability_v2 import DocumentReliabilityWeightsV2
from emgimu.feature_bank.personal_session_workflow_v1 import PersonalSessionWorkflowV1

ROOT=Path(__file__).resolve().parents[1]
CHANNELS=tuple(f'measured_{i}' for i in range(8))


def windows(prefix,n=2):
    labels=np.repeat(np.arange(6),n*2)
    rng=np.random.default_rng(27+len(prefix))
    x=rng.normal(size=(len(labels),40,8))*(.05+labels[:,None,None])
    x*=1+np.arange(8)[None,None,:]*(labels[:,None,None]+1)/9
    ids=np.array([f'{prefix}_{c}_{t}' for c in range(6) for t in range(n) for _ in range(2)],dtype=object)
    return FeatureBatch(x.astype(np.float32),200.),ids,np.tile([0,1],len(ids)//2),dict(zip(ids,labels))


@pytest.fixture(scope='module')
def workflow():
    bank=pickle.loads((ROOT/'feature_bank/models/epn_emg_calibrated_bank_v1.pkl').read_bytes())
    return PersonalSessionWorkflowV1(bank,channel_ids=CHANNELS,preprocessing_id='fixture_preprocessing',rest_label=0)


def kwargs(session='fixture_new'):
    return dict(user_id='fixture_user',session_id=session,observed_channel_ids=CHANNELS,preprocessing_id='fixture_preprocessing')


@pytest.fixture
def profiles(workflow):
    b,ids,o,y=windows('long',4)
    personal=workflow.enroll_user(b,ids,y,window_offsets=o,**kwargs('fixture_long'))
    b,ids,o,y=windows('current',2)
    session=workflow.calibrate_session(b,ids,y,window_offsets=o,personal=personal,**kwargs())
    return personal,session


def test_personal_and_session_calibration_leave_source_and_long_term_unchanged(workflow):
    before=pickle.dumps(workflow.bank);b,ids,o,y=windows('long',4)
    personal=workflow.enroll_user(b,ids,y,window_offsets=o,**kwargs('fixture_long'))
    long_before=pickle.dumps(personal);b,ids,o,y=windows('current',2)
    session=workflow.calibrate_session(b,ids,y,window_offsets=o,personal=personal,**kwargs())
    assert pickle.dumps(workflow.bank)==before and pickle.dumps(personal)==long_before
    assert len(personal.calibration_trials)==24 and len(session.calibration_trials)==12
    for name,anchor in session.anchors.items():
        np.testing.assert_allclose(session.prototype_beta[name],2/3,rtol=0,atol=1e-12)
        np.testing.assert_allclose(anchor['blended'].prototypes_,
            personal.anchors[name].prototypes_*2/3+anchor['local'].prototypes_/3,rtol=0,atol=1e-12)
    readout=workflow.bank.predict_providers(b,ids,window_offsets=o,user_id='fixture_user')
    unique=readout['trial_ids'];labels=np.array([y[t] for t in unique])
    p=workflow.bank.policy_
    expected=DocumentReliabilityWeightsV2(p.classes,p.providers,personal.fusion_state.fusion_state.weights,
        p.n0,p.temperature,personal.profile_id).calculate({k:(v,labels,unique) for k,v in readout['source_standardized_features'].items()})
    np.testing.assert_allclose(session.fusion_state.weights,expected['final'],rtol=0,atol=1e-12)
    assert session.descriptor['current_sessions']==['fixture_new']


def test_predict_has_no_labels_preserves_state_and_handles_missing_providers(workflow,profiles):
    personal,session=profiles;b,ids,o,_=windows('heldout')
    before=pickle.dumps((workflow,personal,session))
    assert 'labels' not in inspect.signature(workflow.predict).parameters
    result=workflow.predict(b,ids,window_offsets=o,personal=personal,session=session,**kwargs())
    assert result['probabilities'].shape==(12,6) and result['quality_observations'].shape==(24,69)
    assert result['anchor_classes']['F0']==tuple(range(6)) and result['session_profile_id']==session.profile_id
    assert all(set(v)=={'long_term','local','blended'} for v in result['anchor_coordinates'].values())
    # Unknown ADC/ring/filtered-low-frequency observations carry availability=0.
    np.testing.assert_array_equal(result['quality_observations'][:,-4:],0.)
    single=workflow.predict(b,ids,window_offsets=o,personal=personal,session=session,available=('F0',),**kwargs())
    expected=workflow.bank.predict_providers(b,ids,window_offsets=o,user_id='fixture_user',available=('F0',))
    np.testing.assert_array_equal(single['probabilities'],expected['probabilities']['F0'])
    assert single['weights']==(1.,) and set(single['anchor_coordinates'])=={'F0'}
    assert pickle.dumps((workflow,personal,session))==before


def test_zero_session_matches_existing_bank_and_normalization_is_explicit(workflow,profiles):
    personal,session=profiles;b,ids,o,_=windows('heldout')
    result=workflow.predict(b,ids,window_offsets=o,personal=personal,**kwargs())
    expected=workflow.bank.predict(b,ids,window_offsets=o,user_id='fixture_user',user_state=personal.fusion_state)
    np.testing.assert_array_equal(result['probabilities'],expected['probabilities'])
    assert result['session_descriptor'] is None and result['session_profile_id'] is None
    view=workflow.normalized_view(b,personal=personal,session=session,**kwargs())
    np.testing.assert_array_equal(view.emg,(b.emg.astype(float)-session.normalizer.center_)/(session.normalizer.scale_+1e-10))
    permutation=np.arange(8)[::-1];reordered=FeatureBatch(b.emg[:,:,permutation],200.)
    args=kwargs();args['observed_channel_ids']=tuple(np.array(CHANNELS)[permutation])
    actual=workflow.predict(reordered,ids,window_offsets=o,personal=personal,session=session,**args)
    reference=workflow.predict(b,ids,window_offsets=o,personal=personal,session=session,**kwargs())
    np.testing.assert_array_equal(actual['probabilities'],reference['probabilities'])


@pytest.mark.parametrize('case',['user','session','personal_binding','contract','channel_ids','rate','preprocessing','long_trials','session_trials','source_trials'])
def test_wrong_identity_or_leakage_rejected(workflow,profiles,case):
    personal,session=profiles;b,ids,o,_=windows('heldout');args=kwargs()
    if case=='user':args['user_id']='other'
    elif case=='session':args['session_id']='other'
    elif case=='personal_binding':session=replace(session,personal_profile_id='other')
    elif case=='contract':personal=replace(personal,contract_id='other')
    elif case=='channel_ids':args['observed_channel_ids']=('missing',)+CHANNELS[1:]
    elif case=='rate':b=FeatureBatch(b.emg,250.)
    elif case=='preprocessing':args['preprocessing_id']='other'
    elif case=='long_trials':ids[:2]=personal.calibration_trials[0]
    elif case=='session_trials':ids[:2]=session.calibration_trials[0]
    else:ids[:2]=workflow.bank.policy_.source_trials[0]
    with pytest.raises(ValueError):workflow.predict(b,ids,window_offsets=o,personal=personal,session=session,**args)


def test_session_calibration_rejects_existing_session_and_reserved_evaluation(workflow,profiles):
    personal,_=profiles;b,ids,o,y=windows('current')
    with pytest.raises(ValueError,match='new session'):workflow.calibrate_session(b,ids,y,window_offsets=o,personal=personal,**kwargs(personal.session_id))
    with pytest.raises(ValueError,match='overlaps'):workflow.calibrate_session(b,ids,y,window_offsets=o,personal=personal,forbidden_evaluation_trials=(ids[0],),**kwargs())
    b,ids,o,y=windows('long',4)
    with pytest.raises(ValueError,match='overlaps'):workflow.calibrate_session(b,ids,y,window_offsets=o,personal=personal,**kwargs())


def test_save_load_roundtrip_checks_identity_and_keeps_predictions_exact(workflow,profiles,tmp_path):
    personal,session=profiles;p=tmp_path/'personal.zip';s=tmp_path/'session.zip'
    workflow.save_profile(personal,p);workflow.save_profile(session,s)
    restored_personal=workflow.load_profile(p,user_id='fixture_user')
    restored_session=workflow.load_profile(s,user_id='fixture_user',session_id='fixture_new',personal=restored_personal)
    b,ids,o,_=windows('heldout')
    expected=workflow.predict(b,ids,window_offsets=o,personal=personal,session=session,**kwargs())
    actual=workflow.predict(b,ids,window_offsets=o,personal=restored_personal,session=restored_session,**kwargs())
    np.testing.assert_array_equal(actual['probabilities'],expected['probabilities'])
    for k in actual['anchor_coordinates']:
        for mode in actual['anchor_coordinates'][k]:np.testing.assert_array_equal(actual['anchor_coordinates'][k][mode],expected['anchor_coordinates'][k][mode])
    with pytest.raises(FileExistsError):workflow.save_profile(personal,p)
    for args in [dict(user_id='other'),dict(user_id='fixture_user',session_id='wrong',personal=personal),
                 dict(user_id='fixture_user',session_id='fixture_new',personal=replace(personal,profile_id='other'))]:
        with pytest.raises(ValueError):workflow.load_profile(s,**args)
    with zipfile.ZipFile(s) as archive:manifest=archive.read('manifest.json');payload=archive.read('profile.pkl')
    bad=tmp_path/'damaged.zip'
    with zipfile.ZipFile(bad,'w') as archive:
        archive.writestr('manifest.json',manifest);archive.writestr('profile.pkl',payload+b'damaged')
    with pytest.raises(ValueError,match='checksum'):workflow.load_profile(bad,user_id='fixture_user',session_id='fixture_new',personal=personal)
