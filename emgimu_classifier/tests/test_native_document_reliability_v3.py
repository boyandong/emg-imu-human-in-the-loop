from dataclasses import replace
import hashlib
import json
import pickle
import numpy as np
import pytest
from emgimu.feature_bank.native_document_reliability_v3 import (
    NativeDocumentReliabilityV3,hierarchical_weights,compose_views,policy_digest)
from test_native_joint_bout_workflow_v2 import workflow,bouts


def test_exact_hierarchical_shrinkage_counts_trials_and_preserves_long_prior():
    classes=('a','b');groups=('p','q');labels=np.array(['a','a','b','b']);ids=('a0','a1','b0','b1')
    long={'p':(np.array([[0.],[2.],[4.],[6.]]),labels,ids),'q':(np.zeros((4,1)),labels,ids)}
    current={'p':(np.array([[0.],[4.]]),np.array(['a','b']),('ca','cb')),'q':(np.zeros((2,1)),np.array(['a','b']),('ca','cb'))}
    before=pickle.dumps(long)
    weights,lr,cr=hierarchical_weights(classes,groups,(.25,.75),4.,1.,long,current,source_policy_id='fixture')
    np.testing.assert_allclose(lr['final'],(.625,.375),atol=1e-9,rtol=0)
    np.testing.assert_allclose(weights,(.75,.25),atol=1e-9,rtol=0)
    assert (lr['n_cal_trials'],cr['n_cal_trials'])==(4,2) and pickle.dumps(long)==before
    repeated={g:(np.repeat(x,3,axis=0),np.repeat(y,3),np.repeat(t,3)) for g,(x,y,t) in long.items()}
    other,_,_=hierarchical_weights(classes,groups,(.25,.75),4.,1.,repeated,current,source_policy_id='fixture')
    np.testing.assert_allclose(weights,other,atol=1e-12,rtol=0)
    with pytest.raises(ValueError,match='source/evaluation'):hierarchical_weights(classes,groups,(.25,.75),4.,1.,long,current,source_policy_id='fixture',forbidden_trials=('a0',))


def test_document_weights_anchor_routing_and_path_composition_match_rational_mixture():
    raw={'p':np.array([[.9,.1]]),'q':np.array([[.2,.8]])}
    heads={'p':np.array([[.7,.3]]),'q':np.array([[.4,.6]])}
    result=compose_views(raw,heads,[[.8,.2]],[[.6,.4]],(.25,.75),(2.,.5),('p','q'))
    assert result['document_reliability'][0,0]==pytest.approx(.375)
    assert result['document_window'][0,0]==pytest.approx(4/7)
    assert result['document_joint'][0,0]==pytest.approx(.75*4/7+.125*.8+.125*.6)
    with pytest.raises(ValueError):compose_views(raw,heads,[[.8,.2]],[[.6,.4]],(.25,.75),(2.,0.),('p','q'))
    with pytest.raises(ValueError):compose_views(raw,dict(heads,p=np.array([[.7,.7]])),[[.8,.2]],[[.6,.4]],(.25,.75),(2.,.5),('p','q'))


def adapter(w,tmp_path):
    p=dict(schema='native_document_reliability_policy_v3',inference_bank_id=w.window.bank.bank_id_,
        providers=w.window.bank.providers_,classes=w.window.bank.classes_,sensor_contract=w.window.bank.sensor_contract_,
        channel_ids=w.window.channels,preprocessing_id=w.window.preprocessing_id,population=w.window.bank.policy_.population,
        n0=4.,temperature=.5,source_query_recording_ids=['source-policy-recording'])
    p['policy_id']=policy_digest(p);path=tmp_path/'policy.json';path.write_text(json.dumps(p),encoding='utf8')
    checksum=hashlib.sha256(path.read_bytes()).hexdigest()
    return NativeDocumentReliabilityV3(w,path,expected_sha256=checksum),path


def test_native_document_policy_composes_label_free_queries_without_mutating_profiles(workflow,tmp_path):
    w=workflow;model,_=adapter(w,tmp_path);long,y=bouts(w,'long');p=w.enroll(long,y,user_id='user',session_id='long')
    current,cy=bouts(w,'current');s=w.enroll(current,cy,user_id='user',session_id='current',personal=p)
    state=model.prepare_state(personal=p,session=s,user_id='user',session_id='current')
    assert (state.long_trials,state.current_trials)==(6,6)
    query,_=bouts(w,'query');before=pickle.dumps((w,p,s))
    result=model.predict(query,personal=p,session=s,user_id='user',session_id='current',state=state)
    assert result['trial_ids']==query.trial_ids and result['calibration_trials']==12
    assert result['document_joint'].shape==(6,3)
    np.testing.assert_allclose(result['document_joint'].sum(1),1.,atol=1e-12)
    assert pickle.dumps((w,p,s))==before
    with pytest.raises(ValueError,match='Disjoint'):model.compose(replace(state,weights=(1.,0.,0.,0.,0.,0.,0.)),{},{},[],[],trial_ids=query.trial_ids,recording_ids=query.recording_ids,user_id='user',session_id='current',provider_trial_ids={},provider_classes={})
    with pytest.raises(ValueError):model.predict(long,personal=p,session=s,user_id='user',session_id='current',state=state)


def test_policy_checksum_and_source_bank_contract_are_enforced(workflow,tmp_path):
    w=workflow;_,path=adapter(w,tmp_path)
    with pytest.raises(ValueError,match='checksum'):NativeDocumentReliabilityV3(w,path,expected_sha256='0'*64)
    p=json.loads(path.read_text());p['inference_bank_id']='different';p['policy_id']=policy_digest(p);path.write_text(json.dumps(p))
    with pytest.raises(ValueError,match='contract'):NativeDocumentReliabilityV3(w,path,expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
