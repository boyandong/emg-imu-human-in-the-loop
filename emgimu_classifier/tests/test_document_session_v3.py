import pickle
import numpy as np
import pytest
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.document_session_v3 import DocumentSessionDescriptorV3


def test_four_block_descriptor_has_independent_known_geometry_and_names():
    source={0:dict(pattern=np.array([1.,0.]),log_bands=np.zeros(2),
                   covariance=np.eye(8),channel_quality=np.ones(8),log_scale=0.),
            1:dict(pattern=np.array([0.,1.]),log_bands=np.ones(2),
                   covariance=np.eye(8),channel_quality=np.ones(8),log_scale=0.)}
    current={0:dict(pattern=np.array([2.,0.]),log_bands=np.ones(2),
                    covariance=2*np.eye(8),channel_quality=.5*np.ones(8),log_scale=np.log(2.)),
             1:dict(pattern=np.array([0.,2.]),log_bands=2*np.ones(2),
                    covariance=2*np.eye(8),channel_quality=.5*np.ones(8),log_scale=np.log(2.))}
    frozen=pickle.dumps((source,current))
    values,names=DocumentSessionDescriptorV3._assemble(source,current)
    # H=2, K=4: 2HK + K*C(H,2) + H*(4+C) = 44.
    assert values.shape==(44,)
    assert len(names)==len(set(names))==44
    np.testing.assert_allclose(values[:8],[1.,1.,np.sqrt(2),np.sqrt(2),np.sqrt(8),np.sqrt(8),np.sqrt(2),np.sqrt(2)])
    np.testing.assert_allclose(values[8:16],[1.,1.,0.,1.,1.,1.,1.,1.])
    np.testing.assert_allclose(values[16:20],[np.sqrt(2),0.,0.,0.],atol=1e-12)
    expected_one=np.r_[np.log(2.),1.,1.,np.sqrt(8)*np.log(2.),np.full(8,-.5)]
    np.testing.assert_allclose(values[20:],np.tile(expected_one,2))
    assert names[16]=='F8.pattern.geometry_delta.0.1'
    assert names[-1]=='F8.channel_quality_shift.1.ch8'
    assert pickle.dumps((source,current))==frozen


def fixture(ring=False):
    amplitude=np.repeat(np.arange(1.,5.),2)
    wave=(-1.)**np.arange(200)
    x=amplitude[:,None,None]*wave[None,:,None]*np.ones((1,1,8))
    labels=np.repeat([0,1],4)
    ids=np.repeat(['s-a','s-b','s-c','s-d'],2)
    batch=FeatureBatch(x,1000.)
    fitted=DocumentSessionDescriptorV3().fit_long_term(batch,labels,ids,
        user_id='u1',session_ids=['day1']*8,ring_topology=ring)
    return fitted,batch,labels


def test_channel_quality_difference_is_not_channel_variance_and_is_frozen():
    fitted,batch,labels=fixture()
    x=batch.emg.copy();x[:,:,0]=0.
    before=pickle.dumps(fitted)
    result=fitted.from_calibration(FeatureBatch(x,1000.),labels,
        np.repeat(['c-a','c-b','c-c','c-d'],2),user_id='u1',session_ids=['day2']*8)
    # Source RMS median=2.5, MAD=1.0; source quality=1-|z|/6.
    expected=-np.mean(1.-np.abs(np.array([1.,2.])-2.5)/(1.4826*6.))
    np.testing.assert_allclose(result['classes']['0']['channel_quality_score_shift'],
                               np.r_[expected,np.zeros(7)],atol=1e-7)
    assert result['classes']['0']['channel_variance_residual_json'][0] < -1.
    assert result['phi_dimension']==44
    assert len(result['phi_feature_names'])==44
    assert pickle.dumps(fitted)==before


def test_rlcs_descriptor_uses_only_eight_lag_coordinates():
    fitted,batch,labels=fixture(ring=True)
    assert fitted.reference_[0]['ring'].shape==(8,)
    result=fitted.from_calibration(batch,labels,[f'cal-{i}' for i in range(8)],
                                  user_id='u1',session_ids=['day2']*8)
    assert result['phi_dimension']==51
    assert result['classes']['0']['ring_vector_residual_norm']==pytest.approx(0.)


@pytest.mark.parametrize('user,sessions,trials,error',[
    ('u2',['day2']*8,[f'c{i}' for i in range(8)],'same user'),
    ('u1',['day1']*8,[f'c{i}' for i in range(8)],'new sessions'),
    ('u1',['day2']*4+['day3']*4,[f'c{i}' for i in range(8)],'one session'),
    ('u1',['day2']*8,['s-a']*2+[f'c{i}' for i in range(6)],'overlap'),
])
def test_identity_and_trial_boundaries_are_rejected_without_mutation(user,sessions,trials,error):
    fitted,batch,labels=fixture();before=pickle.dumps(fitted)
    with pytest.raises(ValueError,match=error):
        fitted.from_calibration(batch,labels,trials,user_id=user,session_ids=sessions)
    assert pickle.dumps(fitted)==before


def test_session_metadata_and_native_channel_requirements():
    fitted,batch,labels=fixture()
    with pytest.raises(ValueError,match='span'):
        fitted.from_calibration(batch,labels,['c']*4+['d']*4,user_id='u1',
            session_ids=['day2','day3']*4)
    with pytest.raises(ValueError,match='eight'):
        DocumentSessionDescriptorV3().fit_long_term(FeatureBatch(batch.emg[:,:,:4],1000.),
            labels,[f's{i}' for i in range(8)],user_id='u1',session_ids=['day1']*8)
