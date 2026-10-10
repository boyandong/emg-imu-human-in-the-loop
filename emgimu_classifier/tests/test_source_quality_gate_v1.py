"""Independent structural boundaries, effective weights and input availability."""
import pickle
import numpy as np
import pytest
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.source_quality_gate_v1 import SourceQualityGateV1

CHANNELS=tuple(f'CH{i+1}' for i in range(8))


def gate():
    x=np.random.default_rng(21).integers(-1000,1000,size=(30,50,8))
    return SourceQualityGateV1(channel_ids=CHANNELS,adc_range=(-8388608,8388607),
        adc_range_provenance='signed24 acquisition fixture').fit(FeatureBatch(x,250.),[f'source{i}' for i in range(30)])


def test_structural_thresholds_and_unknown_are_exact_not_a_rest_prediction():
    g=gate();x=np.random.default_rng(9).integers(-1000,1000,size=(4,50,8))
    x[0,:,2]=0;x[1,:,2]=8388607;x[2,:26,2]=17;x[3,:25,2]=17
    result=g.observe(FeatureBatch(x,250.),['a','b','c','d'],observed_channel_ids=CHANNELS)
    bad=result['trial_structural_invalid_channels']
    assert bad[:,2].tolist()==[True,True,True,False]
    q=np.array([[.7,.1,.1,.1]]*4)
    r=g.decide({'F0':q,'F1':q},[.4,.6],FeatureBatch(x,250.),['a','b','c','d'],
        observed_channel_ids=CHANNELS,class_names=('fist','pinch','neutral','open'),mode='structural',
        provider_trial_ids={'F0':('a','b','c','d'),'F1':('a','b','c','d')})
    assert r['labels']==('Unknown','Unknown','Unknown','fist') and r['rejected'].tolist()==[True,True,True,False]
    np.testing.assert_allclose(r['probabilities'],q,rtol=0,atol=1e-12) # Retained scores are not claimed accepted.


def test_soft_quality_multiplies_source_weights_and_missing_branch_renormalizes():
    g=gate();before=pickle.dumps(g);x=np.random.default_rng(7).integers(-1000,1000,size=(2,50,8));x[0,:,2]=0
    providers={'F0':np.array([[.7,.1,.1,.1]]*2),'F1':np.array([[.1,.1,.1,.7]]*2)}
    r=g.decide(providers,[.4,.6],FeatureBatch(x,250.),['a','b'],observed_channel_ids=CHANNELS,
        class_names=('a','b','c','d'),mode='soft',provider_trial_ids={'F0':('a','b'),'F1':('a','b')})
    cq=r['channel_quality'];weights=np.column_stack([.4*cq.min(1),.6*cq.mean(1)])
    weights/=weights.sum(1,keepdims=True)
    np.testing.assert_allclose(r['provider_weights'],weights,rtol=0,atol=1e-12)
    expected=weights[:,0,None]*providers['F0']+weights[:,1,None]*providers['F1']
    np.testing.assert_allclose(r['probabilities'],expected,rtol=0,atol=1e-12)
    assert cq[0,2]==0 and not r['rejected'][0] # Mean-quality branches survive a single failed channel.
    assert pickle.dumps(g)==before


def test_channel_reorder_trial_pooling_source_leakage_and_raw_contract():
    g=gate();x=np.random.default_rng(8).integers(-1000,1000,size=(3,50,8));x[1,:,1]=0
    a=g.observe(FeatureBatch(x,250.),['trial','trial','other'],observed_channel_ids=CHANNELS)
    b=g.observe(FeatureBatch(x[:,:,::-1],250.),['trial','trial','other'],observed_channel_ids=CHANNELS[::-1])
    np.testing.assert_array_equal(a['trial_structural_invalid_channels'],b['trial_structural_invalid_channels'])
    assert a['trial_ids']==('other','trial') and a['trial_structural_invalid_channels'][1,1]
    for batch,ids,channels in [(FeatureBatch(x,200.),['a','b','c'],CHANNELS),
        (FeatureBatch(x,250.),['source0','b','c'],CHANNELS),(FeatureBatch(x,250.),['a','b','c'],['CH1']*8)]:
        with pytest.raises(ValueError):g.observe(batch,ids,observed_channel_ids=channels)
