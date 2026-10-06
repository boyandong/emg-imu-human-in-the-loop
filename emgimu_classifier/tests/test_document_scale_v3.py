import pickle
import numpy as np
import pytest
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.document_scale_v3 import DocumentScalePatternV3


@pytest.mark.parametrize('gain',[0.,1e-10,1.,31.])
def test_exact_pattern_and_separate_context_on_known_channel_rms(gain):
    channel=np.arange(1.,9.)*gain
    x=np.tile(channel,(2,40,1))
    source=FeatureBatch(x,250.)
    model=DocumentScalePatternV3().fit(source);frozen=pickle.dumps(model)
    expected_scale=np.sqrt(sum(v*v for v in channel)/8)
    expected=channel/(expected_scale+1e-10)
    np.testing.assert_allclose(model.transform(source),np.tile(expected,(2,1)),atol=1e-15)
    np.testing.assert_allclose(model.activation_context(source),np.full((2,1),np.log(expected_scale+1e-10)),atol=1e-15)
    assert len(model.feature_names)==8 and all('scale' not in n for n in model.feature_names)
    assert pickle.dumps(model)==frozen
    if gain==1e-10:
        floor=channel/max(expected_scale,1e-10)
        assert np.max(np.abs(expected-floor))>.1


def test_waveform_rms_sign_invariance_and_sensor_contract():
    wave=np.array([[1.,2.],[3.,4.],[5.,6.]])
    batch=FeatureBatch(wave[None],250.)
    model=DocumentScalePatternV3().fit(batch)
    channel=np.array([np.sqrt(35/3),np.sqrt(56/3)])
    expected=channel/(np.sqrt(np.mean(channel**2))+1e-10)
    np.testing.assert_allclose(model.transform(batch)[0],expected)
    np.testing.assert_array_equal(model.transform(batch),model.transform(FeatureBatch(-wave[None],250.)))
    for target in (FeatureBatch(np.ones((1,3,8)),250.),FeatureBatch(wave[None],200.)):
        with pytest.raises(ValueError,match='contract'):model.transform(target)
        with pytest.raises(ValueError,match='contract'):model.activation_context(target)
