import pickle
import numpy as np
import pytest
from scipy.fft import dct
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.document_spectral_v3 import DocumentSpectralStateV3
from emgimu.feature_bank.families import SpectralStateFamily


@pytest.mark.parametrize('gain',[1.,1e-6,0.])
def test_all_spectral_blocks_against_direct_fourier_and_independent_dct(gain):
    n=100;rate=250.;t=np.arange(n)/rate
    raw=gain*np.stack((np.sin(2*np.pi*30*t),.4*np.cos(2*np.pi*75*t),np.zeros(n)),axis=1)
    batch=FeatureBatch(raw[None],rate)
    model=DocumentSpectralStateV3().fit(batch);before=pickle.dumps(model)
    actual=model.transform(batch)[0]
    taper=(raw-raw.mean(axis=0))*(.5-.5*np.cos(2*np.pi*np.arange(n)/(n-1)))[:,None]
    freq=np.arange(n//2+1)*rate/n
    real=np.cos(2*np.pi*np.outer(np.arange(n//2+1),np.arange(n))/n)@taper
    imag=-np.sin(2*np.pi*np.outer(np.arange(n//2+1),np.arange(n))/n)@taper
    power=(real**2+imag**2)/n
    expected=[]
    for i,(low,high) in enumerate(model.bands_):
        mask=(freq>=low)&(freq<=high if i==3 else freq<high)
        energy=power[mask].sum(axis=0)
        expected.extend(energy/(np.sqrt(np.sum(energy**2))+1e-10))
    total=power.sum(axis=0);q=power/(total+1e-10)
    expected.extend(total)
    expected.extend((freq[:,None]*power).sum(axis=0)/(total+1e-10))
    expected.extend(freq[np.argmax(np.cumsum(power,axis=0)>=total*.5,axis=0)])
    expected.extend(-(q*np.log(q+1e-10)).sum(axis=0))
    coefficients=dct(np.log(power+1e-10),type=2,norm='ortho',axis=0)
    for k in range(1,5):expected.extend((coefficients[k].mean(),coefficients[k].std()))
    np.testing.assert_allclose(actual,expected,atol=1e-5,rtol=1e-6)
    assert actual.shape==(4*3+4*3+2*4,)
    assert len(model.feature_names)==len(set(model.feature_names))==32
    assert pickle.dumps(model)==before
    if gain==1e-6:
        legacy=SpectralStateFamily().fit_transform(batch)[0]
        assert not np.allclose(actual[:12],legacy[:12],atol=1e-3)
        assert not np.allclose(actual[15:18],legacy[15:18],atol=1e-3)
    if gain==0.:
        np.testing.assert_allclose(actual[:24],0.)


def test_source_grid_and_hyperparameters_are_fixed():
    source=FeatureBatch(np.ones((2,100,8)),250.)
    model=DocumentSpectralStateV3().fit(source)
    before=model.transform(source);names=model.feature_names
    model.band_count=2;model.cepstral_coefficients=2
    np.testing.assert_array_equal(model.transform(source),before)
    assert model.feature_names==names
    for raw,rate in ((np.ones((2,100,4)),250.),(np.ones((2,200,8)),250.),(np.ones((2,100,8)),500.)):
        with pytest.raises(ValueError,match='contract'):model.transform(FeatureBatch(raw,rate))
    with pytest.raises(ValueError,match='exclude DC'):
        DocumentSpectralStateV3(cepstral_coefficients=51).fit(source)
    with pytest.raises(ValueError,match='empty'):
        DocumentSpectralStateV3(band_count=100).fit(source)
    for value in (0,-1,1.5,True):
        with pytest.raises(ValueError,match='integer'):DocumentSpectralStateV3(band_count=value)
