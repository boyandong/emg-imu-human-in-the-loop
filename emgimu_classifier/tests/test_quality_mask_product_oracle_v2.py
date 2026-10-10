"""Direct F9g source quantile, seven-factor product and summary arithmetic.

Synthetic observation tables deliberately isolate mask arithmetic from the
separately tested Fourier/raw-signal observations. They are not physical faults.
"""
import pickle

import numpy as np
import pytest

from emgimu.feature_bank.quality_mask_v1 import SourceCalibratedQualityMask


PREFIXES = ('F9.zero_fraction','F9v2.longest_flatline_ratio','F9.clip_fraction',
            'F9.amplitude_z','F9v2.correlation_anomaly','F9.line_noise_ratio',
            'F9v2.low_frequency_power_ratio')
AVAILABILITY = ('adc_clipping','line_noise','low_frequency_pre_highpass')
NAMES = tuple(f'{prefix}.ch{c}' for prefix in PREFIXES for c in (1,2,3)) + tuple(
    'F9v2.available.'+key for key in AVAILABILITY)


def source(available=True):
    # q=.9 with five rows is exactly the .6/.4 interpolation of sorted rows3/4.
    zero=np.array([0,.02,.04,.06,.08])
    flat=np.array([.01,.03,.05,.07,.09])
    clip=np.array([0,.002,.004,.006,.008])
    amplitude=np.array([-1,2,-4,8,-10])
    correlation=np.array([0,2,4,8,16])
    line=np.array([1,2,4,8,24])
    low=np.array([0,3,6,9,30])
    values=np.column_stack([v for component in (zero,flat,clip,amplitude,correlation,line,low)
                            for v in (component,component*.5,component*.25)])
    return np.column_stack((values,np.full((5,3),float(available))))


def test_independent_source_quantile_product_and_four_summaries_with_exact_boundaries():
    model=SourceCalibratedQualityMask(source_quantile=.9).fit(source(),NAMES)
    expected_thresholds=np.array([[.1,.1,.1],[.1,.1,.1],[.01,.01,.01],
                                 [9.2,6,6],[12.8,6.4,6],[17.6,10,10],[21.6,10.8,10]])
    for key, threshold in zip(('zero','flat','clip','amplitude','correlation','line','low_frequency'),expected_thresholds):
        np.testing.assert_allclose(model.thresholds_[key],threshold,atol=1e-14,rtol=0)
    target=np.column_stack((np.tile(expected_thresholds.reshape(-1),(4,1)),np.ones((4,3))))
    # Row0: three independent moderate anomalies on ch1 and two on ch2.
    target[0,9]=1.25*9.2;target[0,12]=1.5*12.8;target[0,15]=1.2*17.6
    target[0,4]=1.5*.1;target[0,7]=1.5*.01
    # Row1 is at every source threshold (severity zero). Row2 is at twice
    # the zero threshold (severity one); row3 tests negative amplitude abs.
    target[2,:3]=.2
    target[3,9]=-1.25*9.2;target[3,10]=-2*6
    frozen=pickle.dumps((model,target))
    expected_channels=np.array([[.3,.25,1],[1,1,1],[0,0,0],[.75,0,1]])
    expected=np.column_stack((expected_channels,np.array([2,0,3,1]),
                              np.array([31/60,1,0,7/12]),np.array([.25,1,0,0]),
                              np.array([211/1800,0,0,13/72])))
    observed=model.transform(target,NAMES)
    np.testing.assert_allclose(observed,expected,atol=5e-8,rtol=0)
    np.testing.assert_array_equal(model.transform(target[[2,0,3,1]],NAMES),observed[[2,0,3,1]])
    assert pickle.dumps((model,target)) == frozen
    # Quality score rejection and structural invalidity are separate rules.
    assert not model.structural_invalid(target,NAMES).any()
    structural=target[:1].copy();structural[0,[0,4,8]]=[.5,.5,.1]
    np.testing.assert_array_equal(model.structural_invalid(structural,NAMES),[[True,True,True]])


def test_unavailable_adc_line_and_raw_low_band_never_become_quality_evidence():
    model=SourceCalibratedQualityMask(source_quantile=.9).fit(source(False),NAMES)
    target=source(False)[:1].copy();target[:,6:9]=1e9;target[:,15:21]=1e9
    # None of the unavailable observations can lower quality, even if their
    # numeric placeholders are large. Availability remains explicit in state.
    np.testing.assert_array_equal(model.transform(target,NAMES),[[1,1,1,0,1,1,0]])
    assert model.available_ == dict(adc_clipping=False,line_noise=False,low_frequency_pre_highpass=False)
    assert not model.structural_invalid(target,NAMES).any()
    target[0,-1]=1
    with pytest.raises(ValueError,match='availability'):model.transform(target,NAMES)


def test_source_quality_quantile_is_not_refit_by_extreme_query_rows():
    model=SourceCalibratedQualityMask(source_quantile=.9).fit(source(),NAMES)
    before=pickle.dumps(model)
    target=source()[:2].copy();target[1,:21]=1e20
    result=model.transform(target,NAMES)
    np.testing.assert_array_equal(result[0],model.transform(target[:1],NAMES)[0])
    np.testing.assert_array_equal(result[1,:3],0)
    assert pickle.dumps(model)==before
