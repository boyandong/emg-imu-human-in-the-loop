"""Direct known-waveform checks for document F5 ratios and trajectory terms."""
import pickle

import numpy as np
import pytest

from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.document_temporal_v3 import DocumentTemporalFormV3


def test_every_temporal_coordinate_against_known_eight_sample_waveform():
    wave = np.array([[1., 3.], [2., 3.], [3., 2.], [4., 2.],
                     [5., 1.], [6., 1.], [7., 0.], [8., 0.]])
    batch = FeatureBatch(wave[None], 1000.)
    family = DocumentTemporalFormV3(envelope_ms=1.).fit(batch)
    before = pickle.dumps(family)
    actual = family.transform(batch)[0]
    names = family.feature_names
    early = np.sqrt(np.mean(wave[:4] ** 2, axis=0))
    late = np.sqrt(np.mean(wave[4:] ** 2, axis=0))
    np.testing.assert_allclose([actual[names.index(f'F5v3.early_rms.ch{c}')]
                                for c in (1, 2)], early, rtol=1e-6)
    np.testing.assert_allclose([actual[names.index(f'F5v3.late_rms.ch{c}')]
                                for c in (1, 2)], late, rtol=1e-6)
    np.testing.assert_allclose([actual[names.index(f'F5v3.early_over_late.ch{c}')]
                                for c in (1, 2)], early / (late + 1e-10), rtol=1e-6)
    np.testing.assert_allclose([actual[names.index(f'F5v3.late_minus_early.ch{c}')]
                                for c in (1, 2)], late - early, rtol=1e-6)
    slopes = [np.polyfit(np.arange(8) / 1000., wave[:, c], 1)[0] for c in (0, 1)]
    np.testing.assert_allclose([actual[names.index(f'F5v3.envelope_slope_per_second.ch{c}')]
                                for c in (1, 2)], slopes, rtol=1e-6)
    np.testing.assert_allclose([actual[names.index(f'F5v3.peak_index_over_T.ch{c}')]
                                for c in (1, 2)], [7 / 8, 0.], atol=1e-7)
    q = wave / (wave.sum(axis=0) + 1e-10)
    entropy = -np.sum(q * np.log(q + 1e-10), axis=0)
    np.testing.assert_allclose([actual[names.index(f'F5v3.temporal_entropy.ch{c}')]
                                for c in (1, 2)], entropy, rtol=1e-6)
    p = wave / (wave.sum(axis=1, keepdims=True) + 1e-10)
    velocity = np.linalg.norm(np.diff(p, axis=0), axis=1).mean()
    assert actual[-1] == pytest.approx(velocity, rel=1e-6)
    assert len(names) == 15
    assert pickle.dumps(family) == before


def test_constant_and_zero_waveforms_have_no_padded_edge_activity():
    batch = FeatureBatch(np.stack([np.zeros((40, 2)), np.ones((40, 2))]), 200.)
    family = DocumentTemporalFormV3().fit(batch)
    observed = family.transform(batch)
    names = family.feature_names
    assert np.isfinite(observed).all()
    np.testing.assert_allclose(observed[:, [names.index('F5v3.envelope_slope_per_second.ch1'),
                                           names.index('F5v3.spatial_map_velocity')]], 0., atol=1e-12)
    with pytest.raises(ValueError, match='contract'):
        family.transform(FeatureBatch(batch.emg, 250.))


@pytest.mark.parametrize('gain',[1.,1e-6])
def test_default_25ms_envelope_on_odd_windows_against_explicit_local_means(gain):
    # At 250 Hz the fixed 25 ms envelope spans six samples. Independent
    # local slices exercise its asymmetric even-width padding and odd split.
    wave=gain*np.array([[1.,4.],[2.,3.],[5.,1.],[3.,2.],[0.,4.],[6.,1.],
                        [2.,0.],[7.,2.],[1.,3.],[4.,1.],[0.,2.]])
    batch=FeatureBatch(wave[None],250.)
    family=DocumentTemporalFormV3().fit(batch);frozen=pickle.dumps(family)
    observed=family.transform(batch)[0]
    padded=np.vstack([np.repeat(wave[:1],2,axis=0),wave,np.repeat(wave[-1:],3,axis=0)])
    envelope=np.stack([np.sqrt(np.mean(padded[i:i+6]**2,axis=0)) for i in range(11)])
    early=np.sqrt(np.mean(wave[:5]**2,axis=0));late=np.sqrt(np.mean(wave[5:]**2,axis=0))
    slope=np.linalg.lstsq(np.column_stack([np.arange(11)/250.,np.ones(11)]),envelope,rcond=None)[0][0]
    q=envelope/(envelope.sum(axis=0)+1e-10)
    p=envelope/(envelope.sum(axis=1,keepdims=True)+1e-10)
    expected=np.r_[early,late,early/(late+1e-10),late-early,slope,
                   np.argmax(envelope,axis=0)/11.,-(q*np.log(q+1e-10)).sum(axis=0),
                   np.linalg.norm(np.diff(p,axis=0),axis=1).mean()]
    np.testing.assert_allclose(observed,expected,atol=1e-7,rtol=1e-6)
    assert observed.size==7*2+1
    assert pickle.dumps(family)==frozen
