import numpy as np
import pytest

from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.families import SpdTangentFamily
from emgimu.feature_bank.new_bank_v2 import (
    DocumentRingRelativeCovarianceV2, DocumentSpdTangentV2, DocumentTraceCovarianceV2, RestNoiseDetailV2,
    RingRelativeCovarianceV2, TraceCovarianceV2,
    new_bank_v2_registry,
)


def test_v2_registers_only_independent_v1_and_new_blocks():
    registry = new_bank_v2_registry()
    assert len(registry.family_ids) == 10
    assert isinstance(registry.create("new_v2_rest_noise_detail"), RestNoiseDetailV2)
    assert isinstance(registry.create("new_v2_trace_covariance"), TraceCovarianceV2)
    assert isinstance(registry.create("new_v2_document_trace_covariance"), DocumentTraceCovarianceV2)
    assert isinstance(registry.create("new_v2_document_spd_tangent"), DocumentSpdTangentV2)
    assert isinstance(registry.create("new_v2_document_ring_relative_covariance"), DocumentRingRelativeCovarianceV2)
    assert isinstance(registry.create("new_v2_ring_relative_covariance"), RingRelativeCovarianceV2)


def test_rest_noise_detail_requires_rest_and_uses_only_rest_thresholds():
    rng = np.random.default_rng(112)
    rest = rng.normal(scale=0.02, size=(4, 200, 8))
    active = rng.normal(scale=20, size=(4, 200, 8))
    source = FeatureBatch(np.concatenate((rest, active)), 1000.0)
    labels = np.array([0] * 4 + [1] * 4)
    family = RestNoiseDetailV2(rest_label=0)
    with pytest.raises(ValueError, match="Rest"):
        family.fit(source, np.ones(8))
    family.fit(source, labels)
    thresholds = family.thresholds_.copy()
    modified = FeatureBatch(np.concatenate((rest, active * 100)), 1000.0)
    second = RestNoiseDetailV2(rest_label=0).fit(modified, labels)
    np.testing.assert_allclose(thresholds, second.thresholds_, rtol=0, atol=0)
    assert family.rest_windows_ == 4
    assert family.transform(source).shape == (8, 48)
    with pytest.raises(ValueError, match="sample rate"):
        family.transform(FeatureBatch(source.emg, 999.0))


def test_rest_noise_detail_six_document_formulas_on_known_waveform():
    # Source Rest fixes the noise floor; the held waveform exercises all counts.
    rest = np.zeros((1, 4, 8), dtype=np.float64)
    active = np.broadcast_to(np.array([-2.0, -1.0, 1.0, 0.0])[None, :, None],
                             (1, 4, 8)).copy()
    batch = FeatureBatch(np.concatenate((rest, active)), 20.0)
    family = RestNoiseDetailV2(rest_label=0).fit(batch, np.array([0, 1]))
    output = family.transform(batch)
    np.testing.assert_allclose(output[0], 0.0, atol=0)
    expected = (np.sqrt(1.5), 1.0, 4.0, 1.0, 1.0, 3.0)
    for block, value in enumerate(expected):
        np.testing.assert_allclose(output[1, block * 8:(block + 1) * 8],
                                   value, atol=1e-6)
    np.testing.assert_array_equal(family.thresholds_, np.full(8, 1e-12))


def test_trace_covariance_matches_formula_and_is_gain_invariant():
    rng = np.random.default_rng(113)
    x = rng.normal(size=(3, 250, 8)) * np.arange(1, 9)
    batch = FeatureBatch(x, 1000.0)
    family = TraceCovarianceV2(shrinkage=0.05).fit(batch)
    actual = family.transform(batch)
    centered = x[0] - x[0].mean(axis=0)
    covariance = centered.T @ centered / 249
    shrunk = 0.95 * covariance + 0.05 * np.trace(covariance) / 8 * np.eye(8)
    normalized = shrunk / np.trace(shrunk)
    i, j = np.triu_indices(8)
    expected = normalized[i, j].copy()
    expected[i != j] *= np.sqrt(2)
    np.testing.assert_allclose(actual[0], expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(actual, family.transform(FeatureBatch(x * 7, 1000.0)), atol=1e-7)
    assert actual.shape == (3, 36)


def test_document_trace_covariance_matches_uncentered_formula_and_preserves_offset():
    samples=40
    signal=np.zeros((2,samples,8),dtype=float)
    signal[0,:,0]=2.
    signal[0,:,1]=1.
    signal[1,:,0]=np.linspace(-1.,1.,samples)
    batch=FeatureBatch(signal,200.)
    family=DocumentTraceCovarianceV2(shrinkage=.05).fit(batch)
    actual=family.transform(batch)
    moment=signal[0].T@signal[0]/samples
    shrunk=.95*moment+.05*np.trace(moment)/8*np.eye(8)
    expected=shrunk/np.trace(shrunk)
    i,j=np.triu_indices(8)
    vector=expected[i,j].copy();vector[i!=j]*=np.sqrt(2.)
    np.testing.assert_allclose(actual[0],vector,rtol=1e-6,atol=1e-7)
    np.testing.assert_allclose(actual,family.transform(FeatureBatch(signal*7,200.)),atol=1e-7)
    centered=TraceCovarianceV2(shrinkage=.05).fit(batch).transform(batch)
    assert np.max(np.abs(actual[0]-centered[0]))>.1
    assert actual.shape==(2,36)
    with pytest.raises(ValueError,match='sample rate'):
        family.transform(FeatureBatch(signal,250.))


def test_document_spd_uses_uncentered_source_reference_without_target_refit():
    import pickle
    rng=np.random.default_rng(20261001)
    source=rng.normal(size=(4,40,8))*.2+np.arange(1,9)[None,None,:]
    target=rng.normal(size=(2,40,8))*.2+np.arange(8,0,-1)[None,None,:]
    family=DocumentSpdTangentV2(shrinkage=.05).fit(FeatureBatch(source,200.))
    frozen=pickle.dumps(family)
    observed=family.transform(FeatureBatch(target,200.))
    expected=[]
    reference=family.reference_
    eigen,rotation=np.linalg.eigh(reference)
    inverse_root=(rotation*eigen**-.5)@rotation.T
    for window in target:
        moment=window.T@window/len(window)
        regularized=.95*moment+.05*np.trace(moment)/8*np.eye(8)+1e-12*np.eye(8)
        covariance=regularized/np.trace(regularized)
        whitened=inverse_root@covariance@inverse_root
        values,vectors=np.linalg.eigh((whitened+whitened.T)/2)
        tangent=(vectors*np.log(values))@vectors.T
        i,j=np.triu_indices(8)
        coordinates=tangent[i,j].copy();coordinates[i!=j]*=np.sqrt(2)
        expected.append(coordinates)
    np.testing.assert_allclose(observed,expected,rtol=1e-5,atol=1e-5)
    assert pickle.dumps(family)==frozen
    assert observed.shape==(2,36)
    centered=SpdTangentFamily().fit(FeatureBatch(source,200.)).transform(FeatureBatch(target,200.))
    assert np.max(np.abs(observed-centered))>.01
    with pytest.raises(ValueError,match='sample rate'):
        family.transform(FeatureBatch(target,250.))


def test_ring_covariance_preserves_rotation_but_detects_nonring_swap():
    rng = np.random.default_rng(114)
    x = rng.normal(size=(5, 200, 8))
    x[:, :, 1] = 0.8 * x[:, :, 0] + 0.2 * rng.normal(size=(5, 200))
    source = FeatureBatch(x, 1000.0)
    family = RingRelativeCovarianceV2().fit(source)
    actual = family.transform(source)
    rotated = family.transform(FeatureBatch(np.roll(x, 2, axis=2), 1000.0))
    swapped = family.transform(FeatureBatch(x[:, :, [0, 2, 1, 3, 4, 5, 6, 7]], 1000.0))
    assert actual.shape == (5, 24)
    np.testing.assert_allclose(actual, rotated, atol=1e-6)
    assert np.max(np.abs(actual - swapped)) > 1e-4
    np.testing.assert_allclose(actual, family.transform(FeatureBatch(x * 5, 1000.0)), atol=1e-6)


def test_ring_covariance_short_window_omits_unstable_temporal_block():
    short = FeatureBatch(np.zeros((2, 50, 8)), 250.0)
    family = RingRelativeCovarianceV2().fit(short)
    assert not family.with_temporal_
    assert len(family.feature_names) == 20
    np.testing.assert_array_equal(family.transform(short), np.zeros((2, 20)))
    with pytest.raises(ValueError, match="window samples"):
        family.transform(FeatureBatch(np.zeros((2, 51, 8)), 250.0))


def test_document_ring_covariance_requires_topology_and_matches_uncentered_lags():
    rng=np.random.default_rng(1130)
    x=rng.normal(scale=.1,size=(2,40,8))+np.array([2.,1.,0.,0.,0.,0.,0.,0.])
    batch=FeatureBatch(x,200.)
    with pytest.raises(ValueError,match='topology'):
        DocumentRingRelativeCovarianceV2().fit(batch)
    family=DocumentRingRelativeCovarianceV2(ring_topology=True).fit(batch)
    observed=family.transform(batch)
    moment=x[0].T@x[0]/40
    matrix=.95*moment+.05*np.trace(moment)/8*np.eye(8)+1e-12*np.eye(8)
    matrix/=np.trace(matrix)
    values=np.array([matrix[i,(i+1)%8] for i in range(8)])
    np.testing.assert_allclose(observed[0,:5],
                               [values.mean(),np.median(values),values.std(),
                                np.quantile(values,.25),np.quantile(values,.75)],atol=1e-7)
    rotated=family.transform(FeatureBatch(np.roll(x,2,axis=2),200.))
    np.testing.assert_allclose(observed,rotated,atol=1e-7)
    centered=RingRelativeCovarianceV2().fit(batch).transform(batch)
    assert np.max(np.abs(observed-centered))>.01
    assert observed.shape==(2,20)
