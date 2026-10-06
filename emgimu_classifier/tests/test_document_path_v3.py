import pickle
import numpy as np
import pytest
from emgimu.feature_bank.temporal import CompleteSequenceBatch
from emgimu.feature_bank.document_path_v3 import (
    document_envelope_path,document_dtw_distance,DocumentTemporalTemplatesV3,DocumentPathSignatureV3)


def batch(x):
    x=np.asarray(x,dtype=float)
    return CompleteSequenceBatch(x,32.,durations_seconds=np.ones(len(x)),full_coverage=True)


def test_near_zero_additive_normalization_and_polygon_signature():
    raw=np.array([[1e-10,0.],[0.,1e-10],[1e-10,1e-10]])
    expected=np.array([[.5,0.],[0.,.5],[1/(np.sqrt(2)+1)]*2])
    np.testing.assert_allclose(document_envelope_path(raw),expected,atol=1e-15)
    family=DocumentPathSignatureV3().fit(batch(raw[None]));frozen=pickle.dumps(family)
    first=expected[1]-expected[0];second=expected[2]-expected[1]
    signature=np.r_[first+second,(np.outer(first,first)/2+np.outer(first,second)+np.outer(second,second)/2).ravel()]
    np.testing.assert_allclose(family.transform(batch(raw[None]))[0],signature,atol=1e-15)
    assert len(family.feature_names)==2+2**2
    assert pickle.dumps(family)==frozen


def test_dtw_matches_exhaustive_small_path_cost_and_length():
    a=np.array([[0.,0.],[1.,0.],[1.,2.]])
    b=np.array([[0.,1.],[1.,1.],[1.,2.]])
    possible=[]
    def visit(i,j,cost,length):
        cost+=np.sqrt(np.sum((a[i]-b[j])**2));length+=1
        if i==2 and j==2:possible.append((cost,length));return
        for di,dj in ((1,0),(0,1),(1,1)):
            ni,nj=i+di,j+dj
            if ni<=2 and nj<=2 and abs(ni-nj)<=1:visit(ni,nj,cost,length)
    visit(0,0,0.,0)
    cost,length=min(possible)
    assert document_dtw_distance(a,b,1)==pytest.approx(cost/length)
    assert document_dtw_distance(b,a,1)==pytest.approx(cost/length)
    assert document_dtw_distance(a,b,0)==pytest.approx(2/3)
    with pytest.raises(ValueError,match='endpoint'):document_dtw_distance(a,b[:2],0)
    with pytest.raises(ValueError,match='Finite'):document_dtw_distance(a,np.full_like(a,np.nan),1)


def test_medoid_trial_provenance_and_frozen_warp_policy():
    vectors=np.array([[1.,0.],[1.,1.],[0.,1.],[2.,0.]])
    paths=np.repeat(vectors[:,None,:],4,axis=1);ids=['a','middle','b','other-class']
    family=DocumentTemporalTemplatesV3(.25).fit(batch(paths),[0,0,0,1],trial_ids=ids)
    assert family.medoid_trial_ids_==['middle','other-class']
    frozen=pickle.dumps(family)
    query=batch(paths[:1])
    result=family.transform(query,trial_ids=['new'])
    assert result[0,0]==pytest.approx(np.sqrt(2-np.sqrt(2)),abs=1e-9)
    assert pickle.dumps(family)==frozen
    family.band_fraction=1.
    np.testing.assert_array_equal(family.transform(query,trial_ids=['new']),result)
    with pytest.raises(ValueError,match='overlap'):family.transform(query,trial_ids=['a'])
    with pytest.raises(ValueError,match='Duplicate'):
        DocumentTemporalTemplatesV3().fit(batch(paths),[0,0,0,1],trial_ids=['a']*4)
    with pytest.raises(ValueError,match='one-dimensional'):
        DocumentTemporalTemplatesV3().fit(batch(paths),[[0],[0],[0],[1]],trial_ids=ids)
    with pytest.raises(ValueError,match='nonnegative'):document_envelope_path(-paths[0])


def test_signature_is_unchanged_by_duplicate_vertices_and_needs_full_sequences():
    points=np.array([[1.,0.],[0.,1.],[1.,1.]])
    family=DocumentPathSignatureV3().fit(batch(points[None]))
    original=family.transform(batch(points[None]))
    np.testing.assert_allclose(family.transform(batch(np.repeat(points,2,axis=0)[None])),original,atol=1e-15)
    from emgimu.feature_bank.core import FeatureBatch
    with pytest.raises(ValueError,match='complete sequences'):family.transform(FeatureBatch(points[None],200.))
