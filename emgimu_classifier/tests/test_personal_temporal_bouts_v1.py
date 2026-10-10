"""Native coverage, independent medoids/signatures and eight-channel composition."""
from dataclasses import replace
import pickle
import numpy as np
import pytest
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.personal_temporal_bouts_v1 import TemporalBoutBatchV1,PersonalTemporalBoutsV1
from emgimu.feature_bank.complete_bout_window_adapter_v1 import CompleteBoutWindowAdapterV1


def batch(prefix,channels=8,classes=('fist','index_pinch','neutral','open_hand'),shots=2):
    rng = np.random.default_rng(len(prefix)+channels)
    ids = tuple(f'{prefix}:{c}:{j}' for c in classes for j in range(shots))
    x = []
    for i,c in enumerate(classes):
        for j in range(shots):
            n = 250+13*j
            v = rng.normal(size=(n,channels))*(.3+np.roll(np.arange(1,channels+1),i))[None]
            v *= (1+.5*np.sin(np.linspace(0,2*np.pi,n)))[:,None]
            x.append(v.astype(np.float32))
    b = TemporalBoutBatchV1(tuple(x),ids,tuple(prefix for _ in ids),tuple(0 for _ in ids),250.,
        tuple(f'CH{i+1}' for i in range(channels)),'fixture','complete_cued')
    return b,{t:c for t,c in zip(ids,np.repeat(classes,shots))}


def exhaustive(a,b,band):
    # Independent enumeration on tiny paths, including shortest equal-cost tie.
    def visit(i,j,cost,length):
        cost += np.linalg.norm(a[i]-b[j]);length += 1
        if i == len(a)-1 and j == len(b)-1:return [(cost,length)]
        result=[]
        for u,v in ((i+1,j+1),(i+1,j),(i,j+1)):
            if u < len(a) and v < len(b) and abs(u-v) <= band:result.extend(visit(u,v,cost,length))
        return result
    cost,length = min(visit(0,0,0.,0),key=lambda v:(v[0],v[1]))
    return cost/length


def test_complete_bout_medoids_geometry_signatures_and_persistence(tmp_path):
    from emgimu.feature_bank.document_path_v3 import document_dtw_distance
    a = np.array([[1.,0.],[.5,.5],[0.,1.]])
    b = np.array([[1.,0.],[.7,.3],[0.,1.]])
    assert document_dtw_distance(a,b,1) == exhaustive(a,b,1)
    cal,labels = batch('long')
    w = PersonalTemporalBoutsV1(source_bank_id='fixture',sample_rate_hz=250.,channel_ids=cal.channel_ids,
        preprocessing_id='fixture',class_names=('fist','index_pinch','neutral','open_hand'))
    p = w.enroll(cal,labels,user_id='u',session_id='long')
    envelope = np.stack([np.stack([np.sqrt((part.astype(float)**2).mean(0)) for part in np.array_split(x,32)]) for x in cal.sequences])
    paths = envelope/(np.linalg.norm(envelope,axis=2,keepdims=True)+1e-10)
    for i,c in enumerate(w.class_names):
        positions = [j for j,t in enumerate(cal.trial_ids) if labels[t] == c]
        distances = [[document_dtw_distance(paths[a],paths[b],3) for b in positions] for a in positions]
        chosen = positions[np.argmin(np.sum(distances,axis=1))]
        assert p.medoid_trial_ids[i] == cal.trial_ids[chosen]
        np.testing.assert_array_equal(p.templates[i],paths[chosen])
    current,local_labels = batch('current')
    s = w.enroll(current,local_labels,user_id='u',session_id='current',personal=p)
    held,_ = batch('held',shots=1)
    before = pickle.dumps((w,p,s))
    r = w.predict(held,personal=p,session=s,user_id='u',session_id='current',
        base_probabilities=np.full((4,4),.25),base_trial_ids=held.trial_ids,base_class_names=w.class_names)
    z = held.paths();z /= np.linalg.norm(z,axis=2,keepdims=True)+1e-10
    signatures=[]
    for path in z:
        first=np.zeros(8);second=np.zeros((8,8))
        for change in np.diff(path,axis=0):
            second += np.outer(first,change)+.5*np.outer(change,change);first += change
        signatures.append(np.r_[first,second.ravel()])
    np.testing.assert_allclose(r['signature_coordinates'],signatures,rtol=0,atol=1e-14)
    np.testing.assert_allclose(r['arms']['base_full'],.75*.25+.125*(r['arms']['DTW_blended']+r['arms']['signature_blended']),rtol=0,atol=1e-15)
    assert pickle.dumps((w,p,s)) == before
    path = tmp_path/'personal.zip';w.save_profile(p,path)
    assert w.load_profile(path,user_id='u').profile_id == p.profile_id
    with pytest.raises(FileExistsError):w.save_profile(p,path)
    with pytest.raises(ValueError,match='identity'):w.load_profile(path,user_id='other')
    with pytest.raises(ValueError,match='overlaps'):w.predict(cal,personal=p,user_id='u',session_id='held')
    with pytest.raises(ValueError,match='recording'):w.predict(replace(held,recording_ids=cal.recording_ids[:4]),personal=p,user_id='u',session_id='held')
    with pytest.raises(ValueError,match='Complete'):w.enroll(replace(cal,boundary_kind='estimated'),labels,user_id='u',session_id='long')
    with pytest.raises(ValueError,match='window'):w.predict(FeatureBatch(np.zeros((4,50,8)),250.),personal=p,user_id='u',session_id='held')
    with pytest.raises(ValueError,match='axes'):w.predict(held,personal=p,user_id='u',session_id='held',base_probabilities=np.full((4,4),.25),base_trial_ids=held.trial_ids[::-1],base_class_names=w.class_names)
    estimated=w.predict(replace(held,boundary_kind='estimated'),personal=p,user_id='u',session_id='held')
    assert not estimated['certified_full_coverage'] and estimated['boundary_kind']=='estimated'
    assert not any(r[k] for k in ('default_promoted','physical_validation_proven'))


def test_actual_eight_channel_window_composition_and_raw_rejection():
    from test_personal_session_stream_v4 import workflow
    w = workflow()
    cal,labels = batch('long',classes=tuple(w.bank.classes_))
    cal = replace(cal,preprocessing_id=w.preprocessing_id)
    t = PersonalTemporalBoutsV1(source_bank_id=w.bank.bank_id_,sample_rate_hz=250.,
        channel_ids=w.channels,preprocessing_id=w.preprocessing_id,class_names=tuple(w.bank.classes_))
    p = t.enroll(cal,labels,user_id='fixture',session_id='long')
    adapter = CompleteBoutWindowAdapterV1(w,t)
    held,_ = batch('held',classes=tuple(w.bank.classes_));held = replace(held,preprocessing_id=w.preprocessing_id)
    raw = tuple(x.copy() for x in held.sequences)
    for x in raw:x[:,2]=0
    raw = replace(held,sequences=raw,preprocessing_id='raw_pre_software_highpass')
    before = pickle.dumps((w,t,p))
    r = adapter.predict(held,temporal_personal=p,user_id='fixture',session_id='current',
        raw_batch=raw,quality_mode='structural')
    assert r['rejected'].all() and set(r['predicted_labels'])=={'Unknown'}
    assert r['window_unrepresented_tail_samples']==[0,3]*4
    assert r['temporal_uses_all_native_samples'] and r['decision_available_after_interval']
    assert pickle.dumps((w,t,p)) == before
    with pytest.raises(ValueError,match='axes'):adapter.predict(held,temporal_personal=p,user_id='fixture',session_id='current',raw_batch=replace(raw,starts=tuple(1 for _ in raw.starts)))
