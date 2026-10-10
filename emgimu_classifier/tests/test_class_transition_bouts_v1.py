"""Hand-built class changes with equal active energy need no Rest boundary."""
import numpy as np
import pytest
from emgimu.feature_bank.class_transition_bouts_v1 import ConfirmedClassBoutDetectorV1


def fixture():
    ends=np.arange(40,813,8,dtype=np.int64);centers=ends-20
    labels=np.where(centers<100,2,np.where(centers<356,0,np.where(centers<612,1,2)))
    q=np.eye(3)[labels];x=np.zeros((812,8),np.float32);x[100:612]=1.
    return x,ends,q


def detector(**kwargs):
    return ConfirmedClassBoutDetectorV1(bank_id='source',class_names=('close','open','relax'),
        rest_label='relax',sample_rate_hz=200.,**kwargs)


def feed(d,x,ends,q,start=0):
    return d.feed(x,start,recording_id='query',probabilities=q,window_end_indices=ends,bank_id='source')


def test_active_to_active_boundaries_keep_native_samples_and_causal_availability():
    x,ends,q=fixture();d=detector();events=feed(d,x,ends,q)
    assert [(e['start'],e['end'],e['estimated_class']) for e in events]==[(100,356,'close'),(356,612,'open')]
    for e in events:
        np.testing.assert_array_equal(e['emg'],x[e['start']:e['end']])
        assert e['algorithmic_available_at_sample_index']==e['end']+36
        assert e['returned_after_sample_index']==len(x) and e['boundary_kind']=='estimated'
    assert not d.finish()


def test_irregular_chunking_has_same_boundaries_samples_and_algorithmic_time():
    x,ends,q=fixture();d=detector();out=[];pos=0
    for stop in (7,19,101,371,619,len(x)):
        mask=(ends>pos)&(ends<=stop);out+=feed(d,x[pos:stop],ends[mask],q[mask],pos);pos=stop
    assert [(e['start'],e['end']) for e in out]==[(100,356),(356,612)]
    assert [e['algorithmic_available_at_sample_index'] for e in out]==[392,648]
    assert not d.finish()


def test_initial_active_eof_unknown_and_gaps_cannot_invent_complete_bouts():
    x,ends,q=fixture();q[:]=[1,0,0];d=detector()
    assert feed(d,x,ends,q)==[] and d.finish()
    x,ends,q=fixture();q[(ends>=340)&(ends<500)]=1/3
    d=detector(confidence=.7);out=feed(d,x,ends,q)
    assert not any(e['estimated_class']=='close' for e in out) and d.discarded>=1
    d=detector();feed(d,x[:100],ends[ends<=100],q[ends<=100])
    with pytest.raises(ValueError,match='Gap'):feed(d,x[100:200],ends[(ends>100)&(ends<=200)],q[(ends>100)&(ends<=200)],101)
    assert d.next_sample is None


@pytest.mark.parametrize('mutation',['omit','future','axis','bank'])
def test_invalid_source_axes_or_noncausal_windows_clear_state(mutation):
    x,ends,q=fixture();d=detector();bank='source'
    if mutation=='omit':ends=ends[1:];q=q[1:]
    elif mutation=='future':ends=ends.copy();ends[-1]+=8
    elif mutation=='axis':q=q[:,:2]
    else:bank='different'
    with pytest.raises(ValueError):d.feed(x,0,recording_id='query',probabilities=q,window_end_indices=ends,bank_id=bank)
    assert d.next_sample is None and d.stable is None


def test_overlong_capture_is_discarded_and_memory_is_bounded():
    d=detector(max_duration_s=2.);x=np.ones((1000,8),np.float32)
    ends=np.arange(40,1001,8,dtype=np.int64);q=np.tile([1.,0,0],(len(ends),1));q[ends<100]=[0,0,1]
    assert feed(d,x,ends,q)==[] and not d.known_start
    assert len(d.buffer)<=d.maximum+d.size+d.hop*(d.confirmations+d.smoothing)
