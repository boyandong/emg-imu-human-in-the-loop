"""Pre-freeze independent arithmetic and recording-level split checks."""
from dataclasses import replace
import numpy as np
import pytest
from benchmarks.new_bank_v3.personal_temporal_unibo_v1 import (
    CLASSES,dtw_oracle,path_and_signature,oracle_read,verify_profile,select,costs)
from emgimu.feature_bank.personal_temporal_bouts_v1 import PersonalTemporalBoutsV1
from emgimu.feature_bank.document_path_v3 import document_dtw_distance
from test_personal_temporal_bouts_v1 import batch


@pytest.mark.parametrize('channels',[4,8])
def test_independent_native_oracles_and_fixed_costs(channels):
    b,y=batch('source',channels=channels,classes=CLASSES,shots=3)
    w=PersonalTemporalBoutsV1(source_bank_id='oracle',sample_rate_hz=b.sample_rate_hz,
        channel_ids=b.channel_ids,preprocessing_id=b.preprocessing_id,class_names=CLASSES)
    p=w.enroll(b,y,user_id='u',session_id='source');verify_profile(b,y,p)
    current,cy=batch('current',channels=channels,classes=CLASSES,shots=2)
    s=w.enroll(current,cy,user_id='u',session_id='current',personal=p)
    verify_profile(current,cy,s,p)
    held,_=batch('held',channels=channels,classes=CLASSES,shots=1)
    paths,signature=path_and_signature(held)
    a,sa=oracle_read(paths,signature,p,3);c,sc=oracle_read(paths,signature,s,3)
    q=np.tile(np.array([.1,.2,.3,.4]),(4,1))
    result=w.predict(held,personal=p,session=s,user_id='u',session_id='current',
        base_probabilities=q,base_trial_ids=held.trial_ids,base_class_names=CLASSES)
    np.testing.assert_allclose(result['arms']['base_full'],.75*q+.0625*(a+c+sa+sc),rtol=0,atol=1e-12)
    np.testing.assert_allclose(result['signature_coordinates'],signature,rtol=0,atol=1e-14)
    assert abs(dtw_oracle(paths[0],paths[1],3)-document_dtw_distance(paths[0],paths[1],3))<1e-14
    assert costs('DTW_local',0)==(20,0) and costs('DTW_local',2)==(0,8)
    assert costs('base_full',5)==(20,20) and costs('base_uniform',5)==(0,0)
    with pytest.raises(ValueError,match='session identity'):
        w.predict(held,personal=p,user_id='u',session_id='')
    with pytest.raises(ValueError,match='identity'):
        w.enroll(b,y,user_id=None,session_id='source')


def test_recording_reservation_is_deterministic_nested_and_strict():
    rows=[dict(id=f'{c}_{j}_{k}',trial=f'r{c}_{j}',label=c) for c in range(4) for j in range(7) for k in range(2)]
    a=select(rows,'seed');assert a==select(rows[::-1],'seed')
    assert len(a)==20 and len({r['trial'] for r in a})==20
    held=[r for r in rows if r['trial'] not in {v['trial'] for v in a}]
    assert len(held)==16 and not {r['trial'] for r in a}&{r['trial'] for r in held}
    for n in (1,2,5):
        selected=[r for c in range(4) for r in [v for v in a if v['label']==c][:n]]
        assert len(selected)==4*n and {r['trial'] for r in selected}.isdisjoint(r['trial'] for r in held)
    with pytest.raises(ValueError,match='distinct recording'):
        select([r for r in rows if r['label']!=0 or r['trial']=='r0_0'],'seed')
