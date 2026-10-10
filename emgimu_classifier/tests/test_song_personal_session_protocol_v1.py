"""Protocol primitives checked before the new native source fits."""
import numpy as np
import pytest
from benchmarks.song_real8.song_personal_session_workflow_v1 import CLASSES,select,score


def test_nested_trial_budget_reservation_and_metric_convention():
    ids=np.array([f'{c}_{i}' for c in CLASSES for i in range(9)]);y=np.repeat(CLASSES,9)
    reserved,evaluation=select(ids,y,5,'fixture')
    assert len(reserved)==20 and len(evaluation)==16 and not set(ids[reserved])&set(ids[evaluation])
    for n in (1,2,5):
        cal,_=select(ids,y,n,'fixture')
        assert len(cal)==4*n and set(cal)<=set(reserved)
    q=np.full((len(y),4),.25);result=score(y,q)
    assert result['log_loss']==pytest.approx(np.log(4)) and result['brier']==pytest.approx(.1875)
    with pytest.raises(ValueError):select(ids[:-1],y,5,'fixture')
