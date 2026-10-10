import ast
from pathlib import Path
import numpy as np
import pytest
from benchmarks.new_bank_v3.roam_precision_transition_v2 import grid,summarize,select,detect
from benchmarks.new_bank_v3.verify_roam_precision_transition_v2 import scores,same
from benchmarks.new_bank_v3.verify_roam_class_transition_v1 import interval_oracle


def user(matched=8,extra=4):
    return dict(user=1,references=10,detections=matched+extra,matched=matched,
                unmatched_detections=extra,detection_f1=2*matched/(10+matched+extra),
                stable_class_proxy_macro_f1=.5,stable_class_proxy_correct=matched)


def test_precision_objective_keeps_misses_and_false_events_and_uses_equal_users():
    users=[user(),dict(user(matched=2,extra=0),user=2)]
    a=summarize(0,grid()[0],users);b=scores(0,grid()[0],users);same(a,b)
    assert a['mean_detection_recall']==.5
    assert a['mean_detection_fbeta_half']==pytest.approx((10/14.5+2.5/4.5)/2)
    assert a['total_unmatched_detections']==4


def test_selector_prohibits_source_recall_collapse_and_more_false_events():
    baseline=summarize(0,grid()[0],[user()])
    low_recall=summarize(1,grid()[1],[user(matched=4,extra=0)])
    more_false=summarize(2,grid()[2],[user(matched=10,extra=5)])
    better=summarize(3,grid()[3],[user(matched=8,extra=1)])
    assert select([baseline,low_recall,more_false,better],baseline)==better
    assert select([baseline,low_recall,more_false],baseline)==baseline
    with pytest.raises(ValueError):select([low_recall,more_false],baseline)


def test_frozen_grid_baseline_and_tie_order_are_deterministic():
    assert len(grid())==36 and len({tuple(c.items()) for c in grid()})==36
    assert grid()[0]==dict(confirmations=5,smoothing_windows=1,confidence=0.)
    a=summarize(0,grid()[0],[user()]);b=dict(a,index=1)
    assert select([b,a],a)==a


def test_longer_confirmation_changes_availability_without_inventing_reference_edges():
    x=np.zeros((3000,8),np.float32);ends=np.arange(40,len(x)+1,8);centers=ends-20
    labels=np.where(centers<400,2,np.where(centers<1200,0,np.where(centers<2000,1,2)))
    q=np.eye(3)[labels]
    for config in grid():
        actual,censored,discarded=detect(x,ends,q,'bank','query',config)
        expected,ec,ed=interval_oracle(q,ends,config)
        assert [{k:e[k] for k in ('start','end','estimated_class','algorithmic_available_at_sample_index')} for e in actual]==expected
        assert (censored,discarded)==(ec,ed)
    events,_,_=detect(x,ends,q,'bank','query',dict(confirmations=20,smoothing_windows=1,confidence=0.))
    assert [(e['start'],e['end']) for e in events]==[(404,1204),(1204,2004)]
    assert all(e['algorithmic_available_at_sample_index']==e['end']+172 for e in events)


def test_new_policy_phases_do_not_retrain_or_search_target_parameters():
    path=Path(__file__).resolve().parents[1]/'benchmarks/new_bank_v3/roam_precision_transition_v2.py'
    tree=ast.parse(path.read_text(encoding='utf8'))
    calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    assert not set(calls)&{'fit','fit_rest','fit_temperature','enroll','enroll_user','calibrate_session'}
    source=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='source_run')
    assert not any(isinstance(n,ast.Name) and n.id in ('TARGET','TARGET_RECEIPT','ARCHIVE') for n in ast.walk(source))
    target=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='target_run')
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ('grid','select','sorted') for n in ast.walk(target))
