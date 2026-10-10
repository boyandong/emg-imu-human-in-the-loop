import ast
from pathlib import Path
from benchmarks.new_bank_v3.roam_class_transition_v1 import grid,source_metric,matching
from benchmarks.new_bank_v3.verify_roam_class_transition_v1 import interval_oracle,source_counts
from test_class_transition_bouts_v1 import fixture,detector,feed


def test_grid_is_finite_and_source_objective_is_not_successful_detection_accuracy():
    assert len(grid())==18 and len({tuple(c.items()) for c in grid()})==18
    refs=[dict(start=100,end=356,class_name='close'),dict(start=356,end=612,class_name='open')]
    events=[dict(start=100,end=356,estimated_class='close'),dict(start=700,end=956,estimated_class='close')]
    record=dict(references=refs,events=events,matches=matching(events,refs))
    actual=source_metric([record]);assert actual==source_counts([record])
    assert actual['detection_f1']==.5 and actual['stable_class_proxy_correct']==1


def test_independent_class_state_machine_matches_known_active_changes_and_confidence_censoring():
    x,ends,q=fixture()
    for config in grid():
        d=detector(**config);actual=feed(d,x,ends,q);discarded=d.discarded;censored=d.finish()
        expected,other_censored,other_discarded=interval_oracle(q,ends,config)
        assert [{k:e[k] for k in ('start','end','estimated_class','algorithmic_available_at_sample_index')} for e in actual]==expected
        assert censored==other_censored and discarded==other_discarded


def test_source_and_target_phases_have_no_classifier_training_or_target_search():
    path=Path(__file__).resolve().parents[1]/'benchmarks/new_bank_v3/roam_class_transition_v1.py'
    tree=ast.parse(path.read_text(encoding='utf8'))
    calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    assert not set(calls)&{'fit','fit_rest','fit_temperature','enroll','enroll_user','calibrate_session'}
    source=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='source_run')
    assert not any(isinstance(n,ast.Name) and n.id in ('TARGET','TARGET_RECEIPT') for n in ast.walk(source))
    target=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='target_run')
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ('grid','min','max','sorted') for n in ast.walk(target))
