import ast
from pathlib import Path
import pytest
from benchmarks.new_bank_v3.roam_native_continuous_v1 import matching,metrics,ARMS


def fixture():
    refs=[dict(start=100,end=500,class_name='open'),dict(start=600,end=1000,class_name='close')]
    events=[dict(start=100,end=500,labels={a:'open' for a in ARMS},probabilities={a:[.1,.8,.1] for a in ARMS}),
            dict(start=1200,end=1600,labels={a:'open' for a in ARMS},probabilities={a:[.1,.8,.1] for a in ARMS})]
    return dict(references=refs,events=events,matches=matching(events,refs),censored_end=False)


def test_end_to_end_retains_missed_references_and_unmatched_false_detections():
    r=fixture();m=metrics([r],'joint_full')
    assert m['references']==m['detections']==2 and m['matched']==m['missed']==m['unmatched_detections']==1
    assert m['end_to_end_success']==.5 and m['conditional_accuracy']==1.
    assert m['active_recall']=={'close':0.,'open':1.} and m['active_event_macro_f1']==pytest.approx(1/3)
    assert r['matches'][0]['iou']==1. and r['matches'][0]['onset_error_s']==0.


def test_empty_detections_do_not_drop_references_or_invent_probability_scores():
    r=fixture();r.update(events=[],matches=[],censored_end=True)
    m=metrics([r],'source_window')
    assert m['missed']==2 and m['end_to_end_success']==0 and m['active_event_macro_f1']==0
    assert m['conditional_accuracy'] is None and m['conditional_log_loss'] is None and m['conditional_brier'] is None
    assert m['eof_censored_recordings']==1


def test_matching_is_prediction_independent_one_to_one_with_fixed_iou_cutoff():
    r=fixture();events=[dict(start=90,end=510),dict(start=100,end=500),dict(start=490,end=610)]
    pairs=matching(events,r['references'])
    assert [(p['event_index'],p['reference_index']) for p in pairs]==[(1,0)]


def test_native_runner_has_no_training_or_enrollment_calls():
    root=Path(__file__).resolve().parents[1]
    tree=ast.parse((root/'benchmarks/new_bank_v3/roam_native_continuous_v1.py').read_text())
    banned={'fit','fit_rest','enroll','enroll_user','calibrate_session','fit_temperature'}
    assert not [n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in banned]


def test_independent_metric_oracle_keeps_the_same_full_reference_denominator():
    from benchmarks.new_bank_v3.verify_roam_native_continuous_v1 import independent_metrics,close
    r=fixture();close(metrics([r],'joint_full'),independent_metrics([r],'joint_full'))
    r.update(events=[],matches=[])
    close(metrics([r],'source_window'),independent_metrics([r],'source_window'))
