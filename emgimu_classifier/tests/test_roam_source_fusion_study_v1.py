"""Unknown fallback is scored wrong; source and target stages stay separate."""
import ast
from pathlib import Path
import numpy as np
import pytest
from benchmarks.new_bank_v3.roam_source_fusion_v1 import score,CLASSES
from benchmarks.new_bank_v3.verify_roam_source_fusion_v1 import metric


def test_unknown_probability_fallback_does_not_gain_classification_credit():
    q=np.full((3,3),1/3);pred=np.array(['Unknown']*3)
    actual=score(CLASSES,q,pred);oracle=metric(CLASSES,q,np.ones(3,bool))
    assert actual['accuracy']==0 and actual['macro_f1']==0 and actual['unknown_trials']==3
    assert actual['log_loss']==pytest.approx(np.log(3)) and actual['brier']==pytest.approx(2/9)
    for k in actual:
        if isinstance(actual[k],float):assert actual[k]==pytest.approx(oracle[k],abs=1e-14)
        else:assert actual[k]==oracle[k]


def test_target_application_has_no_fitting_and_requires_committed_source_policy():
    path=Path(__file__).resolve().parents[1]/'benchmarks/new_bank_v3/roam_source_fusion_v1.py'
    tree=ast.parse(path.read_text(encoding='utf8'));functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    target=functions['target_run'];calls=[n for n in ast.walk(target) if isinstance(n,ast.Call)]
    assert not any(isinstance(n.func,ast.Attribute) and n.func.attr in ('fit','enroll','calibrate_session') for n in calls)
    check=ast.get_source_segment(path.read_text(encoding='utf8'),functions['check'])
    assert 'committed(SOURCE)' in check and "committed(OUT/'source/policy.json')" in check
    source=ast.get_source_segment(path.read_text(encoding='utf8'),functions['source_run'])
    assert 'np.load' not in source and 'OLD.read_text' not in source and 'source_data()' in source
