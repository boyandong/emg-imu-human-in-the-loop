import ast
from pathlib import Path
from benchmarks.new_bank_v3.roam_document_reliability_v3 import grid
import numpy as np
from benchmarks.new_bank_v3.roam_native_joint_v1 import CLASSES,GROUPS
from benchmarks.new_bank_v3.verify_roam_document_reliability_v3 import hierarchy
from emgimu.feature_bank.native_document_reliability_v3 import hierarchical_weights


def test_reliability_grid_has_fixed_source_only_temperature_and_trial_shrinkage_candidates():
    assert len(grid())==16 and len({tuple(v.items()) for v in grid()})==16
    assert {v['n0'] for v in grid()}=={1.,4.,16.,64.}
    assert {v['temperature'] for v in grid()}=={.25,.5,1.,2.}


def test_source_policy_phase_cannot_read_target_and_target_cannot_select_hyperparameters():
    path=Path(__file__).resolve().parents[1]/'benchmarks/new_bank_v3/roam_document_reliability_v3.py'
    tree=ast.parse(path.read_text(encoding='utf8'))
    calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    assert not set(calls)&{'fit','enroll','enroll_user','calibrate_session'}
    source=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='source_run')
    assert not any(isinstance(n,ast.Name) and n.id in ('TARGET','TARGET_RECEIPT') for n in ast.walk(source))
    target=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='target_run')
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ('grid','candidate','min','max') for n in ast.walk(target))


def test_independent_three_class_hierarchy_matches_every_precommitted_source_candidate():
    rng=np.random.default_rng(1928)
    long={g:(rng.normal(size=(6,i+2)),np.repeat(CLASSES,2),tuple(f'long{j}' for j in range(6))) for i,g in enumerate(GROUPS)}
    current={g:(rng.normal(size=(3,i+2)),np.asarray(CLASSES),tuple(f'current{j}' for j in range(3))) for i,g in enumerate(GROUPS)}
    population=np.arange(1.,8.);population/=population.sum()
    for c in grid():
        actual,_,_=hierarchical_weights(CLASSES,GROUPS,population,c['n0'],c['temperature'],long,current,source_policy_id='fixture')
        expected=hierarchy(long,current,population,c['n0'],c['temperature'])
        np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=0)
