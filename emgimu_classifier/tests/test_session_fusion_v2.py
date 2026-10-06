import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from emgimu.feature_bank.session_fusion_v2 import PROVIDERS, fuse_disjoint_session_predictions


def fixture():
    return dict(probabilities=np.array([[[1.,0.],[0.,1.]], [[0.,1.],[1.,0.]],
                                       [[.5,.5],[.5,.5]], [[.25,.75],[.75,.25]]]),
        weights=np.array([.4,.3,.2,.1]), source_trials=['source'], calibration_trials=['cal'],
        evaluation_trials=['eval-a','eval-b'], provider_names=PROVIDERS,
        provider_classes=[(0,1)]*4, classes=(0,1))


def test_checked_fusion_known_weights_and_no_input_mutation():
    kw = fixture(); previous = copy.deepcopy(kw)
    actual = fuse_disjoint_session_predictions(**kw)
    np.testing.assert_allclose(actual, [[.525,.475],[.475,.525]])
    np.testing.assert_array_equal(kw['probabilities'], previous['probabilities'])
    np.testing.assert_array_equal(kw['weights'], previous['weights'])


@pytest.mark.parametrize('change', [
    {'calibration_trials':['source']}, {'evaluation_trials':['source','eval-b']},
    {'evaluation_trials':['cal','eval-b']}, {'evaluation_trials':['eval-a','eval-a']},
    {'source_trials':[]}, {'calibration_trials':['']},
    {'provider_names':tuple(reversed(PROVIDERS))},
    {'provider_classes':[(1,0),(0,1),(0,1),(0,1)]}, {'classes':(0,0)},
    {'weights':[.4,.3,.2,float('nan')]}, {'weights':[1.,1.,1.,1.]},
    {'weights':[-.1,.5,.3,.3]}, {'probabilities':np.full((4,2,2), .6)},
])
def test_checked_fusion_rejects_overlap_or_misaligned_probabilities(change):
    kw = fixture(); kw.update(change)
    with pytest.raises(ValueError):
        fuse_disjoint_session_predictions(**kw)


def test_saved_native_fusion_replay_provenance_and_coverage():
    root = Path(__file__).resolve().parents[1]
    here = root/'benchmarks/new_bank_v3'
    r = json.loads((here/'F8_CHECKED_FUSION_V2_RESULTS.json').read_text())
    parent = json.loads((here/'F8_ROUTER_MANUS_V1_RESULTS.json').read_text())
    for name, digest in r['source_sha256'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest
    expected = {(b['phase'], b['user'], b['shots']): b for b in parent['blocks']}
    assert len(r['blocks']) == len(expected) == 24
    assert {(b['phase'], b['user'], b['shots']) for b in r['blocks']} == set(expected)
    for b in r['blocks']:
        p = expected[(b['phase'], b['user'], b['shots'])]
        assert b['evaluation_trials'] == len(p['evaluation_trials'])
        assert not set(p['source_trials']) & (set(p['calibration_trials']) | set(p['evaluation_trials']))
        assert not set(p['calibration_trials']) & set(p['evaluation_trials'])
        assert b['max_probability_error'] <= 1e-12
        assert set(b['rejected']) == {'source_evaluation_overlap', 'calibration_evaluation_overlap',
                                     'class_axis_reversal', 'provider_axis_reversal'}
    assert not r['completion_proven'] and not r['default_promoted']
