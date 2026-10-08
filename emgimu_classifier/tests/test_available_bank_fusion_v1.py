from dataclasses import replace
import pickle
import numpy as np
import pytest
from emgimu.feature_bank.available_bank_fusion_v1 import AvailableBankFusionPolicyV1


def policy():
    return AvailableBankFusionPolicyV1(('A', 'B'), ('flat', 'separated', 'missing'),
                                       (.3, .3, .4), 4., 1., 'fixed-source-policy', ('source1', 'source2'))


def calibration(repeat=False):
    ids = ['a1', 'a2', 'b1', 'b2']; y = ['A', 'A', 'B', 'B']
    a = [-1., 1., -1., 1.]; b = [0., 0., 2., 2.]
    if repeat:
        ids += ['a1']; y += ['A']; a += [-1.]; b += [0.]
    return {'flat': (np.array(a)[:, None], np.array(y), ids),
            'separated': (np.array(b)[:, None], np.array(y), ids)}


def predict(p, state, values):
    return p.predict(state, values, evaluation_trials=('test1',),
                     provider_trial_ids={n: ('test1',) for n in values},
                     provider_classes={n: ('A', 'B') for n in values})


def test_calibration_missing_provider_exact_shrinkage_and_independent_trial_mass():
    p = policy(); state = p.calibrate(calibration(), forbidden_evaluation_trials=('test1',))
    # Flat means coincide and have within-class distance1. The separated
    # means differ by2 with zero within-class distance. Compute softmax directly.
    logits = np.log(np.array([0., 2. / 1e-10]) + 1e-10)
    personal = np.exp(logits - max(logits)); personal /= personal.sum()
    expected = .5 * np.array([.5, .5]) + .5 * personal
    assert state.providers == ('flat', 'separated') and state.n_cal_trials == 4 and state.alpha == .5
    np.testing.assert_allclose(state.weights, expected, atol=1e-12)
    repeated = p.calibrate(calibration(repeat=True))
    assert repeated == state
    values = {'flat': np.array([[.8, .2]]), 'separated': np.array([[.1, .9]])}
    before = pickle.dumps((p, state, values)); result = predict(p, state, values)
    np.testing.assert_allclose(result['probabilities'], expected[0] * values['flat'] + expected[1] * values['separated'])
    assert result['missing_at_calibration'] == ('missing',)
    assert result['missing_at_prediction'] == ()
    assert pickle.dumps((p, state, values)) == before
    missing = predict(p, state, {'flat': values['flat']})
    np.testing.assert_array_equal(missing['probabilities'], values['flat'])
    assert missing['weights'] == (1.,) and missing['missing_at_prediction'] == ('separated',)


def test_zero_shot_available_population_and_no_implicit_calibration():
    p = policy(); state = p.calibrate({}, available=('missing', 'flat'))
    assert state.providers == ('flat', 'missing') and state.calibration_trials == ()
    assert state.n_cal_trials == 0 and state.alpha == 1
    np.testing.assert_allclose(state.weights, [3 / 7, 4 / 7])
    assert state.mode == 'source_population_zero_shot'
    result = predict(p, state, {'flat': [[.2, .8]], 'missing': [[.7, .3]]})
    np.testing.assert_allclose(result['probabilities'], [[17 / 35, 18 / 35]])


@pytest.mark.parametrize('kind', ['source_overlap', 'test_overlap', 'different_labels', 'unknown', 'unavailable', 'incomplete_classes'])
def test_calibration_rejects_leakage_or_misaligned_inputs(kind):
    p = policy(); cal = calibration(); kwargs = {}
    if kind in ('source_overlap', 'test_overlap'):
        trial = 'source1' if kind == 'source_overlap' else 'test1'
        cal = {n: (x, y, [trial] + ids[1:]) for n, (x, y, ids) in cal.items()}
        kwargs['forbidden_evaluation_trials'] = ('test1',)
    elif kind == 'different_labels':
        x, y, ids = cal['separated']; cal['separated'] = (x, np.array(['B', 'A', 'B', 'B']), ids)
    elif kind == 'unknown': cal['unknown'] = cal['flat']
    elif kind == 'unavailable': kwargs['available'] = ('flat',)
    else: cal = {n: (x[:2], y[:2], ids[:2]) for n, (x, y, ids) in cal.items()}
    with pytest.raises(ValueError): p.calibrate(cal, **kwargs)


@pytest.mark.parametrize('kind', ['source_overlap', 'cal_overlap', 'repeated_eval', 'class_order', 'trial_order', 'bad_probability', 'unknown_provider', 'different_policy', 'state_order', 'invalid_weights', 'all_missing'])
def test_inference_rejects_invalid_axes_provenance_and_weights(kind):
    p = policy(); state = p.calibrate(calibration())
    values = {'flat': [[.2, .8]], 'separated': [[.7, .3]]}
    ids = ('test1',); axes = {n: ('A', 'B') for n in values}; trials = {n: ids for n in values}
    if kind == 'source_overlap': ids = ('source1',)
    elif kind == 'cal_overlap': ids = ('a1',)
    elif kind == 'repeated_eval': ids = ('test1', 'test1')
    elif kind == 'class_order': axes['flat'] = ('B', 'A')
    elif kind == 'trial_order': trials['flat'] = ('different',)
    elif kind == 'bad_probability': values['flat'] = [[np.nan, .2]]
    elif kind == 'unknown_provider': values['unknown'] = [[.2, .8]]
    elif kind == 'different_policy': state = replace(state, policy_id='other')
    elif kind == 'state_order': state = replace(state, providers=tuple(reversed(state.providers)))
    elif kind == 'invalid_weights': state = replace(state, weights=(-.1, 1.1))
    else: values = {}; trials = {}; axes = {}
    with pytest.raises(ValueError): p.predict(state, values, evaluation_trials=ids, provider_trial_ids=trials, provider_classes=axes)


def test_imported_policy_is_explicit_and_zero_remaining_weight_is_rejected():
    p = policy(); state = p.from_fitted_weights((1., 0.), calibration_trials=('cal1',), available=('flat', 'separated'))
    assert state.mode == 'imported_frozen_weights' and state.alpha is None
    with pytest.raises(ValueError, match='zero fitted weight'): predict(p, state, {'separated': [[.2, .8]]})
    with pytest.raises(ValueError): p.from_fitted_weights((.5, .5), calibration_trials=('source1',), available=('flat', 'separated'))
    with pytest.raises(ValueError): p.from_fitted_weights((.5, .4), calibration_trials=(), available=('flat', 'separated'))


def test_six_named_providers_are_supported_without_fixed_four_branch_restriction():
    names = tuple(f'F{i}' for i in range(6))
    p = AvailableBankFusionPolicyV1((0, 1), names, (1/6,) * 6, 4., 1., 'six-source-providers', ('s1',))
    state = p.calibrate({})
    values = {name: np.array([[i/6, 1-i/6]]) for i, name in enumerate(names)}
    result = p.predict(state, values, evaluation_trials=('e1',),
                       provider_trial_ids={n: ('e1',) for n in names},
                       provider_classes={n: (0, 1) for n in names})
    np.testing.assert_allclose(result['probabilities'], np.mean(list(values.values()), axis=0))
    assert result['provider_names'] == names and state.n_cal_trials == 0
