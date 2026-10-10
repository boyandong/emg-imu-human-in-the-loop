"""Independent rational-mixture, availability and immutable-state checks."""
from dataclasses import replace
import pickle

import numpy as np
import pytest

from emgimu.feature_bank.available_bank_fusion_v1 import AvailableBankFusionPolicyV1
from emgimu.feature_bank.available_bank_quality_fusion_v2 import AvailableBankQualityFusionV2


def fixture():
    source = AvailableBankFusionPolicyV1(('neutral', 'fist', 'open'),
        ('local', 'temporal', 'context'), (.2, .3, .5), 4., 1.,
        'source-prespecified-probability-policy', ('source1',))
    policy = AvailableBankQualityFusionV2(source, 'source-frozen-quality', .5)
    return policy, policy.calibrate({})


def call(policy, state, probabilities, *, quality=None, ids=None, **kwargs):
    trials = tuple(f'e{i}' for i in range(len(next(iter(probabilities.values()))))) if ids is None else ids
    return policy.predict(state, probabilities, evaluation_trials=trials,
        provider_trial_ids={n:trials for n in probabilities},
        provider_classes={n:policy.source.classes for n in probabilities}, quality=quality,
        quality_trial_ids={n:trials for n in quality} if quality is not None else None,
        quality_policy_id=policy.quality_policy_id if quality is not None else None, **kwargs)


def test_rational_quality_mixture_unknown_fallback_tiny_positive_and_persistence():
    policy, state = fixture()
    probabilities = {n:np.tile(q, (5,1)) for n,q in zip(state.providers,
        ([.8,.1,.1], [.1,.8,.1], [.1,.1,.8]))}
    quality = {'local':np.array([1.,0.,.5,1.,1e-320]),
               'temporal':np.array([.5,0.,1.,0.,5e-321]),
               'context':np.array([0.,0.,1.,0.,0.])}
    before = pickle.dumps((policy, state, probabilities, quality))
    result = call(policy, state, probabilities, quality=quality)
    expected_weights = np.array([[4/7,3/7,0], [.2,.3,.5],
                                 [1/9,1/3,5/9], [1,0,0], [4/7,3/7,0]])
    expected = np.array([[.5,.4,.1], [.24,.31,.45],
                         [8/45,1/3,22/45], [.8,.1,.1], [.5,.4,.1]])
    np.testing.assert_allclose(result['effective_weights'], expected_weights, atol=1e-15, rtol=0)
    np.testing.assert_allclose(result['probabilities'], expected, atol=1e-15, rtol=0)
    assert result['rejection_reason'] == ('','all_quality_rejected','low_confidence','','')
    assert result['labels'] == ('neutral','Unknown','Unknown','neutral','neutral')
    np.testing.assert_array_equal(result['rejected'], [False,True,True,False,False])
    assert result['quality_available'] == (True,True,True) and not result['quality_unavailable']
    assert pickle.dumps((policy, state, probabilities, quality)) == before
    loaded_policy, loaded_state = pickle.loads(pickle.dumps((policy, state)))
    reloaded = call(loaded_policy, loaded_state, probabilities, quality=quality)
    np.testing.assert_array_equal(reloaded['probabilities'], result['probabilities'])
    assert loaded_policy.policy_id == policy.policy_id
    assert replace(policy, minimum_confidence=.6).policy_id != policy.policy_id


def test_missing_quality_is_explicit_and_missing_providers_renormalize_first():
    policy, state = fixture()
    probabilities = {'local':np.array([[.8,.1,.1]]), 'temporal':np.array([[.1,.8,.1]])}
    baseline = call(policy, state, probabilities)
    np.testing.assert_array_equal(baseline['probabilities'], baseline['quality_free_probabilities'])
    assert baseline['quality_unavailable'] == ('local','temporal')
    assert baseline['quality_policy_id'] is None
    result = call(policy, state, probabilities, quality={'local':[0.]})
    np.testing.assert_allclose(result['probabilities'], [[.1,.8,.1]], atol=1e-15)
    assert result['quality_available'] == (True,False)
    assert result['quality_unavailable'] == ('temporal',)
    assert result['missing_at_prediction'] == ('context',)
    # A genuinely unavailable score does not become an all-quality-zero event.
    assert result['rejection_reason'] == ('',)


def test_zero_weight_provider_cannot_rescue_all_rejected_and_confidence_equality_passes():
    policy, _ = fixture()
    state = policy.source.from_fitted_weights((1.,0.,0.), calibration_trials=('cal',))
    probabilities = {n:[[.5,.25,.25]] for n in state.providers}
    result = call(policy, state, probabilities, quality={'local':[0.],'temporal':[1.],'context':[1.]})
    assert result['rejection_reason'] == ('all_quality_rejected',)
    np.testing.assert_array_equal(result['effective_weights'], [[1.,0.,0.]])
    accepted = call(policy, state, probabilities)
    assert accepted['rejection_reason'] == ('',) and accepted['labels'] == ('neutral',)


@pytest.mark.parametrize('kind', ['negative','above_one','nan','wrong_shape','trial_order',
    'quality_identity','missing_axis','unknown_provider','absent_provider','source_overlap',
    'calibration_overlap','class_order','bad_probability'])
def test_quality_and_prediction_axes_and_leakage_are_rejected(kind):
    policy, _ = fixture()
    state = policy.source.from_fitted_weights((.2,.3,.5), calibration_trials=('cal',))
    values = {n:[[.5,.25,.25]] for n in state.providers}
    args = dict(evaluation_trials=('e',), provider_trial_ids={n:('e',) for n in values},
        provider_classes={n:policy.source.classes for n in values},
        quality={'local':[.5]},quality_trial_ids={'local':('e',)},quality_policy_id=policy.quality_policy_id)
    if kind in ('negative','above_one','nan','wrong_shape'):
        args['quality']['local'] = {'negative':[-.1],'above_one':[1.1],'nan':[np.nan], 'wrong_shape':[[.5]]}[kind]
    elif kind == 'trial_order': args['quality_trial_ids']['local'] = ('other',)
    elif kind == 'quality_identity': args['quality_policy_id'] = 'target-tuned-other-policy'
    elif kind == 'missing_axis': args['quality_trial_ids'] = {}
    elif kind == 'unknown_provider': args['quality']['unknown'] = [.5]
    elif kind == 'absent_provider':
        del values['local'];del args['provider_trial_ids']['local'];del args['provider_classes']['local']
    elif kind in ('source_overlap','calibration_overlap'):
        args['evaluation_trials'] = ('source1' if kind == 'source_overlap' else 'cal',)
    elif kind == 'class_order': args['provider_classes']['local'] = tuple(reversed(policy.source.classes))
    else: values['local'] = [[-.1,.6,.5]]
    before = pickle.dumps((policy,state))
    with pytest.raises(ValueError):policy.predict(state, values, **args)
    assert pickle.dumps((policy,state)) == before


@pytest.mark.parametrize('kwargs', [dict(quality_policy_id=''),dict(minimum_confidence=-.1),
    dict(minimum_confidence=float('nan')),dict(minimum_confidence=1.1),dict(unknown_label='fist')])
def test_invalid_prespecified_policy_rejected(kwargs):
    policy, _ = fixture()
    with pytest.raises(ValueError):replace(policy, **kwargs)


def test_arbitrary_ten_provider_names_and_calibration_skip_use_unchanged_source_contract():
    names = tuple(f'F{i}' for i in range(10))
    source = AvailableBankFusionPolicyV1((0,1), names, (.1,)*10, 4.,1., 'ten-provider-policy',('source',))
    policy = AvailableBankQualityFusionV2(source,'ten-provider-quality')
    cal={'F0':(np.array([[0.],[1.]]),np.array([0,1]),('cal0','cal1'))}
    state=policy.calibrate(cal, forbidden_evaluation_trials=('e0',))
    assert state.providers == ('F0',) and state.n_cal_trials == 2
    result = call(policy,state,{'F0':[[.2,.8]]},quality={'F0':[.5]})
    assert result['labels'] == (1,) and result['missing_at_calibration'] == names[1:]
    population=policy.calibrate({})
    values={n:np.array([[i/10,1-i/10]]) for i,n in enumerate(names)}
    result=call(policy,population,values,quality={n:[1.] for n in names})
    np.testing.assert_allclose(result['probabilities'],[[.45,.55]],atol=1e-15)
