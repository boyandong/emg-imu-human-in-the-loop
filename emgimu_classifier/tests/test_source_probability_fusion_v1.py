"""Convex source objective, held-out axes, availability and immutable policy."""
from dataclasses import replace
import json
import pickle
import numpy as np
import pytest
from emgimu.feature_bank.source_probability_fusion_v1 import SourceProbabilityFusionV1


def fit_fixture():
    p={'a':np.array([[.9,.1],[.8,.2]]),'b':np.array([[.2,.8],[.1,.9]])}
    kwargs=dict(classes=('a','b'),providers=('a','b'),trial_ids=('source:1','source:2'),
        scenario_ids=('zero','zero'),user_ids=('one','two'),source_protocol_id='frozen',inference_bank_id='bank')
    return p,kwargs


def test_convex_policy_has_independently_known_optimum_and_manifest_roundtrip():
    p,kwargs=fit_fixture();policy,d=SourceProbabilityFusionV1.fit(p,('a','b'),**kwargs)
    np.testing.assert_allclose(policy.weights,[.5,.5],rtol=0,atol=1e-8)
    assert d['source_log_loss']==pytest.approx(-np.log(.55)) and d['simplex_stationarity_gap']<1e-6
    assert not d['target_labels_used'] and not d['source_training_loss_is_unbiased_evaluation']
    loaded=SourceProbabilityFusionV1.from_manifest(json.loads(json.dumps(policy.manifest())))
    assert loaded==policy and loaded.policy_id==policy.policy_id
    bad=policy.manifest();bad['weights']=[1.,0.]
    with pytest.raises(ValueError,match='identity'):SourceProbabilityFusionV1.from_manifest(bad)


def test_equal_user_weighting_does_not_count_repeated_budget_as_independent_trial():
    p,kwargs=fit_fixture();a,da=SourceProbabilityFusionV1.fit(p,('a','b'),**kwargs)
    p={g:q[[0,0,1]] for g,q in p.items()}
    kwargs.update(trial_ids=('source:1','source:1','source:2'),scenario_ids=('zero','two','zero'),user_ids=('one','one','two'))
    b,db=SourceProbabilityFusionV1.fit(p,('a','a','b'),**kwargs)
    np.testing.assert_allclose(a.weights,b.weights,rtol=0,atol=1e-10)
    assert db['source_rows']==3 and db['independent_source_query_trials']==2 and db['equal_user_weighting']
    with pytest.raises(ValueError,match='class/user'):
        SourceProbabilityFusionV1.fit(p,('a','b','b'),**kwargs)


def test_missing_zero_weight_provider_preserves_Unknown_and_source_policy():
    p,kwargs=fit_fixture();policy,_=SourceProbabilityFusionV1.fit(p,('a','b'),**kwargs)
    policy=replace(policy,weights=(1.,0.));before=pickle.dumps(policy)
    common=dict(evaluation_trial_ids=('query:1','query:2'),inference_bank_id='bank')
    r=policy.predict({'a':p['a']},provider_trial_ids={'a':common['evaluation_trial_ids']},
        provider_classes={'a':policy.classes},**common)
    np.testing.assert_array_equal(r['probabilities'],p['a']);assert not r['rejected'].any()
    r=policy.predict({'b':p['b']},provider_trial_ids={'b':common['evaluation_trial_ids']},
        provider_classes={'b':policy.classes},**common)
    np.testing.assert_array_equal(r['probabilities'],np.full((2,2),.5))
    assert r['rejected'].all() and r['labels']==('Unknown','Unknown') and pickle.dumps(policy)==before


@pytest.mark.parametrize('failure',['overlap','axes','bank','probability','unknown','duplicate'])
def test_source_or_query_identity_errors_reject(failure):
    p,kwargs=fit_fixture()
    if failure=='overlap':
        with pytest.raises(ValueError):SourceProbabilityFusionV1.fit(p,('a','b'),forbidden_trial_ids=('source:1',),**kwargs)
        return
    policy,_=SourceProbabilityFusionV1.fit(p,('a','b'),**kwargs);ids=('query:1','query:2')
    if failure=='axes':ids=('source:1','source:2')
    if failure=='duplicate':ids=('query:1','query:1')
    if failure=='probability':p['a']=p['a']*2
    if failure=='unknown':p['other']=p.pop('b')
    with pytest.raises(ValueError):
        policy.predict(p,evaluation_trial_ids=ids,provider_trial_ids={g:ids for g in p},
            provider_classes={g:policy.classes for g in p},inference_bank_id='other' if failure=='bank' else 'bank')
