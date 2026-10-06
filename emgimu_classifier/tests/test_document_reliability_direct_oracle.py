import pickle
import numpy as np
from emgimu.feature_bank.document_reliability_v2 import DocumentReliabilityWeightsV2


def test_three_class_nonzero_reliability_temperature_and_population_shrinkage():
    # Triangle inter-prototype distances3,4,5; six trial radii are all1.
    triangle = np.array([[-1,0],[1,0],[2,0],[4,0],[-1,4],[1,4]], dtype=float)
    # Collinear prototype distances2,3,5; six trial radii are all0.5.
    line = np.array([[-.5],[.5],[1.5],[2.5],[4.5],[5.5]])
    labels = np.repeat([0,1,2],2); ids = [f'cal-{i}' for i in range(6)]
    policy = DocumentReliabilityWeightsV2((0,1,2), ('triangle','line'), (.8,.2),
        n0=3., temperature=2., source_policy_id='frozen-source-policy-fixture')
    calibration = {'triangle':(triangle,labels,ids), 'line':(line,labels,ids)}
    frozen = pickle.dumps((policy,calibration))
    result = policy.calculate(calibration, forbidden_trial_ids=['source','eval'])
    eps = 1e-10
    expected_ratio = np.array([4/(1+eps), (10/3)/(.5+eps)])
    expected_personal = np.sqrt(expected_ratio+eps)
    expected_personal /= expected_personal.sum()
    np.testing.assert_allclose(result['between'], [4.,10/3], rtol=0, atol=1e-14)
    np.testing.assert_allclose(result['within'], [1.,.5], rtol=0, atol=1e-14)
    np.testing.assert_allclose(result['reliability'], expected_ratio, rtol=0, atol=1e-14)
    np.testing.assert_allclose(result['log_reliability'], np.log(expected_ratio+eps), atol=1e-14)
    np.testing.assert_allclose(result['personal'], expected_personal, atol=1e-14)
    assert result['n_cal_trials'] == 6 and result['alpha'] == 1/3
    np.testing.assert_allclose(result['final'], np.array([.8,.2])/3+expected_personal*2/3, atol=1e-14)
    assert pickle.dumps((policy,calibration)) == frozen
    # Unequal duplicate-window counts cannot turn six trials into21 trials,
    # change prototype mass, or weaken source population shrinkage.
    multiplicity = np.arange(1,7)
    repeated = {name:(np.repeat(x,multiplicity,axis=0), np.repeat(y,multiplicity),
                     np.repeat(trials,multiplicity).tolist()) for name,(x,y,trials) in calibration.items()}
    duplicate = policy.calculate(repeated)
    assert duplicate['n_cal_trials'] == 6 and duplicate['alpha'] == 1/3
    for key in ('between','within','reliability','personal','final'):
        np.testing.assert_allclose(duplicate[key], result[key], atol=1e-14)
