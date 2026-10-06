import pickle
import numpy as np
import pytest
from emgimu.feature_bank.trial_mahalanobis_v1 import TrialMahalanobisAnchorV1


def test_windows_do_not_inflate_independent_trial_budget_and_failure_is_atomic():
    rng = np.random.default_rng(719)
    x = rng.normal(size=(8, 2)); y = np.repeat([0, 1], 4)
    ids = [f't{i}' for i in range(8)]
    m = TrialMahalanobisAnchorV1(.25).fit(x, y, trial_ids=ids)
    before = pickle.dumps(m)
    # Hundreds of windows from one recording remain only one calibration trial.
    with pytest.raises(ValueError, match='independent trials'):
        m.fit(np.repeat(x[[0, 4]], 100, axis=0), np.repeat([0, 1], 100),
              trial_ids=['one']*100+['two']*100)
    assert pickle.dumps(m) == before
    with pytest.raises(ValueError, match='overlaps'):
        m.transform(x[:1], trial_ids=['t0'])
    assert pickle.dumps(m) == before


def test_equal_trial_mean_covariance_matches_independent_quadratic_form():
    rng = np.random.default_rng(733)
    trials = rng.normal(size=(12, 3)); labels = np.repeat([0, 1], 6)
    copies = np.arange(1, 13)
    x = np.repeat(trials, copies, axis=0); y = np.repeat(labels, copies)
    ids = np.repeat([f't{i}' for i in range(12)], copies)
    m = TrialMahalanobisAnchorV1(.2).fit(x, y, trial_ids=ids)
    query = np.array([[.7, -.2, 1.1]])
    expected = []
    for group in [trials[:6], trials[6:]]:
        cov = np.cov(group, rowvar=False); iso = np.trace(cov)/3
        reg = .8*cov+.2*iso*np.eye(3)+max(iso*1e-8, 1e-10)*np.eye(3)
        delta = query[0]-group.mean(axis=0)
        expected.append(np.sqrt(delta @ np.linalg.solve(reg, delta)))
    np.testing.assert_allclose(m.transform(query, trial_ids=['evaluation'])[0, :2], expected, rtol=1e-6)
    assert m.trial_counts_ == {'0': 6, '1': 6}
