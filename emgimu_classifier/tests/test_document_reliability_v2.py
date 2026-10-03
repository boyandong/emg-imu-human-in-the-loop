import numpy as np
import pytest

from emgimu.feature_bank.document_reliability_v2 import DocumentReliabilityWeightsV2, EPS


def _policy():
    return DocumentReliabilityWeightsV2(("A", "B"), ("flat", "separated"), (.6, .4),
                                        n0=4, temperature=1, source_policy_id="frozen-source-cv-test")


def _calibration(repeat_first=False):
    ids = ["A1", "A2", "B1", "B2"]
    labels = ["A", "A", "B", "B"]
    flat = [-1., 1., -1., 1.]
    separated = [0., 0., 2., 2.]
    if repeat_first:
        ids.insert(1, "A1")
        labels.insert(1, "A")
        flat.insert(1, -1.)
        separated.insert(1, 0.)
    return {"flat": (np.asarray(flat)[:, None], np.asarray(labels), ids),
            "separated": (np.asarray(separated)[:, None], np.asarray(labels), ids)}


def test_document_exact_zero_between_and_trial_count():
    policy = _policy()
    observed = policy.calculate(_calibration())
    np.testing.assert_allclose(observed["between"], [0, 2])
    np.testing.assert_allclose(observed["within"], [1, 0])
    assert observed["reliability"][0] == 0
    assert observed["log_reliability"][0] == pytest.approx(np.log(EPS))
    assert observed["n_cal_trials"] == 4
    assert observed["alpha"] == .5
    np.testing.assert_allclose(observed["final"], .5 * np.array([.6, .4]) + .5 * observed["personal"])
    assert observed["personal"][1] > .999999999
    repeated = policy.calculate(_calibration(repeat_first=True))
    assert repeated["n_cal_trials"] == 4
    for key in ("between", "within", "reliability", "personal", "final"):
        np.testing.assert_allclose(repeated[key], observed[key], atol=1e-12)


def test_missing_family_label_mismatch_and_forbidden_trial_are_rejected():
    policy = _policy()
    calibration = _calibration()
    with pytest.raises(ValueError, match="every source-selected family"):
        policy.calculate({"flat": calibration["flat"]})
    with pytest.raises(ValueError, match="source/evaluation"):
        policy.calculate(calibration, forbidden_trial_ids=("B2",))
    bad = dict(calibration)
    x, labels, ids = bad["separated"]
    bad["separated"] = (x, np.array(["A", "A", "B", "A"]), ids)
    with pytest.raises(ValueError, match="same trial identities and labels"):
        policy.calculate(bad)
    x, labels, ids = calibration["flat"]
    bad = dict(calibration)
    bad["flat"] = (np.concatenate((x, x[:1])), np.append(labels, "B"), ids + ["A1"])
    with pytest.raises(ValueError, match="multiple gesture labels"):
        policy.calculate(bad)


def test_all_zero_between_yields_uniform_personal_without_target_tuning():
    policy = _policy()
    calibration = _calibration()
    calibration["separated"] = calibration["flat"]
    observed = policy.calculate(calibration)
    np.testing.assert_allclose(observed["personal"], [.5, .5])
    np.testing.assert_allclose(observed["final"], [.55, .45])
    with pytest.raises(ValueError, match="policy identity"):
        DocumentReliabilityWeightsV2(("A", "B"), ("flat", "separated"), (.6, .4),
                                     n0=4, temperature=1, source_policy_id="")
    prior = np.array([.6, .4])
    frozen = DocumentReliabilityWeightsV2(("A", "B"), ("flat", "separated"), prior,
                                          n0=4, temperature=1, source_policy_id="source-cv")
    prior[:] = [.1, .9]
    np.testing.assert_allclose(frozen.population, [.6, .4])
