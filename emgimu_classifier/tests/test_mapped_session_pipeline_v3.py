import pickle
import numpy as np
import pytest
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.electrode_layout_v1 import RingElectrodeLayoutV1
from emgimu.feature_bank.force_nested_oof import SubjectWindows
from emgimu.feature_bank.mapped_session_pipeline_v3 import MappedSessionCalibrationPipelineV3
from emgimu.feature_bank.session_pipeline import DocumentSessionCalibrationPipelineV2

NAMES = tuple(f'pair_{c}' for c in range(8))


def data(prefix):
    rng = np.random.default_rng(19)
    y = np.repeat(np.arange(4), 8)
    x = rng.normal(size=(32, 50, 8)) * (.1 + y[:, None, None])
    x *= 1 + np.arange(8)[None, None, :] * y[:, None, None] / 10
    return SubjectWindows(FeatureBatch(x, 250), y, np.repeat('user', len(y)),
                          np.array([f'{prefix}_{label}_{rep // 2}' for label in range(4) for rep in range(8)]))


def reorder(dataset, permutation):
    return SubjectWindows(FeatureBatch(dataset.batch.emg[:, :, permutation], 250),
                          dataset.labels, dataset.subjects, dataset.trials)


def layout(arm='right'):
    return RingElectrodeLayoutV1(NAMES, arm, 'clockwise', 'fixture-wire-map', 'circumferential_ring')


def test_different_loader_orders_keep_source_calibration_and_predictions_identical():
    source, cal, target = data('source'), data('cal'), data('target')
    reference = DocumentSessionCalibrationPipelineV2(rest_label=0, ring_topology=True).fit_long_term(source)
    ref_session = reference.calibrate_session(cal)
    expected, expected_trials = reference.predict_unlabeled(target.batch, target.subjects, target.trials, ref_session)
    a, b, c = np.array([5, 1, 7, 0, 6, 3, 2, 4]), np.arange(8)[::-1], np.roll(np.arange(8), 3)
    pipeline = MappedSessionCalibrationPipelineV3(rest_label=0, layout=layout()).fit_long_term(
        reorder(source, a), observed_channel_ids=np.array(NAMES)[a])
    session = pipeline.calibrate_session(reorder(cal, b), observed_channel_ids=np.array(NAMES)[b])
    before = pickle.dumps((pipeline, session))
    target = reorder(target, c)
    actual, trials = pipeline.predict_unlabeled(target.batch, target.subjects, target.trials, session,
                                              observed_channel_ids=np.array(NAMES)[c])
    np.testing.assert_array_equal(trials, expected_trials)
    for branch in expected:
        for family in expected[branch]:
            np.testing.assert_allclose(actual[branch][family], expected[branch][family], atol=1e-12)
    scored, _, scored_trials = pipeline.predict(target, session, observed_channel_ids=np.array(NAMES)[c])
    np.testing.assert_array_equal(scored_trials, trials)
    for branch in actual:
        for family in actual[branch]:
            np.testing.assert_allclose(scored[branch][family], actual[branch][family], atol=1e-12)
    assert before == pickle.dumps((pipeline, session))


def test_mapped_flow_rejects_layout_identity_and_trial_leakage_before_prediction():
    pipeline = MappedSessionCalibrationPipelineV3(rest_label=0, layout=layout())
    with pytest.raises(RuntimeError, match='fit first'):
        pipeline.calibrate_session(data('cal'), observed_channel_ids=NAMES)
    invalid = data('source')
    invalid.trials[0] = ''
    with pytest.raises(ValueError, match='trial identities'):
        pipeline.fit_long_term(invalid, observed_channel_ids=NAMES)
    pipeline.fit_long_term(data('source'), observed_channel_ids=NAMES)
    session = pipeline.calibrate_session(data('cal'), observed_channel_ids=NAMES)
    for prefix in ('source', 'cal'):
        target = data(prefix)
        with pytest.raises(ValueError, match='held-out evaluation'):
            pipeline.predict_unlabeled(target.batch, target.subjects, target.trials, session, observed_channel_ids=NAMES)
    bad_session = dict(session, physical_layout_id=layout('left').layout_id)
    with pytest.raises(ValueError, match='Session physical layout'):
        pipeline.predict(data('target'), bad_session, observed_channel_ids=NAMES)
    with pytest.raises(ValueError, match='Observed channel identities'):
        pipeline.predict(data('target'), session, observed_channel_ids=tuple(f'unknown_{i}' for i in range(8)))
    pipeline.layout = layout('left')
    with pytest.raises(ValueError, match='Physical layout differs'):
        pipeline.predict(data('target'), session, observed_channel_ids=NAMES)
