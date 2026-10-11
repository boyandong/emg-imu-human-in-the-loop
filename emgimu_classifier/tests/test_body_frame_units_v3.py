"""Physical unit identities and disjoint wearing calibration, no native data fit."""
import pickle

import numpy as np
import pytest

from emgimu.feature_bank.body_frame_units_v3 import (
    STANDARD_GRAVITY, CalibratedBodyContextUnitsV3, ImuContractV3,
)
from emgimu.feature_bank.core import FeatureBatch


def contract(acc='m/s^2', gyro='rad/s', rate=100., preprocessing='raw_imu_v1'):
    return ImuContractV3(rate, acc, gyro, preprocessing)


def batch(imu):
    return FeatureBatch(np.ones((len(imu), 50, 8)), 250., imu)


def model(*, units=None, imu=None, neutral=None, axis=(1., 0., 0.)):
    units = contract() if units is None else units
    imu = np.tile([0., 0., STANDARD_GRAVITY, 0., 0., 0.], (1, 20, 1)) if imu is None else imu
    neutral = np.tile([0., 0., STANDARD_GRAVITY, 0., 0., 0.], (100, 1)) if neutral is None else neutral
    return CalibratedBodyContextUnitsV3(imu_sample_rate_hz=100.).fit(batch(imu),
        imu_contract=units, neutral_imu_contract=units, neutral_calibration_imu=neutral,
        forward_axis_device=axis, calibration_trial_ids=['neutral-trial'],
        calibration_recording_ids=['neutral-recording'], user_id='user1', wearing_id='wearing1')


def predict(family, imu, *, units=None, **overrides):
    arguments=dict(imu_contract=contract() if units is None else units,
        trial_ids=['query'], recording_ids=['query-recording'], user_id='user1', wearing_id='wearing1')
    arguments.update(overrides)
    return family.transform(batch(imu), **arguments)


def test_stationary_gravity_and_constant_rotation_are_known_si_coordinates():
    family = model(); frozen = pickle.dumps(family)
    imu = np.tile([0., 0., STANDARD_GRAVITY, 0., 0., 2.], (1, 20, 1))
    actual = predict(family, imu)[0]
    expected = [STANDARD_GRAVITY, 0., STANDARD_GRAVITY, STANDARD_GRAVITY, 0.,
                2., 0., 2., 2., 0., 0., 0., 1., 0., 2.]
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    assert len(family.feature_names) == 15
    assert family.feature_units == ('m/s^2',)*5 + ('rad/s',)*5 + ('dimensionless',)*3 + ('m/s^2','rad/s')
    assert not any('yaw' in name for name in family.feature_names)
    assert pickle.dumps(family) == frozen


@pytest.mark.parametrize('acceleration_unit', ['m/s^2', 'g'])
@pytest.mark.parametrize('angular_unit', ['rad/s', 'deg/s'])
def test_all_four_input_unit_pairs_produce_identical_physical_features(acceleration_unit, angular_unit):
    t = np.arange(20) / 100.
    imu_si = np.column_stack([.2*t, .1*np.sin(2*np.pi*t), np.full(20, STANDARD_GRAVITY),
                             np.full(20, .5), t, np.cos(t)])[None]
    neutral_si = np.tile([0., 0., STANDARD_GRAVITY, 0., 0., 0.], (100, 1))
    scales = np.array([STANDARD_GRAVITY if acceleration_unit=='g' else 1.] * 3
                      + [np.pi/180. if angular_unit=='deg/s' else 1.] * 3)
    units = contract(acceleration_unit, angular_unit)
    physical = model(imu=imu_si, neutral=neutral_si)
    declared = model(units=units, imu=imu_si/scales, neutral=neutral_si/scales)
    expected = predict(physical, imu_si)
    np.testing.assert_allclose(predict(declared, imu_si/scales, units=units), expected, rtol=1e-6, atol=1e-7)
    # A query may declare another known unit without silently changing the frame.
    np.testing.assert_allclose(predict(declared, imu_si), expected, rtol=1e-6, atol=1e-7)


def test_nonorthogonal_device_calibration_rotates_gravity_without_absolute_yaw_claim():
    # Body gravity is measured along device +y and guided forward along +z.
    family = model(imu=np.tile([0., STANDARD_GRAVITY, 0., 0., 0., 0.], (1, 20, 1)),
                   neutral=np.tile([0., STANDARD_GRAVITY, 0., 0., 0., 0.], (100, 1)), axis=(0., .3, 1.))
    result = predict(family, np.tile([0., STANDARD_GRAVITY, 0., 0., 0., 1.], (1, 20, 1)))
    np.testing.assert_allclose(result[0, 10:13], [0., 0., 1.], atol=1e-7)
    np.testing.assert_allclose(family.device_to_body_ @ family.device_to_body_.T, np.eye(3), atol=1e-14)
    assert np.linalg.det(family.device_to_body_) == pytest.approx(1.)


@pytest.mark.parametrize('override', [dict(acceleration_unit='unknown'), dict(angular_velocity_unit='counts'),
    dict(acceleration_unit=''), dict(preprocessing_id=''), dict(sample_rate_hz=True),
    dict(sample_rate_hz=0.), dict(sample_rate_hz=float('nan')),
    dict(sample_rate_hz='100'), dict(sample_rate_hz=None), dict(sample_rate_hz=[]),
    dict(acceleration_unit=[]), dict(angular_velocity_unit=None),
    dict(channel_order=('gx','gy','gz','ax','ay','az'))])
def test_uninterpretable_units_rates_and_column_order_reject_before_feature_computation(override):
    arguments=dict(sample_rate_hz=100., acceleration_unit='m/s^2', angular_velocity_unit='rad/s',
                   preprocessing_id='raw_imu_v1')
    arguments.update(override)
    with pytest.raises(ValueError): ImuContractV3(**arguments)


@pytest.mark.parametrize('overrides', [dict(user_id='other'), dict(wearing_id='redonned'),
    dict(recording_ids=['neutral-recording']), dict(trial_ids=['neutral-trial']),
    dict(trial_ids=[None]), dict(recording_ids=['']), dict(trial_ids=[1]), dict(evaluation=1),
    dict(imu_contract=contract(rate=50.)), dict(imu_contract=contract(preprocessing='highpass_imu')),
    dict(imu_contract=None)])
def test_profile_cross_use_and_recording_leakage_are_rejected_without_state_mutation(overrides):
    family=model(); frozen=pickle.dumps(family)
    with pytest.raises(ValueError):predict(family,np.zeros((1,20,6)),**overrides)
    assert pickle.dumps(family)==frozen


@pytest.mark.parametrize('trial_ids,recording_ids', [([None],['r']),([1],['r']),(['t'],['']),
    (['t','t'],['r','r']),(['t'],[]),([],[]),('t','r')])
def test_frame_calibration_requires_distinct_explicit_independent_trial_provenance(trial_ids,recording_ids):
    family=CalibratedBodyContextUnitsV3(imu_sample_rate_hz=100.)
    with pytest.raises(ValueError):
        family.fit(batch(np.zeros((1,20,6))),imu_contract=contract(),neutral_imu_contract=contract(),
            neutral_calibration_imu=np.tile([0.,0.,STANDARD_GRAVITY,0.,0.,0.],(100,1)),
            forward_axis_device=[1.,0.,0.],calibration_trial_ids=trial_ids,
            calibration_recording_ids=recording_ids,user_id='u',wearing_id='w')
    assert not family.fitted_


def test_window_and_neutral_contract_mismatch_cannot_create_a_fitted_profile():
    family=CalibratedBodyContextUnitsV3(imu_sample_rate_hz=100.)
    with pytest.raises(ValueError,match='preprocessing'):
        family.fit(batch(np.zeros((1,20,6))),imu_contract=contract(),
            neutral_imu_contract=contract(preprocessing='other'),
            neutral_calibration_imu=np.tile([0.,0.,STANDARD_GRAVITY,0.,0.,0.],(100,1)),
            forward_axis_device=[1.,0.,0.],calibration_trial_ids=['t'],calibration_recording_ids=['r'],
            user_id='u',wearing_id='w')
    assert not family.fitted_


def test_pickle_round_trip_and_repeated_query_windows_preserve_profile_and_provenance():
    family=pickle.loads(pickle.dumps(model()));frozen=pickle.dumps(family)
    imu=np.tile([0.,0.,STANDARD_GRAVITY,0.,0.,0.],(2,20,1))
    output=predict(family,imu,trial_ids=['q','q'],recording_ids=['r','r'])
    np.testing.assert_array_equal(output[0],output[1])
    with pytest.raises(ValueError,match='different recording'):
        predict(family,imu,trial_ids=['q','q'],recording_ids=['r','other'])
    assert pickle.dumps(family)==frozen


def test_boolean_configured_sampling_rate_cannot_become_one_hz_imu():
    with pytest.raises(ValueError, match='sampling rate'):
        CalibratedBodyContextUnitsV3(imu_sample_rate_hz=True)
