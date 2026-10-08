import pickle
import numpy as np
import pytest
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.body_frame_v2 import CalibratedBodyContextV2
from emgimu.feature_bank.electrode_layout_v1 import RingElectrodeLayoutV1
from emgimu.feature_bank.fault_gate_evaluation_v1 import evaluate_fault_gate


def test_f6_unit_equivalence_gravity_and_calibration_isolation():
    emg = np.zeros((1,40,8)); imu = np.zeros((1,10,6))
    imu[:,:,0] = 1.; imu[:,:,2] = 9.80665; imu[:,:,4] = np.pi/2
    neutral = np.zeros((50,6)); neutral[:,2] = 9.80665
    def fitted(units,scale):
        batch = FeatureBatch(emg,200.,imu/scale)
        family = CalibratedBodyContextV2(imu_sample_rate_hz=50.,acceleration_unit=units[0],angular_velocity_unit=units[1])
        family.fit(batch,neutral_calibration_imu=neutral/scale,forward_axis_device=[1.,0.,0.],calibration_trial_ids=['neutral','forward'])
        return family,batch
    a,batch_a = fitted(('m/s^2','rad/s'),np.ones(6))
    b,batch_b = fitted(('g','deg/s'),np.array([9.80665]*3+[np.pi/180]*3))
    previous = pickle.dumps((a,b,batch_a,batch_b))
    first = a.transform(batch_a,trial_ids=['evaluation'])
    np.testing.assert_allclose(first,b.transform(batch_b,trial_ids=['evaluation']),atol=1e-7)
    # Independent constant-step SI oracle; source gravity is9.80665m/s².
    decay = np.exp(-np.arange(1,11)/(50*.5))
    direction = np.array([1-decay.mean(),0,9.80665]); direction /= np.linalg.norm(direction)
    expected = [np.hypot(1,9.80665),0,np.hypot(1,9.80665),np.hypot(1,9.80665),0,
                np.pi/2,0,np.pi/2,np.pi/2,0,*direction,np.sqrt(np.mean(decay**2)),np.pi/2]
    np.testing.assert_allclose(first[0],expected,atol=1e-6)
    assert pickle.dumps((a,b,batch_a,batch_b)) == previous
    with pytest.raises(ValueError,match='held-out'):
        b.transform(batch_b,trial_ids=['neutral'])
    with pytest.raises(ValueError,match='provenance'):
        b.transform(batch_b,trial_ids=[None])
    with pytest.raises(ValueError,match='units'):
        CalibratedBodyContextV2(imu_sample_rate_hz=50.,acceleration_unit='unknown',angular_velocity_unit='rad/s')
    with pytest.raises(ValueError,match='provenance'):
        a.fit(batch_a,neutral_calibration_imu=neutral,forward_axis_device=[1,0,0],calibration_trial_ids=[None])


def test_f6_hardware_contract_changes_are_rejected():
    batch = FeatureBatch(np.zeros((1,40,8)),200.,np.zeros((1,10,6)))
    neutral = np.zeros((50,6)); neutral[:,2] = 9.81
    f = CalibratedBodyContextV2(imu_sample_rate_hz=50.,acceleration_unit='m/s^2',angular_velocity_unit='rad/s')
    f.fit(batch,neutral_calibration_imu=neutral,forward_axis_device=[1,0,0],calibration_trial_ids=['source'])
    with pytest.raises(ValueError,match='Unique'):
        f.fit(batch,neutral_calibration_imu=neutral,forward_axis_device=[1,0,0],calibration_trial_ids=['source','source'])
    with pytest.raises(ValueError,match='parallel'):
        f.fit(batch,neutral_calibration_imu=neutral,forward_axis_device=[0,0,1],calibration_trial_ids=['source'])
    with pytest.raises(ValueError,match='durations'):
        f.transform(FeatureBatch(batch.emg,250.,batch.imu),trial_ids=['eval'])


def layout(order=None,**kwargs):
    data = dict(ring_channels=order or tuple(f'pair-{i}' for i in range(8)),arm='right',
                direction='clockwise',evidence_id='synthetic-fixture-diagram',topology='circumferential_ring')
    data.update(kwargs)
    return RingElectrodeLayoutV1(**data)


def test_ring_mapping_preserves_identified_signals_across_loader_column_order():
    mapping = layout(); canonical = mapping.ring_channels
    x = np.broadcast_to(np.arange(8),(2,40,8)).copy()
    permutation = [4,7,0,2,6,1,5,3]
    data = FeatureBatch(x[:,:,permutation],200.)
    names = tuple(canonical[i] for i in permutation)
    source = mapping.apply_source(data,observed_channel_ids=names)
    evaluation = mapping.apply_evaluation(data,observed_channel_ids=names,model_layout_id=mapping.layout_id)
    np.testing.assert_array_equal(source.emg,x)
    np.testing.assert_array_equal(evaluation.emg,x)
    data.emg[:] = -100
    np.testing.assert_array_equal(evaluation.emg,x)
    with pytest.raises(ValueError,match='source layout'):
        layout(arm='left').apply_evaluation(source,observed_channel_ids=canonical,model_layout_id=mapping.layout_id)
    with pytest.raises(ValueError,match='source layout'):
        layout(direction='counterclockwise').apply_evaluation(source,observed_channel_ids=canonical,model_layout_id=mapping.layout_id)
    with pytest.raises(ValueError,match='eight mapped'):
        mapping.apply_source(source,observed_channel_ids=['unmapped']*8)


def test_ring_mapping_cannot_be_created_without_explicit_geometry_evidence():
    with pytest.raises(ValueError,match='evidence'):
        layout(evidence_id='')
    with pytest.raises(ValueError,match='topology'):
        layout(topology='unknown')
    with pytest.raises(ValueError,match='unique'):
        layout(order=['pair-0']*8)


def test_f9_known_confusion_unknown_exclusion_trial_mass_and_baseline_harm():
    rejected = np.array([True,False,True,False,True])
    labels = ['normal','normal','fault','fault','unknown']
    ids = ['n1','n2','f1','f2','unlabelled']
    correct = np.array([True,True,False,False,True])
    r = evaluate_fault_gate(rejected,labels,ids,source_trial_ids=['source'],baseline_correct=correct)
    assert r['window_counts'] == dict(true_fault_rejections=1,missed_faults=1,
                                    normal_false_rejections=1,normal_accepted=1,unknown_windows=1)
    assert r['trial_balanced_metrics'] == dict(fault_recall=.5,fault_precision=.5,normal_false_rejection_rate=.5)
    assert r['known_trials'] == 4 and r['all_trials'] == 5
    assert r['baseline_harm']['correct_normal_predictions_rejected'] == 1
    assert r['baseline_harm']['correct_normal_rejection_rate'] == .5
    # Duplicating one recording cannot increase its influence in trial metrics.
    repeats = [20,1,7,1,30]
    duplicate = evaluate_fault_gate(np.repeat(rejected,repeats),np.repeat(labels,repeats),
                                   np.repeat(ids,repeats),source_trial_ids=['source'])
    assert duplicate['known_trials'] == 4
    assert duplicate['trial_balanced_metrics'] == pytest.approx(r['trial_balanced_metrics'])
    assert not r['physical_validation_proven']


def test_f9_missing_labels_and_source_reuse_do_not_become_validation():
    unknown = evaluate_fault_gate(np.array([True,False]),['unknown']*2,['e1','e2'],source_trial_ids=['source'])
    assert unknown['known_trials'] == 0
    assert all(v is None for v in unknown['trial_balanced_metrics'].values())
    with pytest.raises(ValueError,match='overlaps'):
        evaluate_fault_gate(np.array([True]),['fault'],['source'],source_trial_ids=['source'])
    with pytest.raises(ValueError,match='annotation'):
        evaluate_fault_gate(np.array([True]),['static_means_fault'],['eval'],source_trial_ids=['source'])
    with pytest.raises(ValueError,match='Boolean'):
        evaluate_fault_gate([.9],['fault'],['eval'],source_trial_ids=['source'])


def test_f6_arbitrary_mount_rotation_matches_independent_closed_form_features():
    # Constant held-out signals give a closed-form gravity-filter response.
    # Reexpress every sensor and measured calibration vector in a rotated
    # device frame; the calibrated body representation must remain unchanged.
    acceleration=np.array([[.7,-.4,10.1],[-.1,.8,9.1]])
    angular_velocity=np.array([[.3,-.2,.4],[.5,.6,-.1]])
    native=np.broadcast_to(np.concatenate([acceleration,angular_velocity],axis=1)[:,None,:],(2,10,6)).copy()
    neutral=np.zeros((50,6));neutral[:,2]=9.80665
    emg=np.zeros((2,40,8))
    from scipy.spatial.transform import Rotation
    rotations=[np.eye(3),Rotation.from_euler('xyz',[-26,32,135],degrees=True).as_matrix(),
               Rotation.from_rotvec(np.array([1.,-2.,.5])/np.sqrt(5.25)*2.1).as_matrix()]
    decay=np.exp(-np.arange(1,11)/(50*.5))
    expected=[]
    for accel,gyro in zip(acceleration,angular_velocity):
        a=np.linalg.norm(accel);g=np.linalg.norm(gyro)
        residual=accel-np.array([0.,0.,9.80665])
        mean_gravity=accel-residual*decay.mean()
        direction=mean_gravity/np.linalg.norm(mean_gravity)
        expected.append([a,0,a,a,0,g,0,g,g,0,*direction,
                         np.linalg.norm(residual)*np.sqrt(np.mean(decay**2)),g])
    for index,rotation in enumerate(rotations):
        transformed=native.copy()
        transformed[:,:,:3]=native[:,:,:3]@rotation.T
        transformed[:,:,3:]=native[:,:,3:]@rotation.T
        calibrated=neutral.copy();calibrated[:,:3]=neutral[:,:3]@rotation.T
        batch=FeatureBatch(emg,200.,transformed)
        model=CalibratedBodyContextV2(imu_sample_rate_hz=50.,acceleration_unit='m/s^2',angular_velocity_unit='rad/s')
        model.fit(batch,neutral_calibration_imu=calibrated,forward_axis_device=rotation@np.array([1.,0.,0.]),
                  calibration_trial_ids=['neutral','guided-forward'])
        source_state=pickle.dumps(model)
        output=model.transform(batch,trial_ids=['eval-a','eval-b'])
        np.testing.assert_allclose(output,expected,atol=2e-6,rtol=0)
        np.testing.assert_allclose(model.device_to_body_@rotation,np.eye(3),atol=1e-12)
        assert pickle.dumps(model)==source_state
