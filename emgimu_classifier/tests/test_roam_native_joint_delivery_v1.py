"""Native result persistence, denominators and explicit scientific limits."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def read(name):return json.loads((ROOT/name).read_text(encoding='utf8'))


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def test_native_joint_protocol_and_all_saved_artifacts_match():
    p=read('benchmarks/new_bank_v3/ROAM_NATIVE_JOINT_V1_PROTOCOL.json')
    r=read('benchmarks/new_bank_v3/ROAM_NATIVE_JOINT_V1_RESULTS.json')
    assert r['protocol_sha256']==sha(ROOT/'benchmarks/new_bank_v3/ROAM_NATIVE_JOINT_V1_PROTOCOL.json')
    for name,digest in {**p['source_sha256'],**r['artifacts_sha256']}.items():assert sha(ROOT/name)==digest,name
    assert len(p['native_recordings'])==58
    assert sum(len(n['cue_intervals']) for n in p['native_recordings'])==522
    assert not any(n['excluded_intervals'] for n in p['native_recordings'])


def test_native_joint_queries_are_identical_at_all_budgets_and_whole_recording_disjoint():
    r=read('benchmarks/new_bank_v3/ROAM_NATIVE_JOINT_V1_RESULTS.json')
    arrays=np.load(ROOT/'benchmarks/new_bank_v3/roam_native_joint_v1/readouts.npz',allow_pickle=False)
    source=set(r['source_trial_ids']);assert len(source)==162
    for subject in r['subjects']:
        u=subject['user'];query=set(subject['query_ids']);assert len(query)==18 and not query&source
        assert not query&set(subject['personal_calibration_ids'])
        assert len(subject['personal_calibration_ids'])==6
        assert all('unsupported.csv' in t or 'reaching.csv' in t for t in query)
        for shots in (0,1,2):
            budget=subject['budgets'][str(shots)]
            assert len(budget['current_calibration_ids'])==3*shots
            assert not query&set(budget['current_calibration_ids'])
            assert all('hanging.csv' in t for t in budget['current_calibration_ids'])
            assert len(budget['arms'])==29
            for arm in budget['arms']:
                q=arrays[f'u{u}_s{shots}_{arm}'];assert q.shape==(18,3)
                np.testing.assert_allclose(q.sum(1),1.,rtol=0,atol=1e-12)
    arrays.close()
    assert len(r['cells'])==870 and r['prediction_rows']==15660 and r['independent_query_trials']==180


def test_native_joint_receipt_preserves_negative_guards_and_unavailable_sensor_claims():
    r=read('benchmarks/new_bank_v3/ROAM_NATIVE_JOINT_V1_RESULTS.json')
    a=read('feature_bank/ROAM_NATIVE_JOINT_V1_ACCEPTANCE.json')
    assert a['result_sha256']==sha(ROOT/'benchmarks/new_bank_v3/ROAM_NATIVE_JOINT_V1_RESULTS.json')
    assert a['primary_guards']==r['primary_guards'] and a['primary_pass']==all(r['primary_guards'].values())
    assert a['read_only_no_fitting'] and a['calibration_query_recordings_disjoint'] and a['source_user_folds_disjoint']
    assert a['channels']==8 and a['native_sample_rate_hz']==200. and a['classes']==['close','open','relax']
    for key in ('maximum_manual_source_probability_error','maximum_composition_probability_error',
                'maximum_independent_temporal_probability_error','maximum_saved_profile_replay_error'):
        assert a[key]<1e-12
    for key in ('resampled','raw_quality_policy_available','autonomous_segmentation_proven',
                'physiological_boundaries_proven','physical_validation_proven','default_promoted','completion_proven'):
        assert not a[key]
    assert a['previously_inspected_public_users'] and a['profiles_preserve_all_calibration_samples']
