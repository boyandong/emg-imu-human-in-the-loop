import json
from pathlib import Path
from benchmarks.discovery.scripts.current_discovery_state import sha

HERE = Path(__file__).resolve().parents[1] / 'benchmarks/discovery'


def test_current_ds2_state_supersedes_only_stale_force_claims():
    current = json.loads((HERE / 'CURRENT_DISCOVERY_STATE.json').read_text(encoding='utf8'))
    old = json.loads((HERE / current['historical_manifest']).read_text(encoding='utf8'))
    for path, expected in current['source_sha256'].items():
        assert sha(HERE / path) == expected
    before = {d['id']: d for d in old['datasets']}
    after = {d['id']: d for d in current['datasets']}
    assert before.keys() == after.keys()
    assert before['ds2_force']['per_trial_force_labels'] == 'unverified'
    assert after['ds2_force']['per_trial_force_labels'] == 'verified_exact_window_to_raw_trial_join'
    assert after['ds2_force']['force_labelled_raw_trials'] == 2863
    assert after['ds2_force']['subject_gesture_force_eligible_trials'] == 2832
    assert after['ds2_force']['active_study_trials'] == 2297
    assert after['ds2_force']['license'] == before['ds2_force']['license']
    assert all({k: v for k, v in after[key].items() if k != 'primary_metadata_review'} == before[key]
               for key in before if key != 'ds2_force')
    assert current['completion_proven'] is False


def test_secondary_metadata_keeps_sensor_and_license_boundaries():
    review = json.loads((HERE / 'SECONDARY_PRIMARY_REVIEW_V1.json').read_text(encoding='utf8'))
    rows = {r['id']: r for r in review['datasets']}
    assert len(rows) == len(review['datasets']) == 7
    assert review['completion_proven'] is False
    assert all(r['sources'] and r['paper_review'] and r['limitations'] for r in rows.values())
    assert all(r['six_axis_imu_verified'] is False for r in rows.values())
    assert all(r['physical_eight_channel_layout_verified'] is False for r in rows.values())
    assert rows['ninapro_db5']['acceleration_columns'] == 3
    assert rows['ninapro_db6']['active_emg_channels'] == 14
    assert rows['ninapro_db6']['stored_emg_channels'] == 16
    assert rows['great']['uncued_continuous_transitions_verified'] is False
    assert rows['roam_emg']['code_license_verified'] == 'MIT'
    assert rows['roam_emg']['license_verified'] is None
    assert rows['grabmyo']['license_verified'] == 'CC-BY-4.0'
    assert rows['hyser']['license_verified'] == 'ODC-BY-1.0'
    current = json.loads((HERE / 'CURRENT_DISCOVERY_STATE.json').read_text(encoding='utf8'))
    attached = {r['id']: r['primary_metadata_review'] for r in current['datasets']
                if 'primary_metadata_review' in r}
    assert attached == rows
