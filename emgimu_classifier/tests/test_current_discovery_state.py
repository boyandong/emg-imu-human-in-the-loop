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
    assert all({k: v for k, v in after[key].items() if k in before[key] and k != 'license'} ==
               {k: v for k, v in before[key].items() if k != 'license'}
               for key in before if key != 'ds2_force')
    for key in before:
        if after[key]['license'] != before[key]['license']:
            assert after[key]['recorded_license'] == before[key]['license']
            assert after[key]['license'] == after[key]['primary_metadata_review']['license_verified']
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
    assert rows['ninapro_db6']['raw_acceleration_sample_rate_hz'] == 148.148
    assert rows['ninapro_db6']['aligned_export_sample_rate_hz'] == 2000
    assert rows['great']['uncued_continuous_transitions_verified'] is False
    assert rows['roam_emg']['code_license_verified'] == 'MIT'
    assert rows['roam_emg']['license_verified'] is None
    assert rows['grabmyo']['license_verified'] == 'CC-BY-4.0'
    assert rows['hyser']['license_verified'] == 'ODC-BY-1.0'
    current = json.loads((HERE / 'CURRENT_DISCOVERY_STATE.json').read_text(encoding='utf8'))
    attached = {r['id']: r['primary_metadata_review'] for r in current['datasets']
                if 'primary_metadata_review' in r}
    assert attached == rows


def test_publisher_license_identity_not_inferred_from_article_or_code_license():
    metadata = json.loads((HERE / 'SECONDARY_LICENSE_METADATA_V1.json').read_text(encoding='utf8'))
    assert metadata['runner_sha256'] == sha(HERE / 'scripts/fetch_secondary_license_metadata.py')
    rows = {row['dataset']: row for row in metadata['observations']}
    assert set(rows) == {'ninapro_db5', 'great', 'electrode_replacement_secondary'}
    assert all(row['http_status'] == 200 and row['status'] == 'publisher_metadata_observed'
               and len(row['response_sha256']) == 64 for row in rows.values())
    db5 = rows['ninapro_db5']['fields']
    assert db5['doi'] == '10.5281/zenodo.1000116'
    assert db5['title'] == 'Ninapro dataset 5 (double Myo armband)'
    assert db5['license']['id'] == 'cc-by-nd-4.0'
    assert rows['electrode_replacement_secondary']['fields']['doi'] == '10.5281/zenodo.4039550'
    assert rows['electrode_replacement_secondary']['fields']['license']['id'] == 'cc-by-4.0'
    assert rows['great']['fields']['identifier'] == 'doi:10.5061/dryad.8sf7m0czv'
    assert rows['great']['fields']['license'] == 'https://spdx.org/licenses/CC0-1.0.html'


def test_db6_paper_distinguishes_raw_imu_rate_from_aligned_export():
    paper = json.loads((HERE / 'SECONDARY_PAPER_REVIEW_V1.json').read_text(encoding='utf8'))['papers'][0]
    assert paper['dataset'] == 'ninapro_db6'
    assert paper['pages'] == 6 and paper['visually_reviewed_pages'] == [2, 3]
    assert paper['raw_emg_sample_rate_hz'] == paper['aligned_export_sample_rate_hz'] == 2000
    assert paper['raw_acceleration_sample_rate_hz'] == 148.148
    assert paper['raw_acceleration_sample_rate_hz'] != paper['aligned_export_sample_rate_hz']
    assert not paper['gyroscope_confirmed'] and not paper['calibrated_body_frame_confirmed']
    assert sum(len(entry['quote'].split()) for entry in paper['evidence']) <= 25
