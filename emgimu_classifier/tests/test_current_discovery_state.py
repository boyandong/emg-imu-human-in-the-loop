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
    assert all(after[key] == before[key] for key in before if key != 'ds2_force')
    assert current['completion_proven'] is False
