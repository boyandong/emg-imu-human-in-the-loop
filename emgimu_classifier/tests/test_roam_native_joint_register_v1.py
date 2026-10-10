"""Current scientific delivery includes native joint evidence at its actual scope."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def read(name):return json.loads((ROOT/name).read_text(encoding='utf8'))


def test_native_joint_receipt_is_in_current_conclusions_and_delivery_index():
    receipt=read('feature_bank/ROAM_NATIVE_JOINT_V1_ACCEPTANCE.json')
    current=read('feature_bank/CURRENT_SCIENTIFIC_CONCLUSIONS_V1.json')['roam_native_joint']
    assert all(current[k]==v for k,v in receipt.items())
    assert len(current['two_shot_summary'])==12
    index=read('feature_bank/delivery/INDEX.json')['roam_native_joint_acceptance']
    assert index['sha256']==hashlib.sha256((ROOT/'feature_bank/ROAM_NATIVE_JOINT_V1_ACCEPTANCE.json').read_bytes()).hexdigest()
    report=(ROOT/'feature_bank/REPORT.md').read_text(encoding='utf8')
    assert 'Native eight-channel joint cue-interval study' in report
    assert 'not autonomous segmentation or physiological action boundaries' in report
    assert 'not a cross-day or redonning experiment' in report


def test_native_joint_canonical_removals_distinguish_whole_F2_and_whole_F5():
    path=ROOT/'feature_bank/delivery/new_bank_v3/ablation_full_bank.csv'
    with path.open(encoding='utf8',newline='') as stream:
        rows=[r for r in csv.DictReader(stream) if r['run_id']=='roam_native_joint_v1']
    assert len(rows)==624
    assert len([r for r in rows if r['removed_provider']=='family_F2'])==39
    assert len([r for r in rows if r['removed_provider']=='family_F5'])==39
    for row in rows:
        notes=json.loads(row['metadata_notes_json']);shots=int(row['calibration_budget'])
        assert notes['whole_F2_removes_covariance_and_CSP'] and notes['whole_F5_removes_window_and_temporal']
        assert notes['oracle_cue_intervals'] and not notes['autonomous_segmentation_proven']
        assert int(row['full_target_calibration_trials_per_user'])==6+3*shots
        assert int(row['remaining_target_calibration_trials_per_user'])==6+3*shots
        assert int(row['evaluation_trials'])==(18 if row['subject']!='ALL' else 180 if row['session/domain']=='descriptive_all' else 90)
