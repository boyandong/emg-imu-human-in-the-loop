"""Discoverable shared lifecycle does not promote a desktop or native result."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def test_joint_lifecycle_receipt_is_indexed_and_scientific_boundaries_remain_open():
    receipt=ROOT/'feature_bank/JOINT_BOUT_WORKFLOW_V1_ACCEPTANCE.json'
    acceptance=json.loads(receipt.read_text(encoding='utf8'))
    index=json.loads((ROOT/'feature_bank/delivery/INDEX.json').read_text(encoding='utf8'))
    row=index['joint_bout_workflow_acceptance']
    assert (ROOT/'feature_bank/delivery'/row['path']).resolve()==receipt.resolve()
    assert row['sha256']==hashlib.sha256(receipt.read_bytes()).hexdigest()
    conclusions=json.loads((ROOT/'feature_bank/CURRENT_SCIENTIFIC_CONCLUSIONS_V1.json').read_text(encoding='utf8'))
    joint=conclusions['joint_bout_workflow']
    assert all(joint[k]==acceptance[k] for k in joint)
    assert joint['shared_calibration_counted_once'] and joint['saved_profile_roundtrip_and_separate_process_verified']
    assert not any(joint[k] for k in ('desktop_entry_changed','native_accuracy_proven','default_promoted','physical_validation_proven','completion_proven'))
    report=(ROOT/'feature_bank/REPORT.md').read_text(encoding='utf8')
    assert 'Joint complete-action registration V1' in report
    assert 'synthetic complete-bout inputs' in report
    assert not conclusions['completion_proven'] and not index['completion_proven']
