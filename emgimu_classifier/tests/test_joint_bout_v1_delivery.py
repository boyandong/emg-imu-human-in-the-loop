"""Joint lifecycle evidence covers software composition, never device accuracy."""
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def test_joint_lifecycle_frozen_sources_and_actual_junit_receipt():
    source=ROOT/'benchmarks/new_bank_v3/verify_joint_bout_workflow_v1.py'
    spec=importlib.util.spec_from_file_location('joint_lifecycle_verifier',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    actual=module.verify()
    saved=json.loads((ROOT/'feature_bank/JOINT_BOUT_WORKFLOW_V1_ACCEPTANCE.json').read_text(encoding='utf8'))
    assert actual==saved
    assert actual['joint_lifecycle_tests']==54 and actual['combined_window_temporal_quality_cases']==18
    assert not any(actual[k] for k in ('desktop_entry_changed','native_accuracy_proven',
        'physical_validation_proven','default_promoted','completion_proven'))
