"""Desktop evidence is software lifecycle proof, not native efficacy."""
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def test_joint_desktop_receipt_binds_sources_actual_regression_and_entry():
    source=ROOT/'benchmarks/new_bank_v3/verify_joint_bout_gui_v1.py'
    spec=importlib.util.spec_from_file_location('joint_gui_verifier',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    saved=json.loads((ROOT/'feature_bank/JOINT_BOUT_GUI_V1_ACCEPTANCE.json').read_text(encoding='utf8'))
    assert module.verify()==saved
    assert saved['total_passing_cases']==225 and saved['shared_registration_in_desktop']
    assert saved['default_model']=='six'
    assert not any(saved[k] for k in ('default_promoted','native_accuracy_proven','physical_validation_proven','completion_proven'))
