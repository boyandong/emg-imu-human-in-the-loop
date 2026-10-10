"""Current desktop integration is discoverable without changing efficacy gates."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def test_current_joint_gui_index_conclusions_and_windows_render_are_bound():
    path=ROOT/'feature_bank/JOINT_BOUT_GUI_V1_ACCEPTANCE.json'
    receipt=json.loads(path.read_text(encoding='utf8'))
    index=json.loads((ROOT/'feature_bank/delivery/INDEX.json').read_text(encoding='utf8'))
    row=index['joint_bout_gui_acceptance']
    assert (ROOT/'feature_bank/delivery'/row['path']).resolve()==path.resolve()
    assert row['sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    conclusions=json.loads((ROOT/'feature_bank/CURRENT_SCIENTIFIC_CONCLUSIONS_V1.json').read_text(encoding='utf8'))
    gui=conclusions['joint_bout_gui'];assert all(gui[k]==receipt[k] for k in gui)
    assert gui['shared_registration_in_desktop'] and gui['default_model']=='six'
    assert not any(gui[k] for k in ('default_promoted','native_accuracy_proven','physical_validation_proven','completion_proven'))
    render=json.loads((ROOT/'feature_bank/regression/joint_bout_gui_v1/windows_render.json').read_text(encoding='utf8'))
    assert render['platform']=='windows' and render['font']=='Microsoft YaHei UI'
    assert render['personal_trials']==render['session_trials']==4
    assert render['actual_Qt_controls_and_classifier_process'] and not render['new_fit_or_native_scoring']
    source=ROOT/'benchmarks/new_bank_v3/render_joint_bout_gui_v1.py'
    assert render['source_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
    panel=ROOT/'feature_bank/regression/joint_bout_gui_v1/panel.png'
    assert render['panel_sha256']==hashlib.sha256(panel.read_bytes()).hexdigest()
    report=(ROOT/'feature_bank/REPORT.md').read_text(encoding='utf8')
    assert 'Joint registration in the versioned desktop' in report
    assert 'supersedes the earlier API-only desktop-integration gap' in report
    assert not conclusions['completion_proven'] and not index['completion_proven']
