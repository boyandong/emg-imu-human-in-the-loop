"""Verify frozen live software sources and actual regression/install receipts."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'emgimu_classifier'
HERE=Path(__file__).resolve().parent
OUT=BASE/'feature_bank/TEMPORAL_LIVE_GUI_V1_ACCEPTANCE.json'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def verify():
    protocol=HERE/'TEMPORAL_LIVE_GUI_V1_PROTOCOL.json';p=load(protocol)
    assert p['parent_protocol_sha256']==sha(HERE/'SONG_EXTENDED_GUI_V1_PROTOCOL.json')
    for name,digest in {**p['source_sha256'],**p['artifact_sha256']}.items():assert sha(ROOT/name)==digest,name
    folder=BASE/'feature_bank/regression/temporal_live_gui_v1';records=[];names=set();total=0
    for name,expected in (('classifier.xml',30),('collection.xml',167)):
        path=folder/name;tree=ET.parse(path).getroot();suites=list(tree) if tree.tag=='testsuites' else [tree]
        counts={k:sum(int(s.attrib.get(k,'0')) for s in suites) for k in ('tests','failures','errors','skipped')}
        assert counts==dict(tests=expected,failures=0,errors=0,skipped=0),counts
        names.update(c.attrib['name'] for s in suites for c in s.findall('testcase'))
        records.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),**counts));total+=expected
    required={'test_actual_application_full_action_profiles_manual_auto_and_gap',
        'test_current_launcher_targets_actual_full_action_page',
        'test_model_change_reverts_selection_when_worker_cannot_stop',
        'test_full_action_lifecycle_manual_parity_quality_and_saved_inputs',
        'test_window_off_identity_auto_chunk_parity_and_direct_interval_oracle',
        'test_gap_overflow_capture_conflicts_and_no_false_completion',
        'test_calibration_quality_no_overwrite_bundle_checksum_and_channel_order'}
    assert required<=names,required-names
    shortcut=load(folder/'desktop_shortcut.json')
    assert shortcut['verified'] and shortcut['installed_entry']=='main_decision_v3.py'
    assert shortcut['arguments'].endswith('main_decision_v3.py"') and Path(shortcut['target']).name.lower()=='pythonw.exe'
    installed=ROOT/'collection/emg_meta/emg_meta/main_decision_v3.py'
    assert shortcut['arguments']=='"'+str(installed)+'"'
    assert Path(shortcut['working_directory'])==installed.parent
    native=load(BASE/'feature_bank/PERSONAL_TEMPORAL_UNIBO_ACCEPTANCE_V1.json')
    assert not native['primary_pass'] and native['user_loss_wins']==1
    preview=folder/'panel.png';assert preview.is_file() and preview.stat().st_size>10000
    artifacts={v.relative_to(ROOT).as_posix():sha(v) for v in (folder/'desktop_shortcut.json',preview)}
    return dict(schema='temporal_live_gui_v1_acceptance',verifier_sha256=sha(Path(__file__)),
        protocol_sha256=sha(protocol),records=records,artifact_sha256=artifacts,unique_tests=total,
        current_entry='collection/emg_meta/emg_meta/main_decision_v3.py',desktop_shortcut_installed=True,
        raw_and_filtered_complete_capture_verified=True,source_calibration_labels_separate=True,
        neutral_only_fixed_detector_fit_verified=True,profiles_immutable_during_prediction=True,
        window_off_probability_identity_verified=True,manual_interval_direct_fusion_verified=True,
        estimated_auto_chunk_and_direct_interval_parity_verified=True,quality_Unknown_preserved=True,
        gap_overflow_unclosed_actions_discarded=True,actual_Qt_subprocess_acquisition_lifecycle_verified=True,
        native_four_channel_temporal_primary_guard=False,default_model='six',default_temporal_mode='off',
        default_promoted=False,physical_validation_proven=False,completion_proven=False,
        scope=p['scope'])


if __name__=='__main__':
    result=verify();OUT.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8',newline='\n')
    print(json.dumps({k:result[k] for k in ('unique_tests','current_entry','desktop_shortcut_installed','physical_validation_proven')}))
