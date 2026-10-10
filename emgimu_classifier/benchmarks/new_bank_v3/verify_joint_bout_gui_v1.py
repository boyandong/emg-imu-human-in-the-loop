"""Verify versioned desktop joint registration from frozen software evidence."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[2]
REPO=ROOT.parent
PROTOCOL=Path(__file__).with_name('JOINT_BOUT_GUI_V1_PROTOCOL.json')
ACCEPTANCE=ROOT/'feature_bank/JOINT_BOUT_GUI_V1_ACCEPTANCE.json'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    protocol=json.loads(PROTOCOL.read_text(encoding='utf8'))
    for name,digest in protocol['source_sha256'].items():
        if sha(REPO/name)!=digest:raise ValueError('Frozen GUI source differs: '+name)
    records=[]
    for suite in protocol['suites']:
        path=ROOT/suite['junit'];tree=ET.parse(path).getroot();groups=list(tree.iter('testsuite'))
        cases=list(tree.iter('testcase'))
        if (not groups or any(int(g.attrib.get(k,0)) for g in groups for k in ('failures','errors','skipped'))
                or len(cases)!=suite['expected_tests'] or sum(int(g.attrib['tests']) for g in groups)!=len(cases)
                or any(c.find(k) is not None for c in cases for k in ('failure','error','skipped'))):
            raise ValueError('Incomplete or failing GUI regression: '+suite['name'])
        ids=[c.attrib['classname']+'::'+c.attrib['name'] for c in cases]
        if len(set(ids))!=len(ids):raise ValueError('Duplicate GUI regression cases')
        for required in suite['required_names']:
            if not any(c.attrib['name']==required for c in cases):raise ValueError('Missing GUI lifecycle check: '+required)
        records.append(dict(name=suite['name'],path=suite['junit'],sha256=sha(path),tests=len(cases),
            failures=0,errors=0,skipped=0,testcase_ids=ids))
    shortcut=ROOT/protocol['shortcut_receipt'];link=json.loads(shortcut.read_text(encoding='utf8'))
    app=REPO/'collection/emg_meta/emg_meta/main_decision_v4.py'
    if (Path(link['target']).name.lower()!='pythonw.exe' or link['arguments']!='"'+str(app)+'"'
            or Path(link['working_directory']).resolve()!=app.parent.resolve()
            or link['entry_sha256']!=sha(app) or not link['read_back_verified']):
        raise ValueError('Desktop shortcut does not target the verified versioned entry')
    screenshot=ROOT/protocol['panel_artifact']
    if not screenshot.read_bytes().startswith(b'\x89PNG\r\n\x1a\n'):raise ValueError('Missing real Qt panel rendering')
    return dict(schema='joint_bout_gui_v1_acceptance',protocol_sha256=sha(PROTOCOL),verifier_sha256=sha(Path(__file__)),
        records=records,total_passing_cases=sum(r['tests'] for r in records),
        source_files_checked=len(protocol['source_sha256']),
        artifacts={protocol['shortcut_receipt']:sha(shortcut),protocol['panel_artifact']:sha(screenshot)},
        entry='collection/emg_meta/emg_meta/main_decision_v4.py',desktop_shortcut_installed_and_read_back=True,
        shared_registration_in_desktop=True,shared_native_raw_filtered_inputs_saved=True,
        window_and_temporal_profiles_bound_together=True,personal_and_session_reload_verified=True,
        manual_and_estimated_auto_decisions_verified=True,failed_save_retry_and_failed_load_pause_verified=True,
        gap_disconnect_and_cancel_clear_incomplete_actions=True,quality_Unknown_preserved=True,
        window_stream_parity_with_frozen_parent_verified=True,
        default_model='six',default_promoted=False,native_accuracy_proven=False,
        physical_validation_proven=False,completion_proven=False,scope=protocol['scope'])


if __name__=='__main__':
    result=verify();ACCEPTANCE.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8',newline='\n')
    print(json.dumps({k:result[k] for k in ('total_passing_cases','shared_registration_in_desktop',
        'native_accuracy_proven','completion_proven')}))
