"""Verify frozen joint lifecycle software evidence; never train or score efficacy."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[2]
PROTOCOL=Path(__file__).with_name('JOINT_BOUT_WORKFLOW_V1_PROTOCOL.json')
ACCEPTANCE=ROOT/'feature_bank/JOINT_BOUT_WORKFLOW_V1_ACCEPTANCE.json'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    protocol=json.loads(PROTOCOL.read_text(encoding='utf8'))
    for name,digest in protocol['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen joint source differs: '+name)
    junit=ROOT/protocol['junit']
    tree=ET.parse(junit).getroot()
    suites=list(tree.iter('testsuite'))
    if not suites or any(int(s.attrib.get(k,0)) for s in suites for k in ('failures','errors','skipped')):
        raise ValueError('Joint regression must pass without failures, errors or skips')
    cases=list(tree.iter('testcase'))
    if any(c.find(k) is not None for c in cases for k in ('failure','error','skipped')):
        raise ValueError('Individual joint regression case did not pass')
    ids=[c.attrib['classname']+'::'+c.attrib['name'] for c in cases]
    if len(ids)!=len(set(ids)) or sum(int(s.attrib['tests']) for s in suites)!=len(ids):
        raise ValueError('Duplicate or miscounted joint regression cases')
    selected=[i for i in ids if i.startswith(('tests.test_joint_bout_workflow_v1::','tests.test_joint_bout_cli_v1::'))]
    matrix=[i for i in selected if 'test_joint_branch_matrix_independent_mixture_and_immutable_state[' in i]
    if len(selected)!=54 or len(matrix)!=18:
        raise ValueError('Incomplete joint lifecycle or combined decision matrix')
    for name in protocol['required_test_names']:
        if not any(i.split('::')[1]==name for i in ids):raise ValueError('Required joint oracle missing: '+name)
    return dict(schema='joint_bout_workflow_v1_acceptance',protocol_sha256=sha(PROTOCOL),
        verifier_sha256=sha(Path(__file__)),junit=dict(path=protocol['junit'],sha256=sha(junit),
        tests=len(cases),failures=0,errors=0,skipped=0,testcase_ids=ids),
        source_files_checked=len(protocol['source_sha256']),joint_lifecycle_tests=len(selected),
        combined_window_temporal_quality_cases=len(matrix),
        exact_native_calibration_inputs_saved=True,shared_calibration_counted_once=True,
        neutral_only_detector_independent_arithmetic=True,source_and_parent_immutable=True,
        independent_outer_probability_mixture_verified=True,
        saved_profile_roundtrip_and_separate_process_verified=True,
        trial_recording_source_isolation_and_failure_cleanup_verified=True,
        window_quality_Unknown_cannot_be_rescued=True,
        estimated_boundaries_remain_estimated=True,
        desktop_entry_changed=False,default_promoted=False,native_accuracy_proven=False,
        physical_validation_proven=False,completion_proven=False,
        scope=protocol['scope'],known_limitations=protocol['known_limitations'])


if __name__=='__main__':
    result=verify()
    ACCEPTANCE.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8',newline='\n')
    print(json.dumps({k:result[k] for k in ('joint_lifecycle_tests','combined_window_temporal_quality_cases',
        'native_accuracy_proven','completion_proven')}))
