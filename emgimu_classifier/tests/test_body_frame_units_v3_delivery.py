import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(path.read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def test_body_frame_receipt_binds_actual_junit_source_and_physical_limits():
    a=read(ROOT/'feature_bank/BODY_FRAME_UNITS_V3_ACCEPTANCE.json')
    for rel,digest in a['source_sha256'].items():assert sha(ROOT/rel)==digest
    assert a['generator_sha256']==sha(ROOT/'benchmarks/body_frame_units_v3_acceptance.py')
    junit=ROOT/a['regression_junit']['path'];assert sha(junit)==a['regression_junit']['sha256']
    suites=ET.parse(junit).getroot().findall('testsuite')
    assert sum(int(s.get('tests',0)) for s in suites)==a['passing_cases']==58
    assert not any(int(s.get(k,0)) for s in suites for k in ('failures','errors','skipped'))
    assert a['output_dimensions']==len(a['canonical_output_units'])==15
    assert not any(a[k] for k in ('frozen_v1_source_modified','frozen_v2_source_modified','native_experiment_rerun','native_data_used',
        'native_accuracy_proven','physical_validation_proven','absolute_yaw_proven','default_promoted','completion_proven'))


def test_current_delivery_exposes_versioned_units_without_claiming_native_f6():
    a=read(ROOT/'feature_bank/BODY_FRAME_UNITS_V3_ACCEPTANCE.json')
    current=read(ROOT/'feature_bank/CURRENT_SCIENTIFIC_CONCLUSIONS_V1.json')['body_frame_units']
    assert all(current[k]==v for k,v in a.items())
    index=read(ROOT/'feature_bank/delivery/INDEX.json')['body_frame_units_acceptance']
    assert index['sha256']==sha(ROOT/'feature_bank/BODY_FRAME_UNITS_V3_ACCEPTANCE.json')
    assert 'Unit-checked and wearing-bound F6 software' in (ROOT/'feature_bank/REPORT.md').read_text(encoding='utf8')
