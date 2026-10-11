"""Bind actual F6 unit/provenance regression evidence; no native fitting."""
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
TESTS=('tests/test_body_frame_units_v3.py','tests/test_body_frame.py','tests/test_posture_context_oracle.py',
       'tests/test_native_document_reliability_v3.py','tests/test_document_session_v3.py')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    junit=ROOT/'feature_bank/regression/body_frame_units_v3/classifier.xml'
    tree=ET.parse(junit).getroot()
    suites=tree.findall('testsuite')
    counts={k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    assert counts==dict(tests=58,failures=0,errors=0,skipped=0),counts
    cases=list(tree.iter('testcase'))
    for name in TESTS:
        classname=name[:-3].replace('/','.')
        assert any(c.get('classname','').startswith(classname) for c in cases),name
    path=ROOT/'feature_bank/BODY_FRAME_UNITS_V3_ACCEPTANCE.json'
    result=dict(schema='body_frame_units_v3_acceptance',passing_cases=counts['tests'],failures=0,errors=0,skipped=0,
        regression_junit=dict(path=junit.relative_to(ROOT).as_posix(),sha256=sha(junit)),
        generator_sha256=sha(Path(__file__)),
        source_sha256={p:sha(ROOT/p) for p in ('src/emgimu/feature_bank/body_frame_units_v3.py',
            'src/emgimu/feature_bank/body_frame.py','src/emgimu/feature_bank/body_frame_v2.py','src/emgimu/feature_bank/core.py',*TESTS)},
        fixture_imu_rate_hz=100.,fixture_emg_rate_hz=250.,output_dimensions=15,
        acceleration_units=['m/s^2','g'],angular_velocity_units=['rad/s','deg/s'],
        standard_gravity_m_per_s2=9.80665,canonical_output_units=['m/s^2']*5+['rad/s']*5+['dimensionless']*3+['m/s^2','rad/s'],
        numeric_evidence=['constant gravity/rotation hand-computable15-output oracle',
            'all four acceleration/angular unit combinations preserve SI features',
            'guided nonorthogonal device axis creates a right-handed orthonormal frame'],
        contract_evidence=['unknown/empty/non-string units rejected','explicit accel/gyro column order',
            'same real IMU rate and preprocessing for neutral/source/query',
            'distinct nonempty string calibration trials and recording IDs',
            'user and wearing identity bind the calibrated frame',
            'calibration trials/recordings excluded from held-out query',
            'dependent windows keep one trial identity','pickle and inference preserve source profile'],
        frozen_v1_source_modified=False,frozen_v2_source_modified=False,native_experiment_rerun=False,native_data_used=False,
        native_accuracy_proven=False,physical_validation_proven=False,absolute_yaw_proven=False,
        default_promoted=False,completion_proven=False,
        scope='Versioned opt-in F6 software only. Known-unit conversion, guided-frame arithmetic and calibration provenance are verified on synthetic physical signals with100Hz IMU/250Hz EMG. Unit strings are validated, not independently measured. Measured anatomical forward direction, neutral calibration, native units/channel order and real user/wearing data remain required; no absent IMU or physical calibration is fabricated.')
    path.write_bytes((json.dumps(result,indent=2)+'\n').encode())
    print(json.dumps({k:result[k] for k in ('passing_cases','native_data_used','physical_validation_proven','completion_proven')}))
    return result


if __name__=='__main__':build()
