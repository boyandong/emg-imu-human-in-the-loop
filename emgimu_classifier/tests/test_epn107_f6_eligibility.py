"""Read back the complete public EPN107 MAT-schema F6 eligibility census."""

import hashlib
import json
from pathlib import Path

import numpy as np

from benchmarks.new_bank_v3.epn107_f6_eligibility_probe import raw_six_axis


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / 'benchmarks/new_bank_v3/EPN107_F6_ELIGIBILITY.json'
PROBE = ROOT / 'benchmarks/new_bank_v3/epn107_f6_eligibility_probe.py'


def test_raw_imu_requires_both_nonempty_three_axis_arrays():
    good = {'accel': np.ones((50, 3)), 'gyro': np.ones((50, 3))}
    assert raw_six_axis(good)
    assert not raw_six_axis({**good, 'gyro': np.empty(0)})
    assert not raw_six_axis({**good, 'accel': np.ones((50, 4))})


def test_archive_census_rejects_unproved_f6_calibration():
    result = json.loads(RESULT.read_text(encoding='utf-8'))
    assert result['archive_bytes'] == 998699553
    assert result['archive_hash']['md5'] == '90da8bb6d94e72d8821ae1859b4b878f'
    assert result['probe_sha256'] == hashlib.sha256(PROBE.read_bytes()).hexdigest()
    assert result['member_count'] == 107
    assert result['device_count'] == {'myo': 38, 'gForce': 69}
    coverage = result['raw_accel_and_gyro_member_count']
    for subset in ('sync', 'training', 'testing'):
        assert coverage[f'myo:{subset}:True'] == 38
        assert coverage[f'gForce:{subset}:False'] == 69
    assert not result['forward_axis_or_mounting_field_names']
    assert not result['imu_rate_or_unit_field_names']
    assert not result['eligible_for_strict_calibrated_f6']
    assert not result['requirements']['explicit_device_frame_anatomical_forward_axis']
