"""Inspect public EPN107 MAT schemas for strict F6 calibration eligibility.

Only aggregate field names and array shapes are saved. Demographic values,
photos, raw sensor arrays and gesture labels are never exported.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import subprocess

import numpy as np
from scipy.io import loadmat


EXPECTED_MD5 = '90da8bb6d94e72d8821ae1859b4b878f'
PUBLIC_RECORD = 'https://zenodo.org/records/19829636'
MEMBER = re.compile(r'^(training|testing)/user_\d{3}/userData\.mat$')


def digests(path: Path) -> dict[str, str]:
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            md5.update(chunk)
            sha256.update(chunk)
    return {'md5': md5.hexdigest(), 'sha256': sha256.hexdigest()}


def sensor_shape(sample: dict, key: str) -> tuple[int, ...]:
    return tuple(np.asarray(sample.get(key, [])).shape)


def raw_six_axis(sample: dict) -> bool:
    a, g = sensor_shape(sample, 'accel'), sensor_shape(sample, 'gyro')
    return len(a) == len(g) == 2 and a[1] == g[1] == 3 and a[0] > 0 and g[0] > 0


def structural_fields(data: dict) -> set[str]:
    scopes = [data, data['deviceInfo'], data['extraInfo']]
    for subset in ('sync', 'training', 'testing'):
        if len(data[subset]):
            scopes.append(data[subset][0])
    return {str(key) for scope in scopes for key in scope}


def run(archive: Path, bsdtar: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    source_hash = digests(archive)
    if source_hash['md5'] != EXPECTED_MD5:
        raise ValueError('public archive MD5 mismatch')
    listed = subprocess.check_output([str(bsdtar), '-tf', str(archive)], text=True)
    members = sorted(line for line in listed.splitlines() if MEMBER.fullmatch(line))
    if len(members) != 107:
        raise ValueError(f'expected 107 userData MAT files, got {len(members)}')
    device_count = Counter()
    rate_count = Counter()
    six_axis_count = Counter()
    sync_binary_count = Counter()
    structural_sets = defaultdict(Counter)
    forward_named_fields = set()
    unit_named_fields = set()
    for index, member in enumerate(members, 1):
        content = subprocess.check_output([str(bsdtar), '-xOf', str(archive), member])
        data = loadmat(BytesIO(content), variable_names=['userData'],
                       simplify_cells=True)['userData']
        device = str(data['deviceInfo']['DeviceType'])
        rate = int(data['deviceInfo']['emgSamplingRate'])
        device_count[device] += 1
        rate_count[(device, rate)] += 1
        names = structural_fields(data)
        structural_sets[device][tuple(sorted(names))] += 1
        forward_named_fields.update(key for key in names if
                                    re.search(r'anatom|forward|body.?axis|mount|calib.?axis', key, re.I))
        unit_named_fields.update(key for key in names if re.search(r'unit|imu.?rate|gyro.?rate', key, re.I))
        for subset in ('sync', 'training', 'testing'):
            samples = data[subset]
            if not len(samples):
                raise ValueError(f'no {subset} samples in {member}')
            availability = {raw_six_axis(sample) for sample in samples}
            if len(availability) != 1:
                raise ValueError(f'mixed raw IMU availability within {member} {subset}')
            six_axis_count[(device, subset, next(iter(availability)))] += 1
        sync_binary_count[(device, all('groundTruth' in s and
                                      set(np.unique(s['groundTruth']).tolist()) <= {0, 1}
                                      for s in data['sync']))] += 1
        if index % 10 == 0 or index == len(members):
            print(f'inspected {index}/{len(members)} MAT schemas', flush=True)
    result = {
        'study': 'EPN107 public archive F6 calibration eligibility, complete MAT schema census',
        'public_record': PUBLIC_RECORD,
        'archive_bytes': archive.stat().st_size,
        'archive_hash': source_hash,
        'probe_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'member_count': len(members),
        'device_count': dict(device_count),
        'emg_sampling_rate_count': {f'{device}:{rate}Hz': count
                                    for (device, rate), count in sorted(rate_count.items())},
        'raw_accel_and_gyro_member_count': {f'{device}:{subset}:{present}': count
                                            for (device, subset, present), count
                                            in sorted(six_axis_count.items())},
        'sync_binary_groundtruth_member_count': {f'{device}:{present}': count
                                                 for (device, present), count
                                                 in sorted(sync_binary_count.items())},
        'distinct_structural_field_sets_per_device': {device: len(sets)
                                                      for device, sets in structural_sets.items()},
        'forward_axis_or_mounting_field_names': sorted(forward_named_fields),
        'imu_rate_or_unit_field_names': sorted(unit_named_fields),
        'requirements': {
            'raw_three_axis_accel_gyro_for_all_devices': all(
                present for (_, _, present) in six_axis_count),
            'explicit_device_frame_anatomical_forward_axis': bool(forward_named_fields),
            'explicit_imu_units_or_rate_in_mat': bool(unit_named_fields),
            'binary_sync_label_is_not_anatomical_axis': True,
        },
        'eligible_for_strict_calibrated_f6': False,
        'boundary': ('All 107 public MAT userData structures were examined without exporting personal values. '
                     'The public record states 50 Hz IMU but does not supply an anatomical forward-axis protocol. '
                     'Raw IMU and a binary synchronization label do not establish a measured/guided device-frame '
                     'forward vector. Empty gForce raw accel/gyro arrays further restrict coverage. '
                     'No F6 recognition performance is inferred.'),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', type=Path)
    parser.add_argument('bsdtar', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.archive, args.bsdtar, args.output)['requirements']), flush=True)
