import csv
import hashlib
import json
import zipfile
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'benchmarks/new_bank_v3'


def test_native_calibration_ledger_selection_and_method_specific_duration_sums():
    d=json.loads((HERE/'EMG_CALIBRATION_BURDEN_V1.json').read_text(encoding='utf8'))
    for path,digest in d['source_sha256'].items(): assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
    table=HERE/'EMG_CALIBRATION_BURDEN_V1.csv'
    assert hashlib.sha256(table.read_bytes()).hexdigest()==d['table_sha256']
    with table.open(encoding='utf8',newline='') as stream:rows=list(csv.DictReader(stream))
    assert len(rows)==len(d['records'])==690
    ledger=d['native_reserved_trial_ledger'];assert len(ledger)==600
    assert not d['few_second_calibration_proven'] and not d['physical_wall_time_proven']
    for trial,item in ledger.items():
        assert item['sample_rate_hz']==200 and item['channels']==8 and 1<=item['represented_windows']<=4
        assert item['full_recording_seconds']==item['samples_per_channel']/200
        assert item['represented_signal_seconds']==item['represented_windows']*40/200
        assert item['full_recording_seconds']>=item['represented_signal_seconds']
    selections={(s['run_id'],s['subject'],s['shots_per_class']):s for s in d['calibration_selections']}
    assert len(selections)==80
    for row,native in zip(rows,d['records']):
        for key,value in native.items():
            assert row[key]==str(value)
        key=(native['run_id'],native['subject'],native['shots_per_class']);s=selections[key]
        assert len(s['calibration_ids'])==len(set(s['calibration_ids']))==6*native['shots_per_class']
        assert len(set(s['reserved_ids']))==30 and len(set(s['evaluation_ids']))==120
        assert not set(s['reserved_ids'])&set(s['evaluation_ids'])
        assert set(s['calibration_ids'])<=set(s['reserved_ids'])
        arm=native['feature_bank']
        used=s['calibration_ids'] if (arm in ('F7','F0_F7') if key[0]=='EMG_F0_F7_BANK_V1' else arm.startswith('reliability')) else []
        assert native['used_calibration_trials']==len(used)
        assert native['used_calibration_windows']==sum(ledger[t]['represented_windows'] for t in used)
        assert abs(native['used_signal_seconds']-sum(ledger[t]['represented_signal_seconds'] for t in used))<1e-12
        assert abs(native['used_trials_full_recording_seconds']-sum(ledger[t]['full_recording_seconds'] for t in used))<1e-12
        assert abs(native['reserved_full_recording_seconds']-sum(ledger[t]['full_recording_seconds'] for t in s['reserved_ids']))<1e-12
        assert native['requires_all_native_gestures']==bool(used)
        assert all(native[k]=='N/A' for k in ('device_wall_time_seconds','requires_target_force','requires_target_posture','requires_each_session'))
        if used: np.testing.assert_array_equal(np.bincount([ledger[t]['class_label'] for t in used]),[native['shots_per_class']]*6)


def test_recording_duration_and_gesture_labels_match_actual_public_archive():
    d=json.loads((HERE/'EMG_CALIBRATION_BURDEN_V1.json').read_text(encoding='utf8'))
    p=json.loads((HERE/'EMG_F0_F7_BANK_V1_PROTOCOL.json').read_text(encoding='utf8'))
    archive=Path(p['archive'])
    if not archive.exists(): pytest.skip('Native public archive is not bundled with Git results')
    names=('noGesture','fist','waveIn','waveOut','open','pinch')
    ledger=d['native_reserved_trial_ledger']
    with zipfile.ZipFile(archive) as handle:
        for user in sorted({v['subject'] for v in ledger.values()}):
            raw=json.loads(handle.read(f'EMG-EPN612 Dataset/trainingJSON/user{user}/user{user}.json'))
            for trial,item in ledger.items():
                if item['subject']!=user:continue
                sample=raw['trainingSamples'][trial.split(':',2)[2]]
                assert raw['generalInfo']['samplingFrequencyInHertz']==item['sample_rate_hz']
                assert all(len(sample['emg'][f'ch{c}'])==item['samples_per_channel'] for c in range(1,9))
                assert sample['gestureName']==item['native_gesture']==names[item['class_label']]


def test_canonical_cost_table_preserves_method_costs_and_reserved_unused_trials():
    d=json.loads((HERE/'EMG_CALIBRATION_BURDEN_V1.json').read_text(encoding='utf8'))
    with (ROOT/'feature_bank/delivery/new_bank_v3/calibration_burden.csv').open(encoding='utf8',newline='') as stream:
        rows=[r for r in csv.DictReader(stream) if r['run_id'] in ('EMG_F0_F7_BANK_V1','EMG_CALIBRATED_FUSION_V1')]
    assert len(rows)==690
    lookup={(r['run_id'],int(r['subject']),int(r['shots_per_class']),r['feature_bank']):r for r in rows}
    assert len(lookup)==len(rows)
    for native in d['records']:
        row=lookup[native['run_id'],native['subject'],native['shots_per_class'],native['feature_bank']]
        for key,value in native.items():assert row[key]==str(value)
        assert row['source_artifact']=='benchmarks/new_bank_v3/EMG_CALIBRATION_BURDEN_V1.json'
        assert row['source_sha256']==hashlib.sha256((HERE/'EMG_CALIBRATION_BURDEN_V1.json').read_bytes()).hexdigest()
