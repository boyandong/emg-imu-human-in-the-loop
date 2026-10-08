"""Account actual calibration windows separately from complete recording time."""
import json
import zipfile
from pathlib import Path
import numpy as np
from benchmarks.new_bank_v3.mahalanobis_epn_budget_v1 import sha
from emgimu.datasets.epn612 import load_epn612_windows

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def run():
    protocol = HERE / 'MAHALANOBIS_EPN_HOLDOUT_V2_PROTOCOL.json'
    result = HERE / 'MAHALANOBIS_EPN_HOLDOUT_V2_RESULTS.json'
    p, r = (json.loads(path.read_text()) for path in (protocol, result))
    archive = Path(p['archive'])
    if sha(archive) != p['archive_sha256']:
        raise ValueError('Native archive changed')
    data = load_epn612_windows(archive, users=p['target_users'])
    ids, counts = np.unique(data.trials, return_counts=True)
    represented = dict(zip(ids.tolist(), counts.tolist()))
    seconds_per_window = data.batch.emg.shape[1] / data.batch.sample_rate_hz
    full_seconds = {}
    with zipfile.ZipFile(archive) as handle:
        for user in p['target_users']:
            native = json.loads(handle.read(f'EMG-EPN612 Dataset/trainingJSON/user{user}/user{user}.json'))
            rate = native['generalInfo']['samplingFrequencyInHertz']
            for name, sample in native['trainingSamples'].items():
                full_seconds[f'trainingJSON:user{user}:{name}'] = len(sample['emg']['ch1']) / rate
    rows = []
    for b in r['blocks']:
        reserved = next(block['calibration_ids'] for block in r['blocks']
                        if block['user'] == b['user'] and block['shots'] == 20)
        windows = sum(represented[trial] for trial in b['calibration_ids'])
        rows.append({'dataset': 'EPN612', 'subject': b['user'], 'shots_per_class': b['shots'],
                     'classes': 6, 'used_calibration_trials': len(b['calibration_ids']),
                     'used_calibration_windows': windows,
                     'window_samples': data.batch.emg.shape[1], 'sample_rate_hz': data.batch.sample_rate_hz,
                     'used_signal_seconds': windows * seconds_per_window,
                     'used_trials_full_recording_seconds': sum(full_seconds[trial] for trial in b['calibration_ids']),
                     'reserved_calibration_trials': len(reserved),
                     'reserved_signal_seconds': sum(represented[trial] for trial in reserved) * seconds_per_window,
                     'hardware_setup_seconds': 'N/A', 'guided_prompt_rest_seconds': 'N/A',
                     'device_wall_time_seconds': 'N/A', 'scope': 'Extracted signal exposure and complete stored EMG durations; not physical setup or elapsed session time.'})
    evidence = [protocol, result, Path(__file__), ROOT / 'src/emgimu/datasets/epn612.py']
    payload = {'schema': 'epn_holdout_burden_v2', 'archive_sha256': p['archive_sha256'],
               'source_sha256': {path.relative_to(ROOT).as_posix(): sha(path) for path in evidence},
               'records': rows, 'few_second_calibration_proven': False,
               'physical_wall_time_proven': False}
    (HERE / 'MAHALANOBIS_EPN_HOLDOUT_V2_BURDEN.json').write_text(json.dumps(payload, indent=2) + '\n', encoding='utf8')
    print(json.dumps({'records': len(rows), 'used_signal_seconds_by_budget': {
        str(shots): sorted({row['used_signal_seconds'] for row in rows if row['shots_per_class'] == shots})
        for shots in p['budgets']}}), flush=True)


if __name__ == '__main__':
    run()
