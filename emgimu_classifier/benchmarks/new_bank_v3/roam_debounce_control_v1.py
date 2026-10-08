"""Predeclared paired label control on immutable chronological emissions."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from emgimu.feature_bank.causal_label_debounce_v1 import CausalLabelDebounceV1
from emgimu.feature_bank.causal_window_recognition_v1 import transition_hold_diagnostics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROTOCOL = HERE / 'ROAM_DEBOUNCE_CONTROL_V1_PROTOCOL.json'
RESULT = HERE / 'ROAM_DEBOUNCE_CONTROL_V1_RESULTS.json'
TABLE = HERE / 'ROAM_DEBOUNCE_CONTROL_V1_PAIRED.csv'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    if PROTOCOL.exists(): raise FileExistsError('Protocol already frozen')
    paths = [Path(__file__), ROOT / 'src/emgimu/feature_bank/causal_label_debounce_v1.py',
             ROOT / 'src/emgimu/feature_bank/causal_window_recognition_v1.py',
             HERE / 'ROAM_CAUSAL_WINDOW_V1_PROTOCOL.json', HERE / 'ROAM_CAUSAL_WINDOW_V1_RESULTS.json',
             HERE / 'ROAM_CAUSAL_WINDOW_V1_EMISSIONS.csv']
    p = {'schema': 'roam_debounce_control_v1', 'confirmations': 2, 'classes': [0, 1, 2],
         'rate_hz': 200, 'hop_samples': 10, 'reaction_half_buffer_samples': 100,
         'policy': 'Confirm two consecutive emission labels before changing; start unknown; reset pending candidate on stable label. Fresh state per native recording.',
         'evaluation': 'Paired class metrics on shared known samples; additionally full-record accuracy with unknown wrong. All transitions retain unknown failures. No probability metrics for a label-only policy.',
         'scope': 'Fixed descriptive control on previously inspected public cohorts, no fit or parameter search, no hardware timing or own-device efficacy proof. Nominal added confirmation interval 50ms for immediately consistent predictions; no guaranteed event delay.',
         'default_promoted': False,
         'source_sha256': {path.relative_to(ROOT).as_posix(): sha(path) for path in paths}}
    PROTOCOL.write_text(json.dumps(p, indent=2) + '\n', encoding='utf8', newline='\n')
    print('Frozen two-confirmation protocol; no replay performed')


def expand(runs, size):
    out = np.full(size, -999, dtype=int)
    cursor = 0
    for start, stop, value in runs:
        if start != cursor or not start < stop <= size: raise ValueError('Invalid complete RLE')
        out[start:stop] = value; cursor = stop
    if cursor != size: raise ValueError('Incomplete RLE')
    return out


def rle(values):
    bounds = np.r_[0, np.flatnonzero(np.diff(values)) + 1, len(values)]
    return [[int(a), int(b), int(values[a])] for a, b in zip(bounds[:-1], bounds[1:])]


def scores(y, called):
    f = []
    for c in range(3):
        tp = np.sum((y == c) & (called == c)); den = np.sum(y == c) + np.sum(called == c)
        f.append(float(2 * tp / den) if den else 0.0)
    return {'accuracy': float(np.mean(y == called)), 'macro_f1': float(np.mean(f))}


def run():
    if RESULT.exists() or TABLE.exists(): raise FileExistsError('Refuse completed replay overwrite')
    p = json.loads(PROTOCOL.read_text(encoding='utf8'))
    for path, digest in p['source_sha256'].items():
        if sha(ROOT / path) != digest: raise ValueError('Frozen input or implementation changed')
    parent = json.loads((HERE / 'ROAM_CAUSAL_WINDOW_V1_RESULTS.json').read_text(encoding='utf8'))
    grouped = defaultdict(list)
    with (HERE / 'ROAM_CAUSAL_WINDOW_V1_EMISSIONS.csv').open(encoding='utf8', newline='') as handle:
        for row in csv.DictReader(handle): grouped[row['native_file']].append(row)
    records = []; rows = []
    if set(grouped) != {r['native_file'] for r in parent['records']}: raise ValueError('File coverage differs')
    for old in parent['records']:
        n = old['samples']; y = expand(old['truth_rle'], n); raw = expand(old['prediction_rle'], n)
        emissions = grouped[old['native_file']]
        ends = np.array([int(row['emission_sample']) for row in emissions])
        if not np.array_equal(ends, np.arange(39, n, 10)): raise ValueError('Emission cadence changed')
        control = CausalLabelDebounceV1(p['classes'], p['confirmations'])
        called = np.full(n, -1, dtype=int)
        for i, (end, row) in enumerate(zip(ends, emissions)):
            label = int(np.argmax([float(row[f'p_{c}']) for c in p['classes']]))
            stop = int(ends[i + 1]) if i + 1 < len(ends) else n
            if np.any(raw[end:stop] != label): raise ValueError('Baseline differs from saved probabilities')
            called[end:stop] = control.update(label)
        shared = (raw >= 0) & (called >= 0)
        if not np.any(shared): raise ValueError('No paired known samples')
        arms = {}
        for arm, prediction in [('raw', raw), ('confirmed', called)]:
            arms[arm] = {'shared_known_scores': scores(y[shared], prediction[shared]),
                         'full_record_accuracy_unknown_wrong': float(np.mean(y == prediction)),
                         'unknown_samples': int(np.sum(prediction < 0)),
                         'transition_hold': transition_hold_diagnostics(y, prediction, rate_hz=p['rate_hz'], half_buffer_samples=p['reaction_half_buffer_samples']),
                         'log_loss': None, 'brier': None, 'ece': None}
        record = {key: old[key] for key in ('native_file', 'native_sha256', 'user', 'posture', 'phase', 'samples')}
        record.update(shared_known_samples=int(shared.sum()), confirmed_prediction_rle=rle(called), arms=arms)
        records.append(record)
        row = {key: record[key] for key in ('native_file', 'user', 'phase', 'posture', 'samples', 'shared_known_samples')}
        for arm, info in arms.items():
            row.update({f'{arm}_{key}': value for key, value in info['shared_known_scores'].items()})
            row.update({f'{arm}_full_record_accuracy': info['full_record_accuracy_unknown_wrong'],
                        f'{arm}_unknown_samples': info['unknown_samples'],
                        f'{arm}_hold_correct': info['transition_hold']['correct_transitions'],
                        f'{arm}_hold_eligible': info['transition_hold']['eligible_transitions'],
                        f'{arm}_maintenance_switches': info['transition_hold']['maintenance_switches']})
        rows.append(row)
    summaries = {}
    for phase in ('validation', 'descriptive_final'):
        summaries[phase] = {}
        group = [r for r in records if r['phase'] == phase]
        for arm in ('raw', 'confirmed'):
            infos = [r['arms'][arm] for r in group]
            eligible = sum(i['transition_hold']['eligible_transitions'] for i in infos)
            correct = sum(i['transition_hold']['correct_transitions'] for i in infos)
            summaries[phase][arm] = {'recordings': len(group),
                'equal_recording_shared_macro_f1': float(np.mean([i['shared_known_scores']['macro_f1'] for i in infos])),
                'equal_recording_full_accuracy': float(np.mean([i['full_record_accuracy_unknown_wrong'] for i in infos])),
                'unknown_samples': sum(i['unknown_samples'] for i in infos),
                'eligible_transitions': eligible, 'correct_transitions': correct,
                'transition_hold_accuracy': correct / eligible if eligible else None,
                'maintenance_switches': sum(i['transition_hold']['maintenance_switches'] for i in infos)}
    with TABLE.open('w', encoding='utf8', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator='\n'); writer.writeheader(); writer.writerows(rows)
    for path, digest in p['source_sha256'].items():
        if sha(ROOT / path) != digest: raise ValueError('Replay modified frozen inputs')
    result = {'schema': p['schema'], 'protocol_sha256': sha(PROTOCOL), 'table_sha256': sha(TABLE),
              'records': records, 'summaries': summaries, 'input_immutable': True,
              'default_promoted': False, 'physical_validation_proven': False, 'scope': p['scope']}
    RESULT.write_text(json.dumps(result, indent=2) + '\n', encoding='utf8', newline='\n')
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true'); args = parser.parse_args()
    prepare() if args.prepare else run()
