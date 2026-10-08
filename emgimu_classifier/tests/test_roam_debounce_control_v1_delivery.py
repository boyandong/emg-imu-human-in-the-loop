import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'benchmarks/new_bank_v3'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def dense(runs, n):
    out = np.full(n, -999, dtype=int); end = 0
    for a, b, c in runs:
        assert a == end and a < b <= n
        out[a:b] = c; end = b
    assert end == n
    return out


def test_paired_replay_independent_confirmation_metrics_and_unknown_accounting():
    p = json.loads((HERE / 'ROAM_DEBOUNCE_CONTROL_V1_PROTOCOL.json').read_text(encoding='utf8'))
    r = json.loads((HERE / 'ROAM_DEBOUNCE_CONTROL_V1_RESULTS.json').read_text(encoding='utf8'))
    parent = json.loads((HERE / 'ROAM_CAUSAL_WINDOW_V1_RESULTS.json').read_text(encoding='utf8'))
    assert p['confirmations'] == 2 and p['classes'] == [0, 1, 2]
    assert r['protocol_sha256'] == sha(HERE / 'ROAM_DEBOUNCE_CONTROL_V1_PROTOCOL.json')
    assert r['table_sha256'] == sha(HERE / 'ROAM_DEBOUNCE_CONTROL_V1_PAIRED.csv')
    for path, digest in p['source_sha256'].items(): assert sha(ROOT / path) == digest
    assert r['input_immutable'] and not r['default_promoted'] and not r['physical_validation_proven']
    grouped = defaultdict(list)
    with (HERE / 'ROAM_CAUSAL_WINDOW_V1_EMISSIONS.csv').open(encoding='utf8', newline='') as handle:
        for row in csv.DictReader(handle): grouped[row['native_file']].append(row)
    assert len(r['records']) == len(parent['records']) == 40
    assert [v['native_file'] for v in r['records']] == [v['native_file'] for v in parent['records']]
    for record, old in zip(r['records'], parent['records']):
        n = old['samples']; y = dense(old['truth_rle'], n); raw = dense(old['prediction_rle'], n)
        confirmed = dense(record['confirmed_prediction_rle'], n)
        emissions = grouped[old['native_file']]
        ends = [int(e['emission_sample']) for e in emissions]
        labels = [int(np.argmax([float(e[f'p_{c}']) for c in range(3)])) for e in emissions]
        # Independent fixed-two policy: stable changes only on adjacent equal
        # labels. No pending counter or production policy call is used here.
        expected = np.full(n, -1, dtype=int); stable = -1
        for i, end in enumerate(ends):
            if i > 0 and labels[i] == labels[i - 1]: stable = labels[i]
            stop = ends[i + 1] if i + 1 < len(ends) else n
            expected[end:stop] = stable
        np.testing.assert_array_equal(confirmed, expected)
        shared = (raw >= 0) & (confirmed >= 0)
        assert shared.sum() == record['shared_known_samples']
        assert np.all(confirmed[:49] == -1)
        for arm, prediction in [('raw', raw), ('confirmed', confirmed)]:
            info = record['arms'][arm]
            assert all(info[k] is None for k in ('log_loss', 'brier', 'ece'))
            assert info['unknown_samples'] == np.sum(prediction < 0)
            assert abs(info['full_record_accuracy_unknown_wrong'] - np.mean(y == prediction)) < 1e-12
            yy = y[shared]; called = prediction[shared]
            f = []
            for c in range(3):
                tp = np.sum((yy == c) & (called == c)); den = np.sum(yy == c) + np.sum(called == c)
                f.append(2 * tp / den if den else 0)
            assert abs(info['shared_known_scores']['macro_f1'] - np.mean(f)) < 1e-12
            assert abs(info['shared_known_scores']['accuracy'] - np.mean(yy == called)) < 1e-12
            diag = info['transition_hold']; correct = 0; switches = 0
            onsets = np.flatnonzero(np.diff(y)) + 1
            assert len(onsets) == len(diag['events']) == 8
            for i, (event, onset) in enumerate(zip(diag['events'], onsets)):
                stop = int(onsets[i + 1] - 100) if i + 1 < len(onsets) else n
                assert event['eligible'] and event['reaction_start'] == onset - 100
                assert event['reaction_stop'] == onset + 100 and event['maintenance_stop'] == stop
                reaction = prediction[onset - 100:onset + 100]; hold = prediction[onset + 100:stop]
                ok = bool(y[onset] in reaction and set(reaction) <= {y[onset - 1], y[onset]} and np.all(hold == y[onset]))
                errors = np.sum(hold != y[onset]); flicker = np.sum(hold[1:] != hold[:-1])
                assert event['correct'] == ok and event['maintenance_error_samples'] == errors
                assert event['maintenance_switches'] == flicker
                correct += ok; switches += int(flicker)
            assert diag['correct_transitions'] == correct and diag['maintenance_switches'] == switches
            assert diag['eligible_transitions'] == 8
    for phase, arms in r['summaries'].items():
        group = [v for v in r['records'] if v['phase'] == phase]
        assert len(group) == 20
        for arm, summary in arms.items():
            infos = [v['arms'][arm] for v in group]
            assert summary['unknown_samples'] == sum(i['unknown_samples'] for i in infos)
            assert summary['maintenance_switches'] == sum(i['transition_hold']['maintenance_switches'] for i in infos)
            assert summary['correct_transitions'] == sum(i['transition_hold']['correct_transitions'] for i in infos)
            assert summary['eligible_transitions'] == 160
            assert summary['transition_hold_accuracy'] == summary['correct_transitions'] / 160
            assert abs(summary['equal_recording_shared_macro_f1'] - np.mean([i['shared_known_scores']['macro_f1'] for i in infos])) < 1e-12
            assert abs(summary['equal_recording_full_accuracy'] - np.mean([i['full_record_accuracy_unknown_wrong'] for i in infos])) < 1e-12
