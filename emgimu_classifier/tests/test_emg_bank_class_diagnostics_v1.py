import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'benchmarks/new_bank_v3'


def test_class_diagnostics_match_frozen_native_predictions_and_class_axis():
    d = json.loads((HERE / 'EMG_F0_F7_BANK_V1_CLASS_DIAGNOSTICS.json').read_text(encoding='utf8'))
    r = json.loads((HERE / 'EMG_F0_F7_BANK_V1_RESULTS.json').read_text(encoding='utf8'))
    for path, digest in d['source_sha256'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert d['native_class_axis'] == ['noGesture', 'fist', 'waveIn', 'waveOut', 'open', 'pinch']
    assert len(d['cells']) == 143
    assert not d['default_promoted'] and not d['physical_validation_proven'] and not d['completion_proven']
    for cell in d['cells']:
        labels, probabilities = [], []
        for block in r['blocks']:
            if cell['user'] != 'ALL' and str(block['user']) != cell['user']: continue
            q = np.array(block['F0_probabilities'])
            if cell['arm'] == 'F0_uniform': q = .5 * q + .5 / 6
            elif cell['arm'] != 'F0':
                anchor = np.array(next(c for c in block['calibrations'] if c['shots'] == cell['shots_per_class'])['F7_probabilities'])
                q = anchor if cell['arm'] == 'F7' else .5 * q + .5 * anchor
            labels.extend(block['labels']); probabilities.extend(q)
        y = np.array(labels); prediction = np.array(probabilities).argmax(1)
        matrix = confusion_matrix(y, prediction, labels=range(6))
        np.testing.assert_array_equal(matrix, cell['confusion_true_rows_predicted_columns'])
        p, recall, f1, support = precision_recall_fscore_support(y, prediction, labels=range(6), zero_division=0)
        for key, expected in [('precision', p), ('recall', recall), ('f1', f1), ('support', support)]:
            np.testing.assert_allclose(cell[key], expected, atol=1e-12)
        assert cell['evaluation_trials'] == len(y)
        assert cell['active_predicted_rest_count'] == int(np.sum((y != 0) & (prediction == 0)))
        assert cell['rest_predicted_active_count'] == int(np.sum((y == 0) & (prediction != 0)))
        assert abs(cell['active_accuracy'] - np.mean(y[y != 0] == prediction[y != 0])) < 1e-12


def test_paired_class_correction_harm_partitions_match_both_confusions_and_table():
    d = json.loads((HERE / 'EMG_F0_F7_BANK_V1_CLASS_DIAGNOSTICS.json').read_text(encoding='utf8'))
    native = json.loads((HERE / 'EMG_F0_F7_BANK_V1_RESULTS.json').read_text(encoding='utf8'))
    partitions = {}
    for shots in (1, 2, 5):
        for block in native['blocks']:
            y = np.array(block['labels'])
            base = np.array(block['F0_probabilities'])
            anchor = np.array(next(c for c in block['calibrations'] if c['shots'] == shots)['F7_probabilities'])
            a = base.argmax(1) == y; b = (base + anchor).argmax(1) == y
            for label in range(6):
                mask = y == label
                counts = np.array([sum(mask & ~a & b), sum(mask & a & ~b), sum(mask & a & b), sum(mask & ~a & ~b)])
                partitions[shots, str(block['user']), label] = counts
                key = (shots, 'ALL', label)
                partitions[key] = partitions.get(key, np.zeros(4, dtype=int)) + counts
    table = HERE / 'EMG_F0_F7_BANK_V1_CLASS_DIAGNOSTICS.csv'
    assert hashlib.sha256(table.read_bytes()).hexdigest() == d['table_sha256']
    with table.open(encoding='utf8', newline='') as stream: rows = list(csv.DictReader(stream))
    assert len(rows) == len(d['paired_class_changes']) == 198
    cells = {(c['shots_per_class'], c['user'], c['arm']): c for c in d['cells']}
    for row, item in zip(rows, d['paired_class_changes']):
        for key, value in item.items():
            assert float(row[key]) == value if isinstance(value, (int, float)) else row[key] == value
        shots, user, label = item['shots_per_class'], item['user'], item['class_label']
        np.testing.assert_array_equal([item[k] for k in ('corrected', 'harmed', 'both_correct', 'both_wrong')], partitions[shots, user, label])
        a = cells[shots, user, 'F0']; b = cells[shots, user, 'F0_F7']
        assert item['native_gesture'] == d['native_class_axis'][label]
        assert item['support'] == a['support'][label] == b['support'][label]
        assert item['support'] == sum(item[k] for k in ('corrected', 'harmed', 'both_correct', 'both_wrong'))
        assert item['base_correct'] == item['harmed'] + item['both_correct'] == a['confusion_true_rows_predicted_columns'][label][label]
        assert item['full_correct'] == item['corrected'] + item['both_correct'] == b['confusion_true_rows_predicted_columns'][label][label]
        assert item['net_correct_gain'] == item['corrected'] - item['harmed']
        assert abs(item['recall_gain'] - (b['recall'][label] - a['recall'][label])) < 1e-12
