"""Descriptive class-level readout of frozen paired EMG-bank predictions; no refit."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from emgimu.datasets.epn612 import GESTURES

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = HERE / 'EMG_F0_F7_BANK_V1_RESULTS.json'
OUT = HERE / 'EMG_F0_F7_BANK_V1_CLASS_DIAGNOSTICS.json'
TABLE = HERE / 'EMG_F0_F7_BANK_V1_CLASS_DIAGNOSTICS.csv'


def confusion(y, prediction):
    counts = np.zeros((6, 6), dtype=int)
    np.add.at(counts, (y, prediction), 1)
    return counts


def summary(counts):
    support = counts.sum(axis=1)
    predicted = counts.sum(axis=0)
    correct = np.diag(counts)
    recall = np.divide(correct, support, out=np.zeros(6), where=support > 0)
    precision = np.divide(correct, predicted, out=np.zeros(6), where=predicted > 0)
    f1 = np.divide(2 * correct, support + predicted, out=np.zeros(6), where=support + predicted > 0)
    active = counts[1:, :].sum()
    return {'confusion_true_rows_predicted_columns': counts.tolist(),
            'support': support.tolist(), 'predicted_count': predicted.tolist(),
            'recall': recall.tolist(), 'precision': precision.tolist(), 'f1': f1.tolist(),
            'macro_f1': float(f1.mean()),
            'active_trials': int(active),
            'active_accuracy': float(correct[1:].sum() / active),
            'active_predicted_rest_count': int(counts[1:, 0].sum()),
            'active_predicted_rest_rate': float(counts[1:, 0].sum() / active),
            'rest_predicted_active_count': int(counts[0, 1:].sum()),
            'rest_predicted_active_rate': float(counts[0, 1:].sum() / support[0])}


def run():
    native = json.loads(SOURCE.read_text(encoding='utf8'))
    protocol = HERE / 'EMG_F0_F7_BANK_V1_PROTOCOL.json'
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    if digest(protocol) != native['protocol_sha256']:
        raise ValueError('Frozen protocol changed')
    if digest(HERE / 'EMG_F0_F7_BANK_V1_PREDICTIONS.csv') != native['prediction_sha256']:
        raise ValueError('Frozen prediction table changed')
    cells, paired = [], []
    for shots in (0, 1, 2, 5):
        users = []
        for block in native['blocks']:
            y = np.asarray(block['labels'], dtype=int)
            base = np.asarray(block['F0_probabilities'])
            arms = {'F0': base}
            if shots:
                anchor = np.asarray(next(c for c in block['calibrations'] if c['shots'] == shots)['F7_probabilities'])
                arms.update(F7=anchor, F0_F7=(base + anchor) / 2, F0_uniform=(base + 1 / 6) / 2)
            users.append((str(block['user']), y, {name: q.argmax(1) for name, q in arms.items()}))
        pooled_y = np.concatenate([y for _, y, _ in users])
        pooled_predictions = {name: np.concatenate([p[name] for _, _, p in users]) for name in users[0][2]}
        for user, y, predictions in [('ALL', pooled_y, pooled_predictions)] + users:
            for arm, prediction in predictions.items():
                cell = {'shots_per_class': shots, 'user': user, 'arm': arm,
                        'evaluation_trials': len(y), **summary(confusion(y, prediction))}
                reference = native['scores'][str(shots)][arm]
                score = reference['pooled'] if user == 'ALL' else reference['per_user'][user]
                if abs(cell['macro_f1'] - score['macro_f1']) > 1e-12:
                    raise ValueError('Native macro-F1 mismatch')
                cells.append(cell)
            if shots:
                base_correct = predictions['F0'] == y
                full_correct = predictions['F0_F7'] == y
                for label, gesture in enumerate(GESTURES):
                    mask = y == label
                    corrected = int(np.sum(mask & ~base_correct & full_correct))
                    harmed = int(np.sum(mask & base_correct & ~full_correct))
                    both_correct = int(np.sum(mask & base_correct & full_correct))
                    both_wrong = int(np.sum(mask & ~base_correct & ~full_correct))
                    paired.append({'shots_per_class': shots, 'user': user, 'class_label': label,
                                   'native_gesture': gesture, 'support': int(mask.sum()),
                                   'base_correct': int(np.sum(mask & base_correct)),
                                   'full_correct': int(np.sum(mask & full_correct)),
                                   'corrected': corrected, 'harmed': harmed,
                                   'both_correct': both_correct, 'both_wrong': both_wrong,
                                   'net_correct_gain': corrected - harmed,
                                   'recall_gain': (corrected - harmed) / mask.sum()})
    with TABLE.open('w', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
        writer.writeheader(); writer.writerows(paired)
    payload = {'schema': 'emg_bank_class_diagnostics_v1',
               'source_sha256': {p.relative_to(ROOT).as_posix(): digest(p) for p in
                                 (SOURCE, protocol, Path(__file__), ROOT / 'src/emgimu/datasets/epn612.py')},
               'table_sha256': digest(TABLE), 'native_class_axis': list(GESTURES),
               'cells': cells, 'paired_class_changes': paired,
               'conventions': 'Independent native trials; true rows/predicted columns; argmax on fixed class axis. Active means native classes1-5. Precision is0 if no prediction of that class. No windows treated as independent trials.',
               'scope': 'Post-hoc descriptive readout of the completed EPN42-51 experiment on identical held-out trials. No new model, tuning, hypothesis test, hardware evidence or own-user failure diagnosis. Pooled class gains can hide user harms. Off-diagonal fist/open/pinch counts are native labels, not a claim about current electrode montage.',
               'default_promoted': False, 'physical_validation_proven': False,
               'completion_proven': False}
    OUT.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf8', newline='\n')
    print(f'Saved {len(cells)} class readouts and {len(paired)} paired class rows; no retraining')


if __name__ == '__main__': run()
