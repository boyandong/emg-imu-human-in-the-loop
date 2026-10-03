"""Source-only isolated F3b CES increment on frozen GRAB unseen-user Day1 trials."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmarks.grabmyo_crossday import run as grab
from benchmarks.new_bank_v2 import grab_user_run as parent_run
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.document_ces_v3 import DocumentCesFamilyV3
from emgimu.feature_bank.new_bank_v2 import RestNoiseDetailV2


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'new_bank_v2'
PROTOCOL_PATH = HERE / 'F3B_CES_GRAB_PROTOCOL.json'
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding='utf-8'))
CLASSES = np.asarray(grab.GESTURES)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(data_root: Path) -> dict:
    parent_run.check_protocol()
    for filename, key in [('GRAB_USER_PROTOCOL.json', 'parent_protocol_sha256'),
                          ('GRAB_USER_RESULTS.json', 'parent_result_sha256'),
                          ('GRAB_USER_PREDICTIONS.csv', 'parent_prediction_sha256')]:
        if sha256(PARENT / filename) != PROTOCOL[key]:
            raise AssertionError(f'frozen parent changed: {filename}')
    parent = json.loads((PARENT / 'GRAB_USER_RESULTS.json').read_text(encoding='utf-8'))
    split = parent_run.PROTOCOL
    records = [row for row in grab.records() if row['session'] == split['day']]
    if len(records) != 224:
        raise AssertionError('GRAB native Day1 inventory changed')
    manifest_sha = parent_run.check_files(data_root, records)
    if manifest_sha != parent['official_sha256_manifest_sha256']:
        raise AssertionError('GRAB official file manifest changed')
    subjects = np.asarray([row['subject'] for row in records])
    labels = np.asarray([row['gesture'] for row in records])
    source_mask = np.isin(subjects, split['source_subjects'])
    if source_mask.sum() != 112:
        raise AssertionError('GRAB source split changed')
    windows = [grab.read_record(data_root, row) for row in records]
    source_windows = np.concatenate([window for window, selected in zip(windows, source_mask) if selected])
    source_labels = np.concatenate([np.full(len(window), label) for window, label, selected
                                    in zip(windows, labels, source_mask) if selected])
    rest = np.concatenate([window for window, selected, label in zip(windows, source_mask, labels)
                           if selected and label == split['rest_code']])
    families = {
        'F0v2': RestNoiseDetailV2(rest_label=17).fit(FeatureBatch(rest, grab.RATE),
                                                     np.full(len(rest), 17)),
        'F3bCES': DocumentCesFamilyV3().fit(FeatureBatch(source_windows, grab.RATE), source_labels),
    }
    vectors = {name: np.stack([grab.aggregate(family.transform(FeatureBatch(window, grab.RATE)))
                               for window in windows]) for name, family in families.items()}
    rows = []
    scores = {}
    for arm in PROTOCOL['arms']:
        feature = np.concatenate([vectors[part] for part in arm.split('+')], axis=1)
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=1., max_iter=2000, random_state=20260924))
        model.fit(feature[source_mask], labels[source_mask])
        np.testing.assert_array_equal(model[-1].classes_, CLASSES)
        if np.max(model[-1].n_iter_) >= 2000:
            raise RuntimeError(f'classifier did not converge: {arm}')
        scores[arm] = {'source_trials': int(source_mask.sum()), 'dimension': int(feature.shape[1])}
        for phase in ('validation', 'final'):
            mask = np.isin(subjects, split[f'{phase}_subjects'])
            if mask.sum() != 56 or np.any(mask & source_mask):
                raise AssertionError('GRAB unseen-user target split changed')
            probabilities = model.predict_proba(feature[mask])
            scores[arm][phase] = grab.score(labels[mask], probabilities, CLASSES, subjects[mask])
            for record, probability in zip(np.asarray(records, dtype=object)[mask], probabilities):
                rows.append({'arm': arm, 'phase': phase, 'trial_id': record['stem'],
                             'subject': record['subject'], 'gesture': record['gesture'],
                             **{f'p_{label}': float(value) for label, value in zip(CLASSES, probability)}})
        print(f"{arm}: validation F1={scores[arm]['validation']['macro_f1']:.4f}, "
              f"final F1={scores[arm]['final']['macro_f1']:.4f}", flush=True)
    with (PARENT / 'GRAB_USER_PREDICTIONS.csv').open(newline='', encoding='utf-8') as stream:
        previous = {(row['phase'], row['trial_id']): row for row in csv.DictReader(stream)
                    if row['arm'] == 'F0v2'}
    baseline = [row for row in rows if row['arm'] == 'F0v2']
    if len(previous) != len(baseline) or len(baseline) != 112:
        raise AssertionError('parent/new baseline trial coverage changed')
    maximum = 0.
    for row in baseline:
        old = previous[(row['phase'], row['trial_id'])]
        if (row['subject'], row['gesture']) != (int(old['subject']), int(old['gesture'])):
            raise AssertionError('parent/new native trial identity changed')
        maximum = max(maximum, max(abs(row[f'p_{label}'] - float(old[f'p_{label}'])) for label in CLASSES))
    if maximum > 1e-8:
        raise AssertionError(f'GRAB F0v2 parent replay changed: {maximum}')
    path = HERE / 'F3B_CES_GRAB_PREDICTIONS.csv'
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    result = {'protocol_sha256': sha256(PROTOCOL_PATH),
              'parent_prediction_sha256': PROTOCOL['parent_prediction_sha256'],
              'official_sha256_manifest_sha256': manifest_sha,
              'source_windows': int(len(source_windows)), 'rest_windows': int(len(rest)),
              'feature_dimensions': {name: len(family.feature_names) for name, family in families.items()},
              'source_trial_ids': parent['source_trial_ids'],
              'validation_trial_ids': parent['validation_trial_ids'],
              'final_trial_ids': parent['final_trial_ids'],
              'prediction_rows': len(rows), 'prediction_sha256': sha256(path),
              'f0v2_parent_replay_max_abs_error': maximum,
              'scores': scores, 'scope': PROTOCOL['boundary']}
    (HERE / 'F3B_CES_GRAB_RESULTS.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path,
                        default=Path('D:/emg-imu-benchmarks/data/raw/grabmyo_crossday_subset_v1'))
    args = parser.parse_args()
    run(args.data_root)
