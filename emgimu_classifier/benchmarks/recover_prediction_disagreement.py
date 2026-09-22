"""Recover true prediction disagreement for two frozen held-out pair matrices."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

RUNS = (
    'feature_bank_epn_selection_20260915',
    'feature_bank_force_audited_validation_20260915',
)
DIAGNOSTIC_RUNS = (
    'feature_bank_epn_calibration_diagnostics_validation_20260915_v2',
    'feature_bank_epn_calibration_diagnostics_final_20260915_v2',
    'feature_bank_manus_calibration_diagnostics_validation_20260915',
    'feature_bank_manus_calibration_diagnostics_final_20260915',
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def record_hash(row: dict) -> str:
    return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def check_source_rates(row: dict, truth: np.ndarray, a: np.ndarray,
                       b: np.ndarray, weights: np.ndarray, context: str) -> float:
    correctness = float(np.average((a != truth) != (b != truth), weights=weights))
    asymmetric_ab = float(np.average((a == truth) & (b != truth), weights=weights))
    asymmetric_ba = float(np.average((a != truth) & (b == truth), weights=weights))
    for label, actual in (('disagreement', correctness),
                          ('a_correct_b_wrong', asymmetric_ab),
                          ('a_wrong_b_correct', asymmetric_ba)):
        if not np.isclose(float(row[label]), actual, rtol=0, atol=1e-12):
            raise ValueError(f'saved predictions do not replay {label}: {context}')
    return float(np.average(a != b, weights=weights))


def recover_diagnostics(rows: list[dict], processed: Path, raw_root: Path) -> tuple[list[dict], dict]:
    from emgimu.datasets.epn612 import load_epn612_windows
    from emgimu.datasets.semg_manus import load_semg_manus_windows
    from emgimu.feature_bank.epn_study import aggregate_trials
    from emgimu.feature_bank.manus_study import GESTURES, _aggregate

    records, source_hashes, archive_hashes = [], {}, {}
    for run in DIAGNOSTIC_RUNS:
        diag_root = processed / run
        manifest = json.loads((diag_root / 'run_manifest.json').read_text(encoding='utf-8'))
        source_root = processed / manifest['source_run']
        state_path, split_path = source_root / 'fitted_states.pkl', source_root / 'split_trial_ids.json'
        if manifest['fit_performed'] or sha(state_path) != manifest['state_sha256']:
            raise ValueError(f'diagnostic source state is invalid: {run}')
        dataset = manifest.get('dataset') or ('semg_manus' if run.startswith('feature_bank_manus_') else None)
        phase = manifest['phase']
        if dataset == 'epn612':
            users = (16, 17, 18) if phase == 'validation' else (19, 20, 21)
            archive = raw_root / 'epn612/EMG-EPN612-Dataset.zip'
            target = load_epn612_windows(archive, users=users)
            def aggregate(values):
                _, labels, _, trials, _ = aggregate_trials(values, target)
                return labels, trials
            def features(values):
                trial_features, _, _, _, _ = aggregate_trials(values, target)
                return trial_features
        elif dataset == 'semg_manus':
            users = tuple(range(3, 9))
            session = 2 if phase == 'validation' else 3
            archive = raw_root / 'semg_manus/semg-manus-dataset-v1.zip'
            target = load_semg_manus_windows(archive, users=users, sessions=(session,), gestures=GESTURES)
            def aggregate(values):
                _, labels, _, _, _, trials = _aggregate(values, target)
                return labels, trials
            def features(values):
                trial_features, _, _, _, _, _ = _aggregate(values, target)
                return trial_features
        else:
            raise ValueError(f'unsupported diagnostic dataset: {dataset}')
        states, anchors = pickle.loads(state_path.read_bytes())
        splits = json.loads(split_path.read_text(encoding='utf-8'))
        transformed, raw = {}, {}
        labels = trials = None
        for name, (family, scaler, model) in states.items():
            window_features = family.transform(target.batch)
            trial_features = features(window_features)
            if labels is None:
                labels, trials = aggregate(window_features)
            transformed[name] = scaler.transform(trial_features)
            raw[name] = model.predict_proba(transformed[name])
        source_rows = [(index, row) for index, row in enumerate(rows, 1) if row['run_id'] == run]
        lookup = {(int(row['subject']), int(row['shots_per_class']), row['family_a'], row['family_b']): (index, row)
                  for index, row in source_rows}
        if len(lookup) != len(source_rows):
            raise ValueError(f'duplicate diagnostic pair key: {run}')
        consumed = set()
        for split in splits:
            user, shots = int(split['user']), int(split['shots'])
            if user not in users or set(split['calibration']) & set(split['evaluation']):
                raise ValueError(f'invalid diagnostic trial split: {run}/{user}/{shots}')
            ev = np.flatnonzero(np.isin(trials, split['evaluation']))
            if not len(ev) or set(trials[ev]) != set(split['evaluation']):
                raise ValueError(f'incomplete diagnostic trial split: {run}/{user}/{shots}')
            predictions = {}
            for name in states:
                base = raw[name][ev]
                calibrated = base
                if shots:
                    anchor, temperature = anchors[(user, shots, name)]
                    logits = -anchor.transform(transformed[name][ev])[:, :6].astype(float) / temperature
                    logits -= logits.max(1, keepdims=True)
                    q = np.exp(logits)
                    q /= q.sum(1, keepdims=True)
                    calibrated = (1 - shots / (shots + 2)) * base + (shots / (shots + 2)) * q
                predictions[name + '_raw'] = base.argmax(axis=1)
                predictions[name + '_anchor'] = calibrated.argmax(axis=1)
            for a_name, b_name in (('F1_X1H_raw', 'F4_Spectral_raw'),
                                   ('F1_X1H_raw', 'F2b_CSP_raw'),
                                   ('F3_Ring_raw', 'F3_Ring_anchor'),
                                   ('F0_raw', 'F3_Ring_raw'),
                                   ('F5_Temporal_raw', 'F3_Ring_anchor')):
                key = (user, shots, a_name, b_name)
                if key not in lookup:
                    raise ValueError(f'missing diagnostic source pair: {run}/{key}')
                index, row = lookup[key]
                rate = check_source_rates(row, labels[ev], predictions[a_name], predictions[b_name],
                                          np.ones(len(ev)), f'{run}/{index}')
                records.append({'run_id': run, 'source_row_1based': index,
                                'source_record_sha256': record_hash(row),
                                'prediction_disagreement_rate': rate})
                consumed.add(key)
        if consumed != set(lookup):
            raise ValueError(f'diagnostic source pair coverage is incomplete: {run}')
        if archive not in archive_hashes:
            archive_hashes[archive] = sha(archive)
        source_hashes[run] = {'fitted_states.pkl': sha(state_path),
                              'split_trial_ids.json': sha(split_path),
                              'raw_archive': archive_hashes[archive]}
        print(f'{run}: replayed {len(source_rows)} diagnostic pairs', flush=True)
    return records, source_hashes


def recover(results: Path, processed: Path, output: Path, raw_root: Path) -> dict:
    source_path = results / 'error_complementarity.csv'
    with source_path.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    records, source_hashes = [], {}
    for run in RUNS:
        npz_path = processed / run / 'heldout_predictions.npz'
        source_hashes[run] = sha(npz_path)
        with np.load(npz_path, allow_pickle=False) as saved:
            truth = saved['labels']
            weights = saved['weights'] if 'weights' in saved else np.ones(len(truth))
            if len(weights) != len(truth) or np.any(weights <= 0):
                raise ValueError(f'invalid saved trial weights: {run}')
            matched = [(index, row) for index, row in enumerate(rows, 1)
                       if row['run_id'] == run]
            for index, row in matched:
                a_name, b_name = row['family_a'], row['family_b']
                if a_name not in saved or b_name not in saved:
                    raise ValueError(f'missing saved pair predictions: {run}/{index}')
                a = saved[a_name].argmax(axis=1)
                b = saved[b_name].argmax(axis=1)
                if a.shape != truth.shape or b.shape != truth.shape:
                    raise ValueError(f'misaligned predictions: {run}/{index}')
                rate = check_source_rates(row, truth, a, b, weights, f'{run}/{index}')
                records.append({'run_id': run, 'source_row_1based': index,
                                'source_record_sha256': record_hash(row),
                                'prediction_disagreement_rate': rate})
    diagnostic_records, diagnostic_hashes = recover_diagnostics(rows, processed, raw_root)
    records.extend(diagnostic_records)
    if len(records) != 402 or len({(r['run_id'], r['source_row_1based']) for r in records}) != 402:
        raise ValueError('expected 402 unique recoverable source rows')
    artifact = {'scope': 'frozen native held-out predictions; no refit',
                'source_csv_sha256': sha(source_path),
                'source_npz_sha256': source_hashes,
                'diagnostic_input_sha256': diagnostic_hashes,
                'correctness_and_asymmetric_rates_replayed': True,
                'rows': records}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2) + '\n', encoding='utf-8')
    return artifact


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results', type=Path)
    parser.add_argument('processed', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--raw-root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({'rows': len(recover(args.results, args.processed, args.output,
                                          args.raw_root)['rows'])}))
