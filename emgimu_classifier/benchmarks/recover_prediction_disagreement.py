"""Recover true prediction disagreement for two frozen held-out pair matrices."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

RUNS = (
    'feature_bank_epn_selection_20260915',
    'feature_bank_force_audited_validation_20260915',
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_hash(row: dict) -> str:
    return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def recover(results: Path, processed: Path, output: Path) -> dict:
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
                correctness = float(np.average((a != truth) != (b != truth), weights=weights))
                asymmetric_ab = float(np.average((a == truth) & (b != truth), weights=weights))
                asymmetric_ba = float(np.average((a != truth) & (b == truth), weights=weights))
                for label, actual in (('disagreement', correctness),
                                      ('a_correct_b_wrong', asymmetric_ab),
                                      ('a_wrong_b_correct', asymmetric_ba)):
                    if not np.isclose(float(row[label]), actual, rtol=0, atol=1e-12):
                        raise ValueError(f'saved predictions do not replay {label}: {run}/{index}')
                records.append({'run_id': run, 'source_row_1based': index,
                                'source_record_sha256': record_hash(row),
                                'prediction_disagreement_rate': float(np.average(a != b, weights=weights))})
    if len(records) != 102 or len({(r['run_id'], r['source_row_1based']) for r in records}) != 102:
        raise ValueError('expected 102 unique recoverable source rows')
    artifact = {'scope': 'frozen native held-out predictions; no refit',
                'source_csv_sha256': sha(source_path),
                'source_npz_sha256': source_hashes,
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
    args = parser.parse_args()
    print(json.dumps({'rows': len(recover(args.results, args.processed, args.output)['rows'])}))
