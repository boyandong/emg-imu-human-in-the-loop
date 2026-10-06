import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'feature_bank/delivery'
OUT = BASE/'new_bank_v3'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(name):
    with (OUT/name).open(encoding='utf8', newline='') as stream:
        return list(csv.DictReader(stream))


def test_delivery_sources_schemas_and_unavailable_results():
    manifest = json.loads((OUT/'MANIFEST.json').read_text())
    assert not manifest['completion_proven'] and not manifest['default_promoted']
    assert manifest['generator_sha256'] == sha(ROOT/'benchmarks/new_bank_v3/export_current_delivery.py')
    for path, digest in manifest['source_sha256'].items():
        assert sha(ROOT/path) == digest
    expected = {'feature_family_results.csv': 303, 'conditional_incremental.csv': 122,
                'error_complementarity.csv': 122, 'calibration_curve.csv': 303,
                'budget_eligibility.csv': 150, 'boundary_detection.csv': 2}
    for name, count in expected.items():
        table = rows(name)
        assert len(table) == manifest['tables'][name]['rows'] == count
        assert sha(OUT/name) == manifest['tables'][name]['sha256']
        assert all(all(value != '' for value in row.values()) for row in table)
        for row in table:
            assert manifest['source_sha256'][row['source_artifact']] == row['source_sha256']
    index = json.loads((BASE/'INDEX.json').read_text())
    for name, digest in index['base_table_sha256'].items():
        assert sha(BASE/name) == digest
    for name, digest in index['v3_result_inventory_sha256'].items():
        assert sha(ROOT/name) == digest
    for paths in index['tables'].values():
        assert len(paths) == 2 and all((BASE/p).is_file() for p in paths)
    guard = rows('budget_eligibility.csv')
    assert sum(r['eligible'] == 'True' for r in guard) == 20
    assert sum(r['eligible'] == 'False' for r in guard) == 130
    assert all(int(r['required_independent_trials']) == int(r['dimension'])+2 for r in guard)
    dtw = [r for r in rows('feature_family_results.csv') if r['feature_family'] == 'DTW']
    assert len(dtw) == 16
    assert all(r[k] == 'N/A' for r in dtw for k in ('log_loss', 'brier', 'ece'))
    assert all(r['comparison_kind'] == 'paired_alternative_not_concatenated_increment'
               for r in rows('conditional_incremental.csv'))


def test_delivery_pooled_g5_metrics_and_paired_errors_from_predictions():
    source = json.loads((ROOT/'benchmarks/new_bank_v3/DETECTED_G5_UNIBO_V1_RESULTS.json').read_text())
    events = [e for e in source['events'] if e['reference_label'] is not None]
    y = np.array([e['reference_label'] for e in events])
    family = rows('feature_family_results.csv')
    errors = rows('error_complementarity.csv')
    for mode in ('detected', 'matched_oracle'):
        oracle = mode == 'matched_oracle'
        prob = np.array([e['g5_oracle_probability' if oracle else 'g5_probability'] for e in events])
        a = np.array([e['oracle_prediction' if oracle else 'prediction'] for e in events])
        b = prob.argmax(axis=1)
        condition = mode+'_supported_matched_active'
        row = next(r for r in family if r['run_id'] == 'detected_g5_unibo_v1'
                   and r['subject'] == 'ALL' and r['condition'] == condition and r['feature_family'] == 'G5')
        assert np.isclose(float(row['accuracy']), np.mean(b == y))
        assert np.isclose(float(row['macro_f1']), f1_score(y, b, labels=range(4), average='macro', zero_division=0))
        assert np.isclose(float(row['log_loss']), log_loss(y, prob, labels=range(4)))
        assert np.isclose(float(row['brier']), np.mean((prob-np.eye(4)[y])**2))
        pair = next(r for r in errors if r['run_id'] == 'detected_g5_unibo_v1'
                    and r['subject'] == 'ALL' and r['condition'] == condition)
        assert int(pair['a_correct_b_wrong']) == np.sum((a == y) & (b != y))
        assert int(pair['a_wrong_b_correct']) == np.sum((a != y) & (b == y))
        assert np.isclose(float(pair['disagreement_rate']), np.mean(a != b))
