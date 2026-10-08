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
    expected = {'feature_family_results.csv': 387, 'conditional_incremental.csv': 206,
                'error_complementarity.csv': 206, 'calibration_curve.csv': 387,
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


def test_comparison_budget_identity_and_positive_improvement_signs():
    # The source document defines delta loss as M0 minus M1, while F1 is
    # higher-is-better. Resolve every pair to its exact budget, not first match.
    context = ('run_id','dataset','subject','session/domain','condition','calibration_budget')
    family = rows('feature_family_results.csv')
    lookup = {tuple(r[k] for k in context)+(r['feature_family'],):r for r in family}
    assert len(lookup) == len(family)
    for table, pair_fields in [('conditional_incremental.csv',('core_bank','added_family')),
                               ('error_complementarity.csv',('family_a','family_b'))]:
        comparisons = rows(table)
        keys = [tuple(r[k] for k in context+pair_fields) for r in comparisons]
        assert len(keys) == len(set(keys)) == 206
        for row in comparisons:
            key = tuple(row[k] for k in context)
            a = lookup[key+(row[pair_fields[0]],)]
            b = lookup[key+(row[pair_fields[1]],)]
            if table != 'conditional_incremental.csv':
                continue
            for source,delta in [('log_loss','delta_logloss'),('brier','delta_brier'),('macro_f1','delta_macro_f1')]:
                if a[source] == 'N/A' or b[source] == 'N/A':
                    assert row[delta] == 'N/A'
                else:
                    expected = float(b[source])-float(a[source]) if source == 'macro_f1' else float(a[source])-float(b[source])
                    assert np.isclose(float(row[delta]),expected,rtol=0,atol=1e-12)
    for row in rows('calibration_curve.csv'):
        assert row['calibration_budget'] == row['shots_per_class']


def test_error_probabilities_denominators_and_class_confusion_metrics():
    for row in rows('error_complementarity.csv'):
        n = int(row['evaluation_trials']); assert n > 0
        for count,rate in [('a_correct_b_wrong','a_correct_b_wrong_probability'),
                           ('a_wrong_b_correct','a_wrong_b_correct_probability')]:
            assert np.isclose(float(row[rate]),int(row[count])/n,rtol=0,atol=1e-12)
        assert row['correlation_status'] == ('undefined_constant_error_vector' if row['error_correlation'] == 'N/A' else 'defined')
    for row in rows('feature_family_results.csv'):
        metrics = json.loads(row['class_metrics_json']); n = int(row['evaluation_trials'])
        assert sum(c['support'] for c in metrics.values()) == n
        assert sum(c['predicted'] for c in metrics.values()) == n
        for c in metrics.values():
            tp,fp,fn = c['true_positives'],c['false_positives'],c['false_negatives']
            assert tp+fp == c['predicted'] and tp+fn == c['support']
            assert c['recall'] == (tp/(tp+fn) if tp+fn else None)
            assert c['precision'] == (tp/(tp+fp) if tp+fp else None)
            assert c['f1'] == (2*tp/(2*tp+fp+fn) if c['support'] else None)
        if row['run_id'] == 'detected_g5_unibo_v1':
            assert metrics['0']['support'] == 0 and metrics['0']['recall'] is None and metrics['0']['f1'] is None
