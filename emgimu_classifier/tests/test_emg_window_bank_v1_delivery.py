import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'benchmarks/new_bank_v3'


def metrics(y, p):
    prediction = p.argmax(1)
    f1 = []
    for label in range(6):
        denominator = sum(y == label) + sum(prediction == label)
        f1.append(2 * sum((y == label) & (prediction == label)) / denominator if denominator else 0.)
    return {'macro_f1': sum(f1) / 6, 'accuracy': np.mean(y == prediction),
            'log_loss': -np.mean(np.log(np.clip(p[np.arange(len(y)), y], np.finfo(float).eps, 1))),
            'brier': np.mean((p - np.eye(6)[y]) ** 2)}


def load():
    return json.loads((HERE / 'EMG_WINDOW_BANK_V1_RESULTS.json').read_text(encoding='utf8'))


def test_precommitted_source_contract_native_trial_ids_and_independent_model_replay():
    r = load(); path = HERE / 'EMG_WINDOW_BANK_V1_PROTOCOL.json'
    p = json.loads(path.read_text(encoding='utf8'))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == r['protocol_sha256']
    for name, digest in p['source_sha256'].items(): assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    assert p['source_users'] == list(range(1, 16)) and p['target_users'] == list(range(52, 62))
    assert not set(p['target_users']) & set(p['source_users'] + p['known_previous_target_users'])
    assert r['source_state_immutable'] and r['actual_target_calibration_trials'] == 0
    assert not r['default_promoted'] and not r['physical_validation_proven'] and not r['completion_proven']
    assert r['source_trial_count'] == len(set(r['source_trial_ids'])) == 2250
    assert len(r['source_models']) == 13 and len(r['blocks']) == 10
    all_ids = []
    for block in r['blocks']:
        ids = block['evaluation_ids']; all_ids.extend(ids)
        assert len(ids) == len(set(ids)) == 150 and not set(ids) & set(r['source_trial_ids'])
        y = np.array(block['labels'])
        np.testing.assert_array_equal(np.bincount(y), [25] * 6)
        assert set(block['features']) == set(p['groups'])
        for arm, fit in r['source_models'].items():
            groups = fit['groups']
            assert groups == p['compositions'][arm]
            dimension = sum(len(r['group_feature_names'][g]) for g in groups)
            assert dimension == fit['dimension'] and max(fit['iterations']) < 2000
            assert fit['classes'] == list(range(6))
            x = np.concatenate([block['features'][g] for g in groups], axis=1).astype(fit['feature_dtype'])
            # sklearn subtracts/divides in the input dtype; explicitly preserve
            # float32 rounding for compositions without any float64 group.
            x -= np.array(fit['mean'], dtype=x.dtype)
            x /= np.array(fit['scale'], dtype=x.dtype)
            logits = x @ np.array(fit['coef']).T + np.array(fit['intercept'])
            q = np.exp(logits - logits.max(1, keepdims=True)); q /= q.sum(1, keepdims=True)
            np.testing.assert_allclose(q, block['probabilities'][arm], atol=1e-12, rtol=1e-12)
    assert len(all_ids) == len(set(all_ids)) == 1500
    assert r['group_feature_names']['F1'] == [f'F1v3.rms_pattern.ch{c}' for c in range(1, 9)]
    assert r['source_models']['window_bank']['dimension'] == sum(len(n) for n in r['group_feature_names'].values())


def test_independent_paired_scores_primary_guard_and_complete_prediction_table():
    r = load(); table = HERE / 'EMG_WINDOW_BANK_V1_PREDICTIONS.csv'
    assert hashlib.sha256(table.read_bytes()).hexdigest() == r['prediction_sha256']
    with table.open(encoding='utf8', newline='') as stream: rows = list(csv.DictReader(stream))
    assert len(rows) == 19500
    values = {(int(row['user']), row['arm'], row['trial_id']): row for row in rows}
    assert len(values) == len(rows)
    for arm, scores in r['scores'].items():
        labels, probabilities = [], []
        for block in r['blocks']:
            y = np.array(block['labels']); q = np.array(block['probabilities'][arm])
            labels.extend(y); probabilities.extend(q)
            for key, value in metrics(y, q).items(): assert abs(value - scores['per_user'][str(block['user'])][key]) < 1e-12
            for trial, label, probability in zip(block['evaluation_ids'], y, q):
                row = values[block['user'], arm, trial]
                assert int(row['label']) == label
                np.testing.assert_array_equal(probability, [float(row[f'p_{c}']) for c in range(6)])
        for key, value in metrics(np.array(labels), np.array(probabilities)).items(): assert abs(value - scores['pooled'][key]) < 1e-12
        assert scores['minimum_user_macro_f1'] == min(v['macro_f1'] for v in scores['per_user'].values())
    base = r['scores']['F0']; full = r['scores']['window_bank']; primary = r['primary']
    wins = sum(full['per_user'][u]['log_loss'] < v['log_loss'] - 1e-12 for u, v in base['per_user'].items())
    assert primary['user_logloss_wins'] == wins
    expected = {'lower_log_loss': full['pooled']['log_loss'] < base['pooled']['log_loss'],
                'lower_brier': full['pooled']['brier'] < base['pooled']['brier'],
                'nonworse_macro_f1': full['pooled']['macro_f1'] >= base['pooled']['macro_f1'],
                'at_least_seven_user_loss_wins': wins >= 7}
    assert primary['criteria'] == expected and primary['passed'] == all(expected.values())
    assert abs(primary['delta_logloss'] - (base['pooled']['log_loss'] - full['pooled']['log_loss'])) < 1e-12
    assert abs(primary['delta_macro_f1'] - (full['pooled']['macro_f1'] - base['pooled']['macro_f1'])) < 1e-12
    assert abs(primary['delta_brier'] - (base['pooled']['brier'] - full['pooled']['brier'])) < 1e-12


def test_canonical_group_removals_retain_native_scores_and_zero_calibration_cost():
    r = load(); folder = ROOT / 'feature_bank/delivery/new_bank_v3'
    with (folder / 'ablation_full_bank.csv').open(encoding='utf8', newline='') as stream:
        rows = [x for x in csv.DictReader(stream) if x['run_id'] == 'emg_window_bank_v1']
    assert len(rows) == 66
    assert len({(x['subject'], x['removed_provider']) for x in rows}) == 66
    for row in rows:
        user = row['subject']; remaining = row['remaining_bank']
        assert remaining == 'window_bank_minus_' + row['removed_provider']
        assert row['removed_provider'] not in r['source_models'][remaining]['groups']
        assert row['full_bank'] == 'window_bank'
        assert int(row['evaluation_trials']) == (1500 if user == 'ALL' else 150)
        assert int(row['full_target_calibration_trials_per_user']) == int(row['remaining_target_calibration_trials_per_user']) == 0
        a = r['scores']['window_bank']; b = r['scores'][remaining]
        a = a['pooled'] if user == 'ALL' else a['per_user'][user]
        b = b['pooled'] if user == 'ALL' else b['per_user'][user]
        for key in ('macro_f1', 'log_loss', 'brier', 'accuracy'):
            assert abs(float(row['full_' + key]) - a[key]) < 1e-12
            assert abs(float(row['remaining_' + key]) - b[key]) < 1e-12
        assert abs(float(row['delta_logloss']) - (b['log_loss'] - a['log_loss'])) < 1e-12
        assert abs(float(row['delta_macro_f1']) - (a['macro_f1'] - b['macro_f1'])) < 1e-12
        notes = json.loads(row['metadata_notes_json'])
        assert notes['source_refit_for_each_declared_composition'] and notes['no_IMU_features']
        assert notes['ablation_is_six_declared_window_groups_not_whole_document_bank']
    with (folder / 'conditional_incremental.csv').open(encoding='utf8', newline='') as stream:
        additions = [x for x in csv.DictReader(stream) if x['run_id'] == 'emg_window_bank_v1' and x['comparison_kind'] == 'source_refit_concatenated_group_increment']
    assert len(additions) == 55
    for row in additions:
        assert row['core_bank'] == 'F0'
        assert r['source_models'][row['added_family']]['groups'] == ['F0', row['added_family'].removeprefix('F0_plus_')]
