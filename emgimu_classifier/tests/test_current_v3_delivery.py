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


def test_emg_only_bank_provider_removals_and_actual_calibration_cost():
    result = json.loads((ROOT/'benchmarks/new_bank_v3/EMG_F0_F7_BANK_V1_RESULTS.json').read_text(encoding='utf8'))
    table = [r for r in rows('ablation_full_bank.csv') if r['run_id']=='emg_f0_f7_bank_v1']
    assert len(table)==66
    assert len({(r['subject'],r['calibration_budget'],r['removed_provider']) for r in table})==66
    for row in table:
        shots=row['calibration_budget'];user=row['subject'];remaining=row['remaining_bank']
        score=result['scores'][shots]
        full=score['F0_F7']['pooled'] if user=='ALL' else score['F0_F7']['per_user'][user]
        alt=score[remaining]['pooled'] if user=='ALL' else score[remaining]['per_user'][user]
        assert int(row['evaluation_trials'])==(1200 if user=='ALL' else 120)
        assert row['full_bank']=='F0_F7' and row['removed_provider']!=remaining
        assert int(row['full_target_calibration_trials_per_user'])==6*int(shots)
        assert int(row['remaining_target_calibration_trials_per_user'])==(0 if remaining=='F0' else 6*int(shots))
        for key in ('macro_f1','accuracy','log_loss','brier'):
            assert abs(float(row['full_'+key])-full[key])<1e-12
            assert abs(float(row['remaining_'+key])-alt[key])<1e-12
        assert abs(float(row['delta_logloss'])-(alt['log_loss']-full['log_loss']))<1e-12
        assert abs(float(row['delta_macro_f1'])-(full['macro_f1']-alt['macro_f1']))<1e-12
        notes=json.loads(row['metadata_notes_json'])
        assert notes['no_IMU_features'] and notes['actual_target_calibration_trials_per_user']['F0']==0
        assert notes['ablation_is_two_provider_probability_removal_not_whole_document_bank']


def test_label_stability_export_retains_shared_denominator_and_unavailable_probabilities():
    native = json.loads((ROOT/'benchmarks/new_bank_v3/ROAM_DEBOUNCE_CONTROL_V1_RESULTS.json').read_text(encoding='utf8'))
    source = {r['native_file']: r for r in native['records']}
    table = rows('label_stability_control.csv')
    assert len(table) == 80
    assert len({(r['native_file'], r['label_policy']) for r in table}) == 80
    for row in table:
        record = source[row['native_file']]; arm = record['arms'][row['label_policy']]
        assert row['evaluation_unit'] == 'nominal_sample'
        assert int(row['shared_known_samples']) == record['shared_known_samples']
        assert int(row['unknown_samples']) == arm['unknown_samples']
        assert int(row['samples']) == record['samples']
        assert all(row[k] == 'N/A' for k in ('log_loss', 'brier', 'ece'))
        for key in ('accuracy', 'macro_f1'):
            assert float(row[key]) == arm['shared_known_scores'][key]
        assert float(row['full_record_accuracy_unknown_wrong']) == arm['full_record_accuracy_unknown_wrong']
        for key in ('eligible_transitions', 'correct_transitions', 'maintenance_switches'):
            assert int(row[key]) == arm['transition_hold'][key]


def test_delivery_sources_schemas_and_unavailable_results():
    manifest = json.loads((OUT/'MANIFEST.json').read_text())
    assert not manifest['completion_proven'] and not manifest['default_promoted']
    assert manifest['generator_sha256'] == sha(ROOT/'benchmarks/new_bank_v3/export_current_delivery.py')
    for path, digest in manifest['source_sha256'].items():
        assert sha(ROOT/path) == digest
    expected = {'feature_family_results.csv': 3593, 'conditional_incremental.csv': 5072,
                'error_complementarity.csv': 5072, 'calibration_curve.csv': 3593, 'ablation_full_bank.csv':944,
                'budget_eligibility.csv': 170, 'boundary_detection.csv': 2, 'calibration_burden.csv': 710, 'continuous_recognition.csv':40, 'transition_hold.csv':40, 'label_stability_control.csv':80, 'quality_gate.csv':24}
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
    assert sum(r['eligible'] == 'True' for r in guard) == 40
    assert sum(r['eligible'] == 'False' for r in guard) == 130
    assert all(int(r['required_independent_trials']) == int(r['dimension'])+2 for r in guard)
    dtw = [r for r in rows('feature_family_results.csv') if r['feature_family'] == 'DTW']
    assert len(dtw) == 16
    assert all(r[k] == 'N/A' for r in dtw for k in ('log_loss', 'brier', 'ece'))
    for row in rows('conditional_incremental.csv'):
        is_concat=(row['run_id']=='emg_window_bank_v1' and row['core_bank']=='F0' and row['added_family'].startswith('F0_plus_'))
        expected_kind=('paired_fixed_probability_mixture_not_concatenated_increment' if row['run_id'] in ('personal_temporal_unibo_v1','detected_personal_temporal_unibo_v3','calibration_rest_continuous_unibo_v1') else
                       'source_refit_concatenated_group_increment' if is_concat else 'paired_alternative_not_concatenated_increment')
        assert row['comparison_kind']==expected_kind


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
        assert len(keys) == len(set(keys)) == 5072
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
            if row['run_id'] not in ('personal_temporal_unibo_v1','detected_personal_temporal_unibo_v3','calibration_rest_continuous_unibo_v1'):
                assert np.isclose(float(row[rate]),int(row[count])/n,rtol=0,atol=1e-12)
            else:
                assert 0 <= float(row[rate]) <= 1  # Exact trial-mass values checked against retained probabilities below.
        assert row['correlation_status'] == ('undefined_constant_error_vector' if row['error_correlation'] == 'N/A' else 'defined')
    for row in rows('feature_family_results.csv'):
        metrics = json.loads(row['class_metrics_json']); n = int(row['evaluation_trials'])
        assert sum(c['support'] for c in metrics.values()) == n
        assert sum(c['predicted'] for c in metrics.values()) == n
        for c in metrics.values():
            tp,fp,fn = c['true_positives'],c['false_positives'],c['false_negatives']
            assert tp+fp == c['predicted'] and tp+fn == c['support']
            if row['run_id'] in ('personal_temporal_unibo_v1','detected_personal_temporal_unibo_v3','calibration_rest_continuous_unibo_v1'):
                tw,sw,pw=c['true_positive_weight'],c['support_weight'],c['predicted_weight']
                assert c['recall']==(tw/sw if sw else None)
                assert c['precision']==(tw/pw if pw else None)
                assert c['f1']==(2*tw/(sw+pw) if sw else None)
            else:
                assert c['recall'] == (tp/(tp+fn) if tp+fn else None)
                assert c['precision'] == (tp/(tp+fp) if tp+fp else None)
                assert c['f1'] == (2*tp/(2*tp+fp+fn) if c['support'] else None)
        if row['run_id'] == 'detected_g5_unibo_v1':
            assert metrics['0']['support'] == 0 and metrics['0']['recall'] is None and metrics['0']['f1'] is None


def test_holdout_burden_and_brier_definition_match_native_evidence():
    artifact = ROOT/'benchmarks/new_bank_v3/MAHALANOBIS_EPN_HOLDOUT_V2_BURDEN.json'
    native = json.loads(artifact.read_text())
    table = [r for r in rows('calibration_burden.csv') if r['run_id']=='MAHALANOBIS_EPN_HOLDOUT_V2']
    assert len(table) == len(native['records']) == 20
    for row, record in zip(table, native['records']):
        assert all(row[key] == str(value) for key, value in record.items())
    manifest = json.loads((OUT/'MANIFEST.json').read_text())
    assert manifest['brier_normalization'].startswith('mean across trials and classes')
    result = json.loads((ROOT/'benchmarks/new_bank_v3/MAHALANOBIS_EPN_HOLDOUT_V2_RESULTS.json').read_text())
    for row in rows('feature_family_results.csv'):
        if row['run_id'] != 'mahalanobis_epn_holdout_v2':
            continue
        notes = json.loads(row['metadata_notes_json'])
        assert notes['canonical_export_brier_normalization'] == notes['native_score_brier_normalization'] == 'mean across trials and classes'
        blocks = [b for b in result['blocks'] if b['shots'] == int(row['calibration_budget']) and (row['subject'] == 'ALL' or str(b['user']) == row['subject'])]
        probability = np.concatenate([b['probabilities'][row['feature_family'].removeprefix('pattern_')] for b in blocks])
        labels = np.concatenate([b['labels'] for b in blocks])
        assert np.isclose(float(row['brier']), np.mean((probability-np.eye(6)[labels])**2), atol=1e-12, rtol=0)


def test_continuous_tables_preserve_sample_units_warmup_and_native_event_scores():
    result=json.loads((ROOT/'benchmarks/new_bank_v3/ROAM_CAUSAL_WINDOW_V1_RESULTS.json').read_text(encoding='utf8'))
    records={r['native_file']:r for r in result['records']}
    for row in rows('continuous_recognition.csv'):
        native=records[row['native_file']]
        assert row['evaluation_unit']=='nominal_sample' and int(row['native_recordings'])==1
        assert int(row['scored_samples'])==native['samples']-39 and int(row['unknown_warmup_samples'])==39
        assert int(row['calibration_budget'])==0
        for metric in ('macro_f1','accuracy','log_loss','brier'):
            assert abs(float(row[metric])-native['scores'][metric])<1e-12
        counts=json.loads(row['class_metrics_json'])
        assert sum(v['support'] for v in counts.values())==int(row['scored_samples'])
        assert 0<=float(row['ece'])<=1
    for row in rows('transition_hold.csv'):
        native=records[row['native_file']]['transition_hold']
        for field in ('annotated_transitions','eligible_transitions','correct_transitions','maintenance_switches'):
            assert int(row[field])==native[field]
        assert float(row['transition_hold_accuracy'])==native['transition_hold_accuracy']


def test_quality_table_keeps_rejections_separate_from_fallback_probability_accuracy():
    artifact=ROOT/'benchmarks/song_real8/SONG_RAW_QUALITY_V1_RESULTS.json'
    result=json.loads(artifact.read_text(encoding='utf8'))
    with (artifact.parent/'song_raw_quality_v1/predictions.csv').open(encoding='utf8',newline='') as stream:
        native=list(csv.DictReader(stream))
    table=rows('quality_gate.csv')
    assert len(table)==24
    for row in table:
        selected=[v for v in native if (v['scenario'],v['mode'])==(row['scenario'],row['mode'])]
        rejected=np.array([v['rejected']=='True' for v in selected])
        correct=np.array([v['label']==v['predicted_label'] for v in selected])
        assert int(row['evaluation_trials'])==len(selected)==124
        assert int(row['rejected'])==int(rejected.sum())
        assert float(row['accuracy_unknown_wrong'])==float(correct.mean())
        assert float(row['coverage'])==float((~rejected).mean())
        if rejected.all(): assert row['accepted_accuracy']=='N/A'
        else: assert float(row['accepted_accuracy'])==float(correct[~rejected].mean())
        assert row['normal_false_rejection_rate']=='N/A'
        if row['fault_annotation']=='synthetic_known_fault': assert float(row['fault_recall'])==float(rejected.mean())
        else: assert row['fault_recall']=='N/A'
        assert int(row['long_term_calibration_trials'])==int(row['current_calibration_trials'])==20
        assert row['physical_validation_proven']==row['default_promoted']=='False'
    index=json.loads((BASE/'INDEX.json').read_text())
    entry=index['song_raw_quality_acceptance']
    assert entry['sha256']==sha((BASE/entry['path']).resolve())
