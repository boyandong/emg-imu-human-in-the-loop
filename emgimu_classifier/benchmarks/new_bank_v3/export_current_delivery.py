"""Add current V3 native diagnostics to canonical tables without rewriting prior releases."""
import csv
import hashlib
import json
from itertools import combinations
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score, log_loss
from emgimu.feature_bank.screening import expected_calibration_error

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASE = ROOT/'feature_bank/delivery'
OUT = BASE/'new_bank_v3'
METRICS = ('macro_f1', 'accuracy', 'log_loss', 'brier', 'ece')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(y, prediction, probability, classes):
    y = np.asarray(y); prediction = np.asarray(prediction)
    result = {'macro_f1': float(f1_score(y, prediction, labels=np.arange(classes), average='macro', zero_division=0)),
              'accuracy': float(np.mean(y == prediction)), 'log_loss': 'N/A', 'brier': 'N/A', 'ece': 'N/A'}
    if probability is not None:
        p = np.asarray(probability)
        result.update(log_loss=float(log_loss(y, p, labels=np.arange(classes))),
                      brier=float(np.mean((p-np.eye(classes)[y])**2)),
                      ece=expected_calibration_error(y, p, np.ones(len(y))))
    per_class = {}
    for c in range(classes):
        support = int(np.sum(y == c)); predicted = int(np.sum(prediction == c))
        tp = int(np.sum((y == c) & (prediction == c)))
        per_class[str(c)] = {'support':support,'predicted':predicted,'true_positives':tp,
            'false_positives':predicted-tp,'false_negatives':support-tp,
            'precision':tp/predicted if predicted else None,
            'recall':tp/support if support else None,
            'f1':2*tp/(support+predicted) if support else None}
    result['class_metrics_json'] = json.dumps(per_class,sort_keys=True)
    return result


def write(name, rows, fields):
    if not rows:
        raise AssertionError('Empty current V3 delivery table: '+name)
    with (OUT/name).open('w', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def export():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = {}
    family = []; incremental = []; errors = []; curve = []; eligibility = []; boundaries = []
    def read(name):
        path = HERE/name; sources[path.relative_to(ROOT).as_posix()] = sha(path)
        value = json.loads(path.read_text())
        if value.get('default_promoted', False):
            raise AssertionError('Exporter does not authorize default promotion')
        return value
    def add_group(run, source, dataset, subject, domain, shots, condition, arms, y, classes, notes):
        context = {'run_id': run, 'source_artifact': source, 'source_sha256': sources[source],
                   'dataset': dataset, 'subject': str(subject), 'session/domain': domain,
                   'condition': condition, 'calibration_budget': shots,
                   'evaluation_trials': len(y),
                   'metadata_notes_json': json.dumps(notes, sort_keys=True)}
        statistics = {}
        for name, (prediction, probability) in arms.items():
            s = score(y, prediction, probability, classes); statistics[name] = s
            family.append({**context, 'feature_family': name, 'calibration_budget': shots, **s})
            curve.append({**context, 'feature_bank': name, 'shots_per_class': shots, 'method': run,
                          'supported': True, 'macro_f1': s['macro_f1'], 'log_loss': s['log_loss']})
        for a, b in combinations(arms, 2):
            pa = np.asarray(arms[a][0]); pb = np.asarray(arms[b][0]); truth = np.asarray(y)
            ea = pa != truth; eb = pb != truth
            corr = float(np.corrcoef(ea.astype(float), eb.astype(float))[0, 1]) if np.std(ea) > 0 and np.std(eb) > 0 else 'N/A'
            errors.append({**context, 'family_a': a, 'family_b': b, 'error_correlation': corr,
                           'correlation_status': 'undefined_constant_error_vector' if corr == 'N/A' else 'defined',
                           'disagreement_rate': float(np.mean(pa != pb)),
                           'a_correct_b_wrong': int(np.sum(~ea & eb)), 'a_wrong_b_correct': int(np.sum(ea & ~eb)),
                           'a_correct_b_wrong_probability':float(np.mean(~ea & eb)),
                           'a_wrong_b_correct_probability':float(np.mean(ea & ~eb))})
            delta = {}
            for metric, output in [('log_loss', 'delta_logloss'), ('macro_f1', 'delta_macro_f1'), ('brier', 'delta_brier')]:
                x, z = statistics[a][metric], statistics[b][metric]
                delta[output] = ((z-x if metric == 'macro_f1' else x-z)
                                 if x != 'N/A' and z != 'N/A' else 'N/A')
            incremental.append({**context, 'core_bank': a, 'added_family': b,
                                'comparison_kind': 'paired_alternative_not_concatenated_increment', **delta})
    for name, run_id in [('F8_ROUTER_MANUS_V1_RESULTS.json', 'f8_router_manus_v1'),
                         ('F8_CALIBRATED_MANUS_V2_RESULTS.json', 'f8_calibrated_manus_v2')]:
        f8 = read(name); source = (HERE/name).relative_to(ROOT).as_posix()
        for phase in ('validation', 'descriptive_final'):
            for shots in (1, 2):
                selected = [b for b in f8['blocks'] if b['phase'] == phase and b['shots'] == shots]
                for user in ['ALL']+sorted({b['user'] for b in selected}):
                    group = selected if user == 'ALL' else [b for b in selected if b['user'] == user]
                    y = np.concatenate([b['labels'] for b in group])
                    arms = {}
                    for arm in ('TD24', 'uniform', 'F8'):
                        p = np.concatenate([b['probabilities'][arm] for b in group]); arms[arm] = (p.argmax(axis=1), p)
                    add_group(run_id, source, 'sEMG-MANUS', user, phase, shots, 'frozen_evaluation_trials',
                              arms, y, 6, {'classes': 6, 'ontology': 'six native finger-flexion gestures',
                              'scope': f8['scope'], 'target_calibration_disjoint': True,
                              'source_oof_temperature_calibrated': run_id.endswith('v2'), 'TD24_is_document_F0': False})
    name = 'MAHALANOBIS_EPN_BUDGET_V1_RESULTS.json'; md = read(name); source = (HERE/name).relative_to(ROOT).as_posix()
    for block in md['blocks']:
        eligibility.append({'dataset': 'EPN612', 'subject': block['user'], 'feature_family': block['family'],
                            'shots_per_class': block['shots'], 'dimension': block['dimension'],
                            'required_independent_trials': block['required_trials_per_class'],
                            'eligible': block['mahalanobis_eligible'],
                            'reason': 'eligible' if block['mahalanobis_eligible'] else block['ineligible_reason'],
                            'source_artifact': source, 'source_sha256': sources[source]})
    for feature in ('pattern', 'TD24', 'SPD'):
        for shots in (1, 2, 5, 10, 20):
            selected = [b for b in md['blocks'] if b['family'] == feature and b['shots'] == shots]
            for user in ['ALL']+sorted({b['user'] for b in selected}):
                group = selected if user == 'ALL' else [b for b in selected if b['user'] == user]
                y = np.concatenate([b['labels'] for b in group]); arms = {}
                for arm in ('euclidean', 'mahalanobis'):
                    if arm == 'mahalanobis' and not all(b['mahalanobis_eligible'] for b in group):
                        continue
                    p = np.concatenate([b[f'{arm}_probability'] for b in group]); arms[feature+'_'+arm] = (p.argmax(axis=1), p)
                add_group('mahalanobis_epn_budget_v1', source, 'EPN612', user, 'descriptive_users22_31', shots,
                          'fixed_evaluation_trials_across_budgets', arms, y, 6,
                          {'classes': 6, 'ontology': 'EPN612 native six classes', 'dimension': group[0]['dimension'],
                           'scope': md['scope'], 'independent_trial_guard': True})
    name = 'MAHALANOBIS_EPN_HOLDOUT_V2_RESULTS.json'; holdout = read(name); source = (HERE/name).relative_to(ROOT).as_posix()
    for block in holdout['blocks']:
        eligibility.append({'dataset': 'EPN612', 'subject': block['user'], 'feature_family': 'pattern',
                            'shots_per_class': block['shots'], 'dimension': 8,
                            'required_independent_trials': 10, 'eligible': True, 'reason': 'eligible',
                            'source_artifact': source, 'source_sha256': sources[source]})
    for shots in (10, 20):
        selected = [b for b in holdout['blocks'] if b['shots'] == shots]
        for user in ['ALL'] + sorted({b['user'] for b in selected}):
            group = selected if user == 'ALL' else [b for b in selected if b['user'] == user]
            y = np.concatenate([b['labels'] for b in group]); arms = {}
            for arm in ('euclidean', 'mahalanobis'):
                p = np.concatenate([b['probabilities'][arm] for b in group])
                arms['pattern_' + arm] = (p.argmax(axis=1), p)
            add_group('mahalanobis_epn_holdout_v2', source, 'EPN612', user, 'precommitted_users32_41', shots,
                      'fixed_evaluation_trials_across_budgets', arms, y, 6,
                      {'classes': 6, 'ontology': 'EPN612 native six classes', 'dimension': 8,
                       'scope': holdout['scope'], 'independent_trial_guard': True,
                       'native_score_brier_normalization': 'mean across trials and classes',
                       'canonical_export_brier_normalization': 'mean across trials and classes'})
    name = 'DETECTED_G5_UNIBO_V1_RESULTS.json'; g5 = read(name); source = (HERE/name).relative_to(ROOT).as_posix()
    supported = [e for e in g5['events'] if e['reference_label'] is not None]
    for boundary in ('detected', 'matched_oracle'):
        for user in ['ALL']+sorted({e['user'] for e in supported}):
            group = supported if user == 'ALL' else [e for e in supported if e['user'] == user]
            y = np.array([e['reference_label'] for e in group]); oracle = boundary == 'matched_oracle'
            p = np.array([e['g5_oracle_probability' if oracle else 'g5_probability'] for e in group])
            dtw = np.array([e['oracle_prediction' if oracle else 'prediction'] for e in group])
            # Keep prediction-only DTW explicitly probability-unavailable.
            arms = {'DTW': (dtw, None), 'G5': (p.argmax(axis=1), p)}
            add_group('detected_g5_unibo_v1', source, 'UniBo4', user, 'Day6_continuous', 0,
                      boundary+'_supported_matched_active', arms, y, 4,
                      {'classes': 4, 'no_rest_ground_truth': True, 'scope': g5['scope'],
                       'DTW_probability': 'unavailable; distances are not calibrated probabilities',
                       'calibration_budget': 'zero target-Day6 shots; source Days1-5 model/calibration retained'})
    for filename in ('AUTONOMOUS_UNIBO_V1_RESULTS.json', 'AUTONOMOUS_CONTINUOUS_UNIBO_V1_RESULTS.json'):
        result = read(filename); path = (HERE/filename).relative_to(ROOT).as_posix()
        t = result['totals']
        boundaries.append({'run_id': filename.removesuffix('_RESULTS.json'), 'dataset': 'UniBo4',
                           'references': t['references'], 'detections': t['detections'], 'matched': t['matched'],
                           'precision': t['precision'], 'recall': t['recall'],
                           'onset_mae_s': t['onset_mae_s'] if t['onset_mae_s'] is not None else 'N/A',
                           'offset_mae_s': t['offset_mae_s'] if t['offset_mae_s'] is not None else 'N/A',
                           'source_artifact': path, 'source_sha256': sources[path], 'scope': result.get('scope', '')})
    burden_name = 'MAHALANOBIS_EPN_HOLDOUT_V2_BURDEN.json'
    burden = read(burden_name)
    burden_source = (HERE/burden_name).relative_to(ROOT).as_posix()
    burden_rows = [{**row, 'source_artifact': burden_source, 'source_sha256': sources[burden_source]}
                   for row in burden['records']]
    continuous_name = 'ROAM_CAUSAL_WINDOW_V1_RESULTS.json'
    continuous = read(continuous_name)
    continuous_source = (HERE/continuous_name).relative_to(ROOT).as_posix()
    emissions_path = HERE/'ROAM_CAUSAL_WINDOW_V1_EMISSIONS.csv'
    sources[emissions_path.relative_to(ROOT).as_posix()] = sha(emissions_path)
    emitted = {}
    with emissions_path.open(encoding='utf8', newline='') as stream:
        for row in csv.DictReader(stream):
            emitted.setdefault(row['native_file'], []).append(row)
    continuous_rows = []; transition_rows = []
    for record in continuous['records']:
        truth = np.empty(record['samples'], dtype=int)
        for a,b,label in record['truth_rle']:
            truth[a:b] = label
        events = emitted[record['native_file']]
        probability = np.asarray([[float(e[f'p_{c}']) for c in range(3)] for e in events])
        dense = np.repeat(probability, 10, axis=0)[:record['samples']-39]
        metrics = score(truth[39:], dense.argmax(1), dense, 3)
        context = {'run_id': 'roam_causal_window_v1', 'source_artifact': continuous_source,
                   'source_sha256': sources[continuous_source], 'dataset': 'ROAM_EMG',
                   'subject': record['user'], 'session/domain': record['phase'],
                   'condition': record['posture'], 'native_file': record['native_file']}
        continuous_rows.append({**context, 'feature_family': 'F0v2_window48', 'calibration_budget': 0,
             'evaluation_unit': 'nominal_sample', 'native_recordings': 1,
             'scored_samples': record['samples']-39, 'unknown_warmup_samples': 39, **metrics,
             'metadata_notes_json': json.dumps({'classes':3, 'rate_hz':200, 'window_samples':40,
                 'hop_samples':10, 'no_future_samples':True, 'source_only_rest_thresholds':True,
                 'brier_normalization':'mean across scored nominal samples and classes',
                 'scope':continuous['scope'], 'sample_count_is_not_calibration_trial_count':True}, sort_keys=True)})
        diagnostic = record['transition_hold']
        transition_rows.append({**context, 'annotated_transitions': diagnostic['annotated_transitions'],
              'eligible_transitions': diagnostic['eligible_transitions'],
              'correct_transitions': diagnostic['correct_transitions'],
              'transition_hold_accuracy': diagnostic['transition_hold_accuracy'],
              'maintenance_switches': diagnostic['maintenance_switches'],
              'reaction_half_buffer_samples':100, 'nominal_sample_rate_hz':200,
              'scope':diagnostic['scope']})
    control_name = 'ROAM_DEBOUNCE_CONTROL_V1_RESULTS.json'
    control = read(control_name)
    control_source = (HERE/control_name).relative_to(ROOT).as_posix()
    control_rows = []
    for record in control['records']:
        for arm, info in record['arms'].items():
            diagnostic = info['transition_hold']
            control_rows.append({'run_id':'roam_debounce_control_v1',
                'source_artifact':control_source, 'source_sha256':sources[control_source],
                'dataset':'ROAM_EMG', 'subject':record['user'], 'session/domain':record['phase'],
                'condition':record['posture'], 'native_file':record['native_file'], 'label_policy':arm,
                'evaluation_unit':'nominal_sample', 'samples':record['samples'],
                'shared_known_samples':record['shared_known_samples'],
                'unknown_samples':info['unknown_samples'], **info['shared_known_scores'],
                'full_record_accuracy_unknown_wrong':info['full_record_accuracy_unknown_wrong'],
                'eligible_transitions':diagnostic['eligible_transitions'],
                'correct_transitions':diagnostic['correct_transitions'],
                'maintenance_switches':diagnostic['maintenance_switches'],
                'log_loss':'N/A', 'brier':'N/A', 'ece':'N/A', 'scope':control['scope']})
    context_fields = ['run_id', 'source_artifact', 'source_sha256', 'dataset', 'subject', 'session/domain', 'condition', 'calibration_budget', 'evaluation_trials', 'metadata_notes_json']
    tables = {'label_stability_control.csv': (control_rows, list(control_rows[0])),
              'continuous_recognition.csv': (continuous_rows, list(continuous_rows[0])),
              'transition_hold.csv': (transition_rows, list(transition_rows[0])),
              'feature_family_results.csv': (family, context_fields+['feature_family']+list(METRICS)+['class_metrics_json']),
              'conditional_incremental.csv': (incremental, context_fields+['core_bank', 'added_family', 'comparison_kind', 'delta_logloss', 'delta_macro_f1', 'delta_brier']),
              'error_complementarity.csv': (errors, context_fields+['family_a', 'family_b', 'error_correlation', 'correlation_status', 'disagreement_rate', 'a_correct_b_wrong', 'a_wrong_b_correct', 'a_correct_b_wrong_probability', 'a_wrong_b_correct_probability']),
              'calibration_curve.csv': (curve, context_fields+['feature_bank', 'shots_per_class', 'method', 'supported', 'macro_f1', 'log_loss']),
              'calibration_burden.csv': (burden_rows, list(burden_rows[0])),
              'budget_eligibility.csv': (eligibility, list(eligibility[0])), 'boundary_detection.csv': (boundaries, list(boundaries[0]))}
    for filename, (rows, fields) in tables.items():
        write(filename, rows, fields)
    manifest = {'schema': 'current_v3_canonical_delivery_v1', 'generator_sha256': sha(Path(__file__)),
                'source_sha256': sources,
                'brier_normalization': 'mean across trials and classes; not class-summed; compare only matched class ontologies',
                'calibration_burden_scope': 'Extracted window exposure differs from complete recording duration and physical session wall time; hardware times remain N/A.',
                'continuous_brier_normalization': 'mean across scored nominal samples and classes',
                'continuous_evaluation_scope': 'Separate nominal-sample records: source thresholds only, target warmup explicitly unknown, sample counts never treated as independent calibration trials. Transition-hold metric is not exact ReactEMG reproduction or hardware latency.',
                'class_metrics_scope': 'Per-class precision/recall/F1 with explicit support and prediction counts. Undefined precision or absent-ground-truth recall/F1 are null; no true-neutral claim on matched active-only UniBo trials.',
                'error_probability_scope': 'Paired correctness probabilities use the explicit shared evaluation-trial denominator. Undefined correlations are explained, never filled with zero.',
                'delta_convention': {'delta_logloss': 'base minus alternative; positive means improvement',
                                     'delta_brier': 'base minus alternative; positive means improvement',
                                     'delta_macro_f1': 'alternative minus base; positive means improvement'},
                'tables': {n: {'rows': len(rows), 'sha256': sha(OUT/n)} for n, (rows, _) in tables.items()},
                'boundary': 'Paired alternatives are labelled explicitly, not claimed as added-feature increments. DTW has no probability metrics. Budget failures are in eligibility, not fake performance rows. Conditions/datasets/class ontologies cannot be pooled indiscriminately.',
                'default_promoted': False, 'completion_proven': False}
    (OUT/'MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf8')
    index = {'schema': 'versioned_feature_bank_delivery_index_v1',
             'tables': {n: [n, 'new_bank_v3/'+n] for n in tables if (BASE/n).exists()},
             'additional_tables': ['new_bank_v3/boundary_detection.csv', 'new_bank_v3/budget_eligibility.csv', 'new_bank_v3/calibration_burden.csv', 'new_bank_v3/continuous_recognition.csv', 'new_bank_v3/transition_hold.csv', 'new_bank_v3/label_stability_control.csv'],
             'full_bank_ablation': 'ablation_full_bank.csv', 'current_manifest': 'new_bank_v3/MANIFEST.json',
             'base_provenance': 'PROVENANCE_AUDIT.json',
             'base_table_sha256': {f.name: sha(f) for f in sorted(BASE.glob('*.csv'))},
             'v3_result_inventory_sha256': {f.relative_to(ROOT).as_posix(): sha(f) for f in sorted(HERE.rglob('*.json'))
                                            if f.name.endswith('_RESULTS.json') or f.name == 'results.json'},
             'completion_proven': False}
    (BASE/'INDEX.json').write_text(json.dumps(index, indent=2)+'\n', encoding='utf8')
    print(json.dumps({n: len(rows) for n, (rows, _) in tables.items()}), flush=True)
    return manifest


if __name__ == '__main__':
    export()
