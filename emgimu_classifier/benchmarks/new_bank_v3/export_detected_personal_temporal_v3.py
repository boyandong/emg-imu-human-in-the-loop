"""Append weighted, explicitly conditional continuous temporal native results."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
METRICS = ('macro_f1', 'accuracy', 'log_loss', 'brier', 'ece')


def append(sources, family, incremental, errors, curve, ablations):
    result_path = HERE/'DETECTED_PERSONAL_TEMPORAL_UNIBO_V3_RESULTS.json'
    result = json.loads(result_path.read_text(encoding='utf8'))
    source = result_path.relative_to(ROOT).as_posix()
    for path in (result_path, HERE/'DETECTED_PERSONAL_TEMPORAL_UNIBO_V3_PROTOCOL.json',
                 HERE/'detected_personal_temporal_unibo_v3/readouts.npz',
                 HERE/'detected_personal_temporal_unibo_v3/intervals.json',
                 HERE/'verify_detected_personal_temporal_unibo_v3.py', Path(__file__),
                 ROOT/'feature_bank/DETECTED_PERSONAL_TEMPORAL_UNIBO_ACCEPTANCE_V3.json'):
        sources[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    comparisons = [('base', arm) for arm in result['blocks'][0]['scores'] if arm != 'base']
    comparisons += [('base_uniform', 'base_full'), ('DTW_long', 'DTW_local'), ('base_DTW_long', 'base_DTW_blended')]
    with np.load(HERE/'detected_personal_temporal_unibo_v3/readouts.npz', allow_pickle=False) as arrays:
        for block in result['blocks']:
            y = np.array(block['labels']); weights = np.array(block['weights']); weights /= weights.sum()
            probabilities = {arm:arrays[block['key']+'_'+arm][block['matched_positions']] for arm in block['scores']}
            notes = dict(class_names=['neutral', 'index_pinch', 'fist', 'open_hand'],
                metric_weighting='equal observed matched active class within user; pooled equal users',
                conditional_on_supported_matched_references=True, no_rest_ground_truth=True,
                observed_active_classes=sorted(map(int, set(y))),
                primary_eligible=result['primary_eligible'], primary_pass=result['primary_pass'],
                active_macro_f1={arm:score['active_macro_f1'] for arm, score in block['scores'].items()},
                end_to_end_including_missed_references=block['end_to_end'],
                actual_additional_temporal_calibration_trials=block['predictive_calibration_cost'],
                G5_personalized_source_history_common_to_all_arms='Days1-5; not zero onboarding',
                all_reserved_recordings_excluded_at_all_budgets=True,
                exclusion_guard_native_samples=100, no_detector_rerun_or_rematching=True,
                unsupported_unmatched_predictions_retained_separately=True,
                branch_ablation='fixed .25 temporal mass reallocated to the remaining branch; not renormalization',
                oracle_complete_boundaries=block['mode'] == 'matched_oracle',
                boundary_kind='estimated' if block['mode'] == 'detected' else 'protocol_oracle', scope=result['scope'])
            context = dict(run_id='detected_personal_temporal_unibo_v3', source_artifact=source,
                source_sha256=sources[source], dataset='UniBo_native_continuous4', subject=block['user'],
                **{'session/domain':'Day6_'+block['mode']}, condition='supported_matched_active_conditional',
                calibration_budget=block['shots'], evaluation_trials=len(y), metadata_notes_json=json.dumps(notes, sort_keys=True))
            for arm, q in probabilities.items():
                predicted = q.argmax(1); class_metrics = {}
                for c in range(4):
                    support = int((y == c).sum()); guessed = int((predicted == c).sum())
                    tp = int(((y == c) & (predicted == c)).sum())
                    sw = weights[y == c].sum(); pw = weights[predicted == c].sum()
                    tw = weights[(y == c) & (predicted == c)].sum()
                    class_metrics[str(c)] = dict(support=support, predicted=guessed, true_positives=tp,
                        false_positives=guessed-tp, false_negatives=support-tp, support_weight=float(sw),
                        predicted_weight=float(pw), true_positive_weight=float(tw),
                        precision=float(tw/pw) if pw else None, recall=float(tw/sw) if sw else None,
                        f1=float(2*tw/(sw+pw)) if sw else None)
                score = block['scores'][arm]
                family.append({**context, 'feature_family':arm, **{k:score[k] for k in METRICS},
                    'class_metrics_json':json.dumps(class_metrics, sort_keys=True)})
                curve.append({**context, 'feature_bank':arm, 'shots_per_class':block['shots'],
                    'method':'detected_personal_temporal_unibo_v3', 'supported':True,
                    'macro_f1':score['macro_f1'], 'log_loss':score['log_loss']})
            for a, b in comparisons:
                ea = probabilities[a].argmax(1) != y; eb = probabilities[b].argmax(1) != y
                ma = np.average(ea, weights=weights); mb = np.average(eb, weights=weights)
                va = np.average((ea-ma)**2, weights=weights); vb = np.average((eb-mb)**2, weights=weights)
                corr = float(np.average((ea-ma)*(eb-mb), weights=weights)/np.sqrt(va*vb)) if va > 0 and vb > 0 else 'N/A'
                errors.append({**context, 'family_a':a, 'family_b':b, 'error_correlation':corr,
                    'correlation_status':'undefined_constant_error_vector' if corr == 'N/A' else 'defined',
                    'disagreement_rate':float(np.average(ea != eb, weights=weights)),
                    'a_correct_b_wrong':int((~ea & eb).sum()), 'a_wrong_b_correct':int((ea & ~eb).sum()),
                    'a_correct_b_wrong_probability':float(weights[~ea & eb].sum()),
                    'a_wrong_b_correct_probability':float(weights[ea & ~eb].sum())})
                base, alternative = block['scores'][a], block['scores'][b]
                incremental.append({**context, 'core_bank':a, 'added_family':b,
                    'comparison_kind':'paired_fixed_probability_mixture_not_concatenated_increment',
                    'delta_logloss':base['log_loss']-alternative['log_loss'],
                    'delta_macro_f1':alternative['macro_f1']-base['macro_f1'],
                    'delta_brier':base['brier']-alternative['brier']})
            full = block['scores']['base_full']
            for removed, remaining in [('DTW_branch_reallocation', 'base_signature_blended'),
                                       ('signature_branch_reallocation', 'base_DTW_blended')]:
                alternative = block['scores'][remaining]; cost = block['predictive_calibration_cost'][remaining]
                ablations.append({**context, 'full_bank':'base_full', 'removed_provider':removed, 'remaining_bank':remaining,
                    'full_target_calibration_trials_per_user':20+4*block['shots'],
                    'remaining_target_calibration_trials_per_user':cost['long_term']+cost['current'],
                    **{'full_'+k:full[k] for k in METRICS}, **{'remaining_'+k:alternative[k] for k in METRICS},
                    'delta_logloss':alternative['log_loss']-full['log_loss'],
                    'delta_macro_f1':full['macro_f1']-alternative['macro_f1'],
                    'delta_brier':alternative['brier']-full['brier']})
