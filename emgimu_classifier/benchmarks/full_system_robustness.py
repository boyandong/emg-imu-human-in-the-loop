"""Report frozen full-bank evidence without substituting selected single-family experts."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import statistics


def build(results: Path) -> dict:
    sources = {}

    def load(name, run, **filters):
        path = results / name
        sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(encoding='utf-8-sig', newline='') as handle:
            return [r for r in csv.DictReader(handle) if r['run_id'] == run
                    and all(r.get(k) == str(v) for k, v in filters.items())]

    def one(rows, **filters):
        found = [r for r in rows if all(r.get(k) == str(v) for k, v in filters.items())]
        if len(found) != 1:
            raise ValueError(f'Expected exactly one metric row: {filters}, found {len(found)}')
        return found[0]

    rows = []
    specs = (
        ('force', 'feature_bank_force_probability_final_20260915', 'calibration_curve.csv', 'full', 'ALL', None,
         'source Ramp; unseen force; seven providers; F7 inactive; no F8 or quality routing'),
        ('wearing', 'feature_bank_wearing_full_fusion_final_20260915', 'ablation_full_bank.csv', 'full', 'ALL', 'trial_',
         'same-user before/after; nine providers; source before calibration; F7/F8 inactive'),
        ('day', 'feature_bank_unibo_full_fusion_final_20260915', 'ablation_full_bank.csv', 'full', 'ALL', 'd0',
         'native four-channel days 7/8; eight providers; no target calibration or IMU/ring'),
        ('posture', 'feature_bank_unibo_full_fusion_final_20260915', 'ablation_full_bank.csv', 'full', 'ALL', 'posture_',
         'same UniBo observations as day axis; posture strata; posture not a measured IMU direction'),
        ('user', 'feature_bank_epn_probability_final_20260915', 'calibration_curve.csv', 'full', 'cross_user', None,
         'held-out users; eight providers; F7 inactive; no session key or quality routing'),
        ('speed', 'feature_bank_manus_calibrated_quality_final_20260915', 'ablation_full_bank.csv', 'combined_full', 'ALL', None,
         'same users session 3; six finger classes; eight providers; session/speed confounded; F7/F8 inactive at cal0'),
    )
    for factor, run, filename, method, all_condition, prefix, scope in specs:
        data = load(filename, run, shots_per_class=0)
        full = one(data, subject='ALL', condition=all_condition, method=method)
        full_cells = [r for r in data if r['subject'] == 'ALL' and r['method'] == method
                      and r['condition'] != all_condition
                      and (prefix is None or r['condition'].startswith(prefix))]
        full_users = [r for r in data if r['subject'] != 'ALL' and r['method'] == method
                      and r['condition'] == all_condition]
        if factor in ('user', 'speed'):
            diag_run = ('feature_bank_epn_calibration_diagnostics_final_20260915_v2' if factor == 'user'
                        else 'feature_bank_manus_calibration_diagnostics_final_20260915')
            baseline_users = load('per_family_calibration_gain.csv', diag_run, family='F0', shots_per_class=0)
            if {r['subject'] for r in baseline_users} != {r['subject'] for r in full_users}:
                raise ValueError('Full and baseline user populations differ')
            baseline = statistics.mean(float(r['zero_shot_same_eval_macro_f1']) for r in baseline_users)
            baseline_min_user = min(float(r['zero_shot_same_eval_macro_f1']) for r in baseline_users)
            baseline_worst_condition = 'N/A'
            baseline_run = diag_run
            aggregation = 'mean user macro-F1'
        else:
            base = one(data, subject='ALL', condition=all_condition, method='baseline_F0')
            baseline = float(base['macro_f1']); baseline_run = run
            baseline_cells = [r for r in data if r['subject'] == 'ALL' and r['method'] == 'baseline_F0'
                              and r['condition'] != all_condition
                              and (prefix is None or r['condition'].startswith(prefix))]
            baseline_worst_condition = min((float(r['macro_f1']) for r in baseline_cells), default='N/A')
            baseline_min_user = min(float(r['macro_f1']) for r in data if r['subject'] != 'ALL'
                                    and r['method'] == 'baseline_F0' and r['condition'] == all_condition)
            aggregation = 'pooled weighted windows' if factor in ('day', 'posture') else (
                'pooled trials' if factor == 'wearing' else 'mean user macro-F1')
        full_worst_condition = min((float(r['macro_f1']) for r in full_cells), default='N/A')
        condition_run = run
        if factor == 'force':
            condition_run = 'feature_bank_force_condition_final_20260915'
            conditions = load('force_worst_condition.csv', condition_run, shots_per_class=0)
            baseline_worst_condition = float(one(conditions, method='baseline_F0')['minimum_condition_macro_f1'])
            full_worst_condition = float(one(conditions, method='full')['minimum_condition_macro_f1'])
        rows.append({'factor': factor, 'full_run': run, 'baseline_run': baseline_run, 'condition_run': condition_run,
                     'method': method, 'target_shots_per_class': 0, 'aggregation': aggregation,
                     'baseline_R': baseline, 'full_R': float(full['macro_f1']),
                     'delta_R': float(full['macro_f1']) - baseline,
                     'baseline_worst_condition': baseline_worst_condition,
                     'full_worst_condition': full_worst_condition,
                     'baseline_min_user': baseline_min_user,
                     'full_min_user': min(float(r['macro_f1']) for r in full_users), 'scope': scope})
    rows.append({k: ('quality' if k == 'factor' else 'no labelled real-quality benchmark' if k == 'scope' else 'N/A')
                 for k in rows[0]})
    rows.sort(key=lambda r: ('force', 'wearing', 'day', 'user', 'posture', 'speed', 'quality').index(r['factor']))
    available = rows[:-1]
    summary = {'status': 'ok', 'target_budget': 0,
               'scope': 'frozen full-bank algorithms on native benchmark tasks; no best expert substitution; not a universal fitted classifier',
               'mean_R_available': statistics.mean(r['full_R'] for r in available),
               'R_min_available': min(r['full_R'] for r in available),
               'baseline_mean_R_available': statistics.mean(r['baseline_R'] for r in available),
               'baseline_R_min_available': min(r['baseline_R'] for r in available),
               'full_min_observed_condition': min(r['full_worst_condition'] for r in available
                                                 if r['full_worst_condition'] != 'N/A'),
               'improved_axes': [r['factor'] for r in available if r['delta_R'] > 0],
               'declined_axes': [r['factor'] for r in available if r['delta_R'] < 0],
               'source_csv_sha256': sources,
               'warnings': ['Day and posture reuse the same UniBo observations; axes are correlated.',
                            'Mean across unlike tasks is descriptive, not pooled accuracy or statistical independence.',
                            'Missing condition baselines remain N/A; no inferred worst-condition improvement.',
                            'Full-bank F7/F8 are inactive at cal0; quality routing absent where not implemented.',
                            'Current wearable hardware effectiveness requires controlled local data.']}
    with (results / 'full_system_robustness_vector.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (results / 'full_system_robustness_vector.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('results', type=Path)
    args = parser.parse_args(); result = build(args.results)
    print(json.dumps({k: result[k] for k in ('status', 'mean_R_available', 'R_min_available',
                                           'baseline_mean_R_available', 'baseline_R_min_available', 'declined_axes')}))
