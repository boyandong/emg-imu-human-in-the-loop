"""Export the frozen EPN final shortlist with stable canonical table columns."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(source: Path, audit_path: Path, output: Path) -> None:
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    for name, expected in audit['output_sha256'].items():
        if sha(source / name) != expected:
            raise ValueError(f'frozen replay output changed: {name}')

    output.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, expected_rows in (('feature_family_results.csv', 48),
                                ('ablation_full_bank.csv', 20)):
        with (source / name).open(encoding='utf-8', newline='') as handle:
            reader = csv.DictReader(handle)
            fields = [key for key in reader.fieldnames
                      if key != 'delta_logloss_vs_full']
            rows = [{key: row[key] for key in fields} for row in reader]
        if len(rows) != expected_rows:
            raise ValueError(f'unexpected frozen replay row count: {name}')
        with (output / name).open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        counts[name] = len(rows)

    (output / 'split_trial_ids.json').write_bytes(
        (source / 'split_trial_ids.json').read_bytes())
    manifest = {
        'dataset': 'epn612',
        'phase': 'final',
        'source_users': list(range(1, 16)),
        'validation_users': [16, 17, 18],
        'target_users': [19, 20, 21],
        'families': ['F0', 'F3_Ring', 'F2b_CSP', 'F6_IMU'],
        'calibration_budgets': [0],
        'random_seeds': {'source_model_selection': 20260915},
        'model_hyperparameters': 'frozen source fitted states; see original source manifest',
        'preprocessing_config': 'frozen source family states; native EPN trial aggregation',
        'split_ids': 'exact source/validation/final trial IDs in split_trial_ids.json',
        'session_ids': None,
        'force_labels': None,
        'source_run': 'feature_bank_epn_selection_20260915',
        'source_result': str(audit_path),
        'source_result_sha256': sha(audit_path),
        'source_outputs_sha256': audit['output_sha256'],
        'export_script_sha256': sha(Path(__file__)),
        'output_sha256': {name: sha(output / name) for name in counts},
        'rows': counts,
        'boundary': ('Only the supplementary delta_logloss_vs_full column is omitted '
                     'from the canonical ablation CSV; the original value remains '
                     'in the frozen replay source and compact audit. Final users '
                     'were examined elsewhere in the project, and reference Ring '
                     'is not proven historical RLCS.'),
    }
    (output / 'run_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'ok', 'rows': counts}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('audit', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    export(args.source, args.audit, args.output)
