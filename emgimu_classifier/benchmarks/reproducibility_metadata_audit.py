"""Inventory explicit saved metadata; field presence never proves reproduction."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


# The exact saved-information list in goal document lines162–170. Synonyms only
# locate candidate evidence; their meaning remains subject to scientific review.
FIELDS = {
    'split_ids': {'split_ids', 'splits', 'train', 'validation', 'test', 'oof', 'folds'},
    'subject_ids': {'subject_ids', 'subjects', 'source_users', 'target_users', 'users', 'subject'},
    'session_ids': {'session_ids', 'sessions', 'calendar_session_ids', 'domain_ids', 'domains', 'source_condition'},
    'force_labels': {'force_labels', 'force_conditions', 'conditions', 'source_condition'},
    'trial_ids': {'trial_ids', 'trials', 'fit_trials', 'train', 'validation', 'test', 'calibration', 'evaluation'},
    'preprocessing_config': {'preprocessing', 'preprocessing_config', 'window_ms', 'sample_rate_hz', 'sampling_rate'},
    'random_seeds': {'seed', 'random_seed', 'random_state', 'seeds'},
    'feature_config': {'feature_config', 'specifications', 'families', 'feature_families', 'feature_bank', 'feature_family'},
    'model_hyperparameters': {'classifier', 'model_config', 'hyperparameters', 'C', 'max_iter', 'n_estimators'},
}


def walk(value, prefix='$'):
    if isinstance(value, dict):
        for key, child in value.items():
            path = prefix + '.' + key
            yield key, path, child
            yield from walk(child, path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            if isinstance(child, (dict, list)):
                yield from walk(child, f'{prefix}[{index}]')


def audit(manifests, output):
    runs = sorted(manifests.glob('*__run_manifest.json'))
    if not runs:
        raise ValueError('No saved run manifests')
    rows = []
    sources = {}
    for manifest in runs:
        run_id = manifest.name.removesuffix('__run_manifest.json')
        candidates = [manifest]
        # Companion split files can carry the actual trial IDs omitted in a
        # manifest. Do not infer IDs from run names or fabricate missing fields.
        candidates += sorted(manifests.glob(run_id + '__*split*.json'))
        nodes = []
        for path in candidates:
            raw = path.read_bytes()
            sources[path.name] = hashlib.sha256(raw).hexdigest()
            for key, location, value in walk(json.loads(raw)):
                nodes.append((key, path.name, location, value))
        for offset, (requirement, aliases) in enumerate(FIELDS.items(), 162):
            matches = [(file, location, value) for key, file, location, value in nodes if key in aliases]
            populated = [(file, location, value) for file, location, value in matches
                         if value is not None and value != [] and value != {} and value != '']
            state = 'candidate_metadata_present' if populated else 'explicit_null_or_empty_only' if matches else 'not_located'
            rows.append({'run_id': run_id, 'document': 'docx_goal.txt', 'document_line': offset,
                'required_saved_information': requirement, 'inventory_status': state,
                'evidence_locations_json': json.dumps([{'file': file, 'json_path': location,
                    'sha256': sources[file], 'explicit_null': value is None} for file, location, value in matches]),
                'boundary': 'Candidate metadata locations only: aliases may denote different semantics. Null requires capability justification; presence does not establish correctness, source-only fitting, complete configuration or executable reproduction.'})
    output.mkdir(parents=True, exist_ok=True)
    target = output / 'REPRODUCIBILITY_METADATA_INVENTORY.csv'
    with target.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {'status': 'metadata_inventory_scientific_acceptance_unproven', 'completion_proven': False,
        'saved_run_manifests': len(runs), 'required_field_inventory_rows': len(rows),
        'sources_sha256': sources, 'inventory_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
        'by_requirement': {field: dict(Counter(row['inventory_status'] for row in rows
            if row['required_saved_information'] == field)) for field in FIELDS},
        'limitations': ['Semantic alias matches are candidate evidence only',
            'Not a training rerun or prediction replay', 'No historical manifest modifications',
            'Unsupported native labels remain missing or explicit null, never inferred from filenames',
            'No global completion percentage derived from field presence']}
    (output / 'REPRODUCIBILITY_METADATA_AUDIT.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('status', 'saved_run_manifests', 'required_field_inventory_rows', 'by_requirement')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('manifests', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.manifests, args.output)
