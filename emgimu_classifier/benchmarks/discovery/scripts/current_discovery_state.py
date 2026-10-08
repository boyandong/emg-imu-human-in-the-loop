"""Current versioned discovery entry point, preserving SHA-bound historical files."""
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    paths = ['DATASET_MANIFEST.json', 'DISCOVERY_DELIVERY_AUDIT.json',
             'DS2_V9_FORCE_LABEL_AUDIT.json', 'DS2_V9_FORCE_TRIAL_JOIN.csv',
             'public_ds2_force_v9/RESULTS.json', 'public_ds2_force_v9/TRIAL_PREDICTIONS.csv',
             'scripts/current_discovery_state.py', 'SECONDARY_LICENSE_METADATA_V1.json',
             'scripts/fetch_secondary_license_metadata.py', 'SECONDARY_PAPER_REVIEW_V1.json',
             'SECONDARY_PRIMARY_REVIEW_V1.json']
    old = json.loads((HERE / paths[0]).read_text(encoding='utf8'))
    labels = json.loads((HERE / paths[2]).read_text(encoding='utf8'))
    result = json.loads((HERE / paths[4]).read_text(encoding='utf8'))
    if labels['trial_join_csv_sha256'] != sha(HERE / paths[3]):
        raise ValueError('DS2 trial join changed')
    if result['source_sha256']['force_label_audit'] != sha(HERE / paths[2]):
        raise ValueError('DS2 study used a different force-label audit')
    if result['trial_predictions_sha256'] != sha(HERE / paths[5]):
        raise ValueError('DS2 study predictions changed')
    with (HERE / paths[3]).open(encoding='utf8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    indices = [int(row['raw_trial_index_zero_based']) for row in rows]
    if indices != list(range(labels['raw_trials'])):
        raise ValueError('DS2 join is not a complete unique ordered raw-trial index')
    excluded_subject = [int(row['raw_trial_index_zero_based']) for row in rows
                        if row['subject_folder'] == 'N/A']
    excluded_gesture = [int(row['raw_trial_index_zero_based']) for row in rows
                        if row['gesture_code_if_uniform'] == 'N/A']
    if (excluded_subject != labels['unmatched_subject_trials'] or
            excluded_gesture != labels['mixed_gesture_trials']):
        raise ValueError('DS2 exclusions disagree with force-label audit')
    eligible = [row for row in rows if row['subject_folder'] != 'N/A'
                and row['gesture_code_if_uniform'] != 'N/A']
    active = [row for row in eligible if row['gesture_code_if_uniform'] in ('0', '1', '2', '3')]
    if (len(eligible) != labels['eligible_subject_gesture_force_trials'] or
            len(active) != result['eligible_active_trials']):
        raise ValueError('DS2 current cohort counts disagree')
    datasets = [dict(dataset) for dataset in old['datasets']]
    review = json.loads((HERE / paths[-1]).read_text(encoding='utf8'))
    reviewed = {entry['id']: entry for entry in review['datasets']}
    if len(reviewed) != len(review['datasets']):
        raise ValueError('Duplicate secondary dataset review')
    for dataset in datasets:
        if dataset['id'] in reviewed:
            dataset['primary_metadata_review'] = reviewed[dataset['id']]
            license_name = reviewed[dataset['id']]['license_verified']
            if license_name is not None and license_name != dataset['license']:
                dataset['recorded_license'] = dataset['license']
                dataset['license'] = license_name
    ds2 = next(dataset for dataset in datasets if dataset['id'] == 'ds2_force')
    ds2.update({
        'status': 'public_v8_raw_with_verified_publisher_v9_subjective_force_labels',
        'per_trial_force_labels': 'verified_exact_window_to_raw_trial_join',
        'force_label_publisher_version': labels['publisher_version'],
        'force_label_audit': paths[2], 'force_trial_join': paths[3],
        'force_labelled_raw_trials': len(rows),
        'subject_gesture_force_eligible_trials': len(eligible),
        'active_study_trials': len(active),
        'current_force_study': paths[4],
        'note': 'Versioned independent public-data experiment. Missing legacy B0/X1/X2 code is superseded by user instruction and is not a prerequisite. Force labels are subjective instructed categories, not measured mechanical force. Final cohort results remain descriptive; no own-device or deployment claim.'})
    output = {
        'schema': 'current_discovery_state_v1',
        'historical_manifest': paths[0],
        'source_sha256': {path: sha(HERE / path) for path in paths},
        'datasets': datasets,
        'superseded_conclusions': [
            {'source': 'DATASET_MANIFEST.json:ds2_force.per_trial_force_labels',
             'old': 'unverified', 'current': ds2['per_trial_force_labels']},
            {'source': 'DISCOVERY_DELIVERY_AUDIT.json:limitations',
             'old': 'No verified per-trial DS2 force key or force-condition scores.',
             'current': 'Publisher v9 force keys and independent force-category scores now exist.'},
            {'source': 'Historical baseline recovery', 'old': 'Wait for unavailable legacy experiments.',
             'current': 'Superseded by user-authorized versioned independent implementation.'}],
        'remaining': ['Real multi-user/day eight-channel and hardware validation.',
                      'Seven secondary primary metadata reviews exist; full-paper, unresolved dataset-license and native physical-layout checks remain explicit per entry.',
                      'Reported DS2 license conflict is retained; no new license determination.',
                      'This entry point does not rehash multi-GB raw archives or authenticate physical clocks.'],
        'completion_proven': False,
        'scope': 'Current local evidence overlay. Historical SHA-bound artifacts are retained as snapshots. Dataset entries other than DS2 retain their recorded status and are not freshly externally verified.'}
    (HERE / 'CURRENT_DISCOVERY_STATE.json').write_text(json.dumps(output, indent=2) + '\n', encoding='utf8')
    print(json.dumps({'raw_trials': len(rows), 'eligible_trials': len(eligible), 'active_trials': len(active)}))
    return output


if __name__ == '__main__':
    build()
