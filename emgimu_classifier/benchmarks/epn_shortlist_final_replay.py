"""Independent EPN final-user replay of the frozen four-family shortlist."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from emgimu.datasets.epn612 import load_epn612_windows
from emgimu.feature_bank.epn_study import _metrics, aggregate_trials


FINAL_USERS = (19, 20, 21)
VALIDATION_USERS = (16, 17, 18)
EXPECTED_FULL = ('F0', 'F3_Ring', 'F2b_CSP', 'F6_IMU')


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def disjoint_trials(train: list[str], validation: list[str], final: list[str]) -> None:
    groups = [set(train), set(validation), set(final)]
    if any(len(group) != len(values) for group, values in zip(groups, (train, validation, final))):
        raise ValueError('duplicate trial identities in a split')
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError('train, validation and final trials overlap')


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def replay(archive: Path, source: Path, output: Path, compact_path: Path) -> dict:
    manifest_path = source / 'run_manifest.json'
    state_path = source / 'fitted_states.pkl'
    saved_path = source / 'heldout_predictions.npz'
    split_path = source / 'split_trial_ids.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    splits = json.loads(split_path.read_text(encoding='utf-8'))
    if (tuple(manifest['train_users']) != tuple(range(1, 16))
            or tuple(manifest['validation_users']) != VALIDATION_USERS
            or tuple(manifest['families']) != EXPECTED_FULL
            or tuple(manifest['models']['full']) != EXPECTED_FULL):
        raise ValueError('shortlist source is not the frozen development protocol')
    families, classifier_states = pickle.loads(state_path.read_bytes())
    if set(families) != set(EXPECTED_FULL) or set(classifier_states) != set(manifest['models']):
        raise ValueError('saved family/classifier states do not match the manifest')
    before = hashlib.sha256(pickle.dumps((families, classifier_states))).hexdigest()
    predictions, scores = {}, []
    print('validation: loading users 16-18 and checking frozen probabilities', flush=True)
    validation = load_epn612_windows(archive, users=VALIDATION_USERS)
    final = load_epn612_windows(archive, users=FINAL_USERS)
    disjoint_trials(splits['train'], splits['validation'], np.unique(final.trials).tolist())
    with np.load(saved_path, allow_pickle=False) as saved:
        maximum_error = 0.0
        for phase, target in (('validation', validation), ('final', final)):
            transformed = {}
            labels = users = trials = weights = None
            for name, family in families.items():
                values, y, u, ids, w = aggregate_trials(family.transform(target.batch), target)
                transformed[name] = values
                if labels is None:
                    labels, users, trials, weights = y, u, ids, w
            if phase == 'validation':
                for key, array in (('labels', labels), ('users', users), ('trials', trials)):
                    np.testing.assert_array_equal(saved[key], array)
                if set(trials) != set(splits['validation']):
                    raise ValueError('validation trials changed from the source split')
            elif set(users.tolist()) != set(FINAL_USERS):
                raise ValueError('final users are not the prespecified independent cohort')
            for model_name, members in manifest['models'].items():
                scaler, model = classifier_states[model_name]
                values = np.concatenate([transformed[name] for name in members], axis=1)
                probability = model.predict_proba(scaler.transform(values))
                if phase == 'validation':
                    expected = saved[model_name]
                    if probability.shape != expected.shape:
                        raise ValueError(f'validation probability shape changed: {model_name}')
                    maximum_error = max(maximum_error, float(np.max(np.abs(probability - expected))))
                else:
                    predictions[model_name] = probability
                    for subject in ('ALL', *FINAL_USERS):
                        mask = np.ones(len(labels), dtype=bool) if subject == 'ALL' else users == subject
                        scores.append({'dataset': 'epn612', 'phase': 'final', 'subject': subject,
                                       'condition': 'cross_user', 'model': model_name,
                                       'feature_family': '+'.join(members), 'calibration_budget': 0,
                                       'feature_dimension': values.shape[1],
                                       **_metrics(labels[mask], probability[mask], weights[mask])})
            if phase == 'validation' and maximum_error > 1e-10:
                raise ValueError(f'frozen validation probabilities changed: {maximum_error}')
            if phase == 'final':
                final_labels, final_users, final_trials = labels, users, trials
        if hashlib.sha256(pickle.dumps((families, classifier_states))).hexdigest() != before:
            raise ValueError('source fitted state mutated during evaluation')
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    ablations = []
    for subject in ('ALL', *FINAL_USERS):
        cell = [row for row in scores if row['subject'] == subject]
        reference = next(row for row in cell if row['model'] == 'full')
        for row in cell:
            if row['model'] == 'full' or row['model'].startswith('leave_out_'):
                ablations.append({**row, 'removed_family': 'NONE' if row['model'] == 'full'
                                  else row['model'].removeprefix('leave_out_'),
                                  'delta_macro_f1_vs_full': row['macro_f1'] - reference['macro_f1'],
                                  'delta_logloss_vs_full': reference['log_loss'] - row['log_loss']})
    save_csv(output / 'feature_family_results.csv', scores)
    save_csv(output / 'ablation_full_bank.csv', ablations)
    np.savez_compressed(output / 'heldout_predictions.npz', **predictions, labels=final_labels,
                        users=final_users, trials=final_trials)
    (output / 'split_trial_ids.json').write_text(json.dumps({'source_train': splits['train'],
        'source_validation': splits['validation'], 'final': final_trials.tolist()}, indent=2), encoding='utf-8')
    sources = {path.name: sha(path) for path in (manifest_path, state_path, saved_path, split_path)}
    outputs = {path.name: sha(path) for path in output.iterdir() if path.is_file()}
    pooled = [row for row in ablations if row['subject'] == 'ALL']
    compact = {'completion_proven': False, 'protocol': 'frozen source users1-15 fit; shortlist selected on users16-18; independent final users19-21; zero target calibration',
               'source_run': source.name, 'source_sha256': sources, 'archive_sha256': sha(archive),
               'validation_replay_maximum_probability_error': maximum_error,
               'validation_replay_models': len(manifest['models']),
               'output_sha256': outputs, 'final_trials': len(final_labels),
               'final_users': list(FINAL_USERS), 'pooled_full_and_removals': pooled,
               'boundary': 'Independent final-user evaluation of the current EPN F0+Ring+CSP+IMU shortlist only; reference families are not historical RLCS/X1-H, and success on other failures or own devices is unproven.'}
    compact_path.parent.mkdir(parents=True, exist_ok=True)
    compact_path.write_text(json.dumps(compact, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'final_trials': len(final_labels), 'models': len(manifest['models']),
                      'validation_max_error': maximum_error}), flush=True)
    return compact


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', type=Path)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('compact_output', type=Path)
    args = parser.parse_args()
    replay(args.archive, args.source, args.output, args.compact_output)
