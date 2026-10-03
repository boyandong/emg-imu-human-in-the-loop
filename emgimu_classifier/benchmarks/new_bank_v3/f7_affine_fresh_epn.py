"""Once-only fresh EPN612 users22–31 trial-matched Core/F7 readout."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import recall_score

from benchmarks.new_bank_v3.f7_affine_epn import _select, _temperature
from emgimu.datasets.epn612 import GESTURES, load_epn612_windows
from emgimu.feature_bank import AffineSpdPrototypeAnchor, document_spd_matrices
from emgimu.feature_bank.epn_study import _metrics, aggregate_trials


FRESH_USERS = tuple(range(22, 32))
VALIDATION_USERS = (16, 17, 18)
BUDGETS = (1, 2, 5)
FAMILIES = ('F0', 'F3_Ring', 'F2b_CSP', 'F6_IMU')
ARMS = ('Core', 'Core_plus_Uniform', 'Affine_F7', 'Core_plus_Affine_F7')
EXPECTED_SHA = {
    'archive': '4ee8db037385e7bee1e6ac6f9e9eea4f0869e25f7825f5eb7e5be0dff4f93c21',
    'run_manifest.json': '36d86cdb65d03936073b21295bb6db0d4d196c8f271dc82ffe45e01e098663db',
    'fitted_states.pkl': '08e7e4e4af1644446911633411743306be33b3c6d08d584d5967252f4d199aef',
    'heldout_predictions.npz': '8c2a9f8c616e7c836d788ba9b3dcbd5cff92d706741aed576efc33a8445f6b31',
    'split_trial_ids.json': '0e31b7957b2dabd9e480704858b05ea313d995c6955b41081c21c2080143ae57',
    'f7_affine_epn.py': 'ee891ecba8d10b72eb4321d7e1dc596953061c4dca8ec49f3b35fa170643c45a',
    'affine_spd_anchor.py': '4ddd02d2260448aad4f3ab98e013b7213b8ac2a31e93bd89975ec40049239808',
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def core_trials(target, families, state):
    parts = {}
    labels = users = trials = weights = None
    for name in FAMILIES:
        values, y, u, ids, w = aggregate_trials(families[name].transform(target.batch), target)
        parts[name] = values
        if labels is None:
            labels, users, trials, weights = y, u, ids, w
        else:
            for a, b in ((labels, y), (users, u), (trials, ids)):
                np.testing.assert_array_equal(a, b)
    scaler, model = state
    probability = model.predict_proba(scaler.transform(
        np.concatenate([parts[name] for name in FAMILIES], axis=1)))
    if not np.allclose(probability.sum(axis=1), 1., atol=1e-9):
        raise ValueError('Core probabilities do not sum to one')
    return labels, users, trials.astype(str), probability


def paired_arms(core_probability: np.ndarray, f7_probability: np.ndarray) -> dict[str, np.ndarray]:
    if core_probability.shape != f7_probability.shape:
        raise ValueError('Core and F7 trial arrays differ')
    for p in (core_probability, f7_probability):
        if not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(axis=1), 1., atol=1e-9):
            raise ValueError('invalid class probabilities')
    return {'Core': core_probability,
            'Core_plus_Uniform': (core_probability + 1. / len(GESTURES)) / 2.,
            'Affine_F7': f7_probability,
            'Core_plus_Affine_F7': (core_probability + f7_probability) / 2.}


def primary_guard(scores: list[dict]) -> dict:
    chosen = {(str(r['subject']), r['arm']): r for r in scores
              if int(r['shots_per_class']) == 5}
    needed = {('ALL', arm) for arm in ARMS}
    needed |= {(str(user), arm) for user in FRESH_USERS for arm in ('Core', 'Core_plus_Affine_F7')}
    if not needed <= set(chosen) or len(chosen) != 4 * (len(FRESH_USERS) + 1):
        raise ValueError('missing or duplicated primary score cells')
    core, mix, uniform = (chosen[('ALL', arm)] for arm in
                          ('Core', 'Core_plus_Affine_F7', 'Core_plus_Uniform'))
    user_wins = sum(chosen[(str(user), 'Core_plus_Affine_F7')]['log_loss'] <
                    chosen[(str(user), 'Core')]['log_loss'] for user in FRESH_USERS)
    criteria = {
        'pooled_logloss_lower': mix['log_loss'] < core['log_loss'],
        'pooled_macro_f1_nonworse': mix['macro_f1'] >= core['macro_f1'],
        'pooled_brier_lower': mix['brier'] < core['brier'],
        'logloss_better_than_uniform': mix['log_loss'] < uniform['log_loss'],
        'at_least_7_of_10_user_logloss_wins': user_wins >= 7,
    }
    return {'criteria': criteria, 'conjunction_passed': all(criteria.values()),
            'user_logloss_wins': user_wins,
            'delta_logloss': core['log_loss'] - mix['log_loss'],
            'delta_macro_f1': mix['macro_f1'] - core['macro_f1'],
            'delta_brier': core['brier'] - mix['brier'],
            'delta_logloss_over_uniform': uniform['log_loss'] - mix['log_loss']}


def score(truth: np.ndarray, probability: np.ndarray) -> dict:
    result = _metrics(truth, probability, np.ones(len(truth)))
    per_class = recall_score(truth, probability.argmax(axis=1),
                             labels=list(range(len(GESTURES))), average=None, zero_division=0)
    result['per_class_recall_json'] = json.dumps(dict(zip(GESTURES, map(float, per_class))),
                                                 separators=(',', ':'))
    return result


def run(archive: Path, source: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    root = Path(__file__).resolve().parents[2]
    for name, expected in EXPECTED_SHA.items():
        path = (archive if name == 'archive' else
                Path(__file__).with_name(name) if name == 'f7_affine_epn.py' else
                root / 'src/emgimu/feature_bank/affine_spd_anchor.py'
                if name == 'affine_spd_anchor.py' else source / name)
        actual = sha(path)
        if actual != expected:
            raise ValueError(f'frozen input hash mismatch: {name}: {actual}')
    manifest = json.loads((source / 'run_manifest.json').read_text(encoding='utf-8'))
    splits = json.loads((source / 'split_trial_ids.json').read_text(encoding='utf-8'))
    if (tuple(manifest['train_users']) != tuple(range(1, 16)) or
            tuple(manifest['validation_users']) != VALIDATION_USERS or
            tuple(manifest['models']['full']) != FAMILIES or
            tuple(manifest['families']) != FAMILIES):
        raise ValueError('frozen source manifest differs from protocol')
    families, classifiers = pickle.loads((source / 'fitted_states.pkl').read_bytes())
    if set(families) != set(FAMILIES) or 'full' not in classifiers:
        raise ValueError('frozen source state differs from protocol')
    print('validation: replaying frozen Core before reading fresh users', flush=True)
    validation = load_epn612_windows(archive, users=VALIDATION_USERS)
    val_y, val_users, val_trials, val_p = core_trials(validation, families, classifiers['full'])
    with np.load(source / 'heldout_predictions.npz', allow_pickle=False) as saved:
        for actual, expected in ((val_y, saved['labels']), (val_users, saved['users']),
                                 (val_trials, saved['trials'].astype(str))):
            np.testing.assert_array_equal(actual, expected)
        max_error = float(np.max(np.abs(val_p - saved['full'])))
    if max_error > 1e-10 or set(val_trials) != set(splits['validation']):
        raise ValueError(f'frozen Core validation replay differs: {max_error}')
    del validation
    source_trials = set(splits['train']) | set(splits['validation'])
    if len(source_trials) != len(splits['train']) + len(splits['validation']):
        raise ValueError('source train/validation trial overlap')
    print(f'validation: maximum probability error {max_error:.3g}', flush=True)

    scores, predictions, selections = [], [], []
    all_fresh_trials = set()
    for user in FRESH_USERS:
        print(f'fresh user {user}: loading and transforming', flush=True)
        target = load_epn612_windows(archive, users=(user,))
        labels, users, trials, core_p = core_trials(target, families, classifiers['full'])
        if set(users.tolist()) != {user} or len(set(trials)) != len(trials):
            raise ValueError(f'user {user} identities changed')
        if set(trials) & (source_trials | all_fresh_trials):
            raise ValueError('fresh trials overlap source or prior user')
        all_fresh_trials.update(trials)
        trial_index = {trial: i for i, trial in enumerate(trials)}
        matrices = document_spd_matrices(target.batch.emg)
        selected = _select(target.trials, target.labels, user)
        for budget in BUDGETS:
            cal = selected[budget]
            calibration = np.isin(target.trials, list(cal))
            anchor = AffineSpdPrototypeAnchor().fit(
                matrices[calibration], target.labels[calibration], target.trials[calibration])
            if anchor.classes_.tolist() != list(range(len(GESTURES))):
                raise ValueError('F7 calibration classes changed')
            evaluation = ~calibration
            ids, distances = anchor.transform(matrices[evaluation], target.trials[evaluation])
            tau = _temperature(anchor.prototypes_)
            logits = -distances / tau
            logits -= logits.max(axis=1, keepdims=True)
            f7_p = np.exp(logits)
            f7_p /= f7_p.sum(axis=1, keepdims=True)
            kept = [str(t) for t in ids]
            if len(kept) != len(trials) - len(cal) or set(kept) != set(trials) - cal:
                raise ValueError('F7/Core trial match failed')
            indices = [trial_index[t] for t in kept]
            truth = labels[indices]
            for trial in sorted(cal):
                i = trial_index[trial]
                selections.append(dict(subject=user, shots_per_class=budget,
                                       trial_id=trial, label=int(labels[i])))
            arms = paired_arms(core_p[indices], f7_p)
            for arm in ARMS:
                scores.append(dict(subject=str(user), shots_per_class=budget, arm=arm,
                                   evaluation_trials=len(truth), tau_from_calibration=tau,
                                   **score(truth, arms[arm])))
            for trial, label, cp, fp in zip(kept, truth, arms['Core'], arms['Affine_F7']):
                predictions.append(dict(subject=user, shots_per_class=budget, trial_id=trial,
                                        true_label=int(label),
                                        **{f'p_core_{name}': float(cp[i]) for i, name in enumerate(GESTURES)},
                                        **{f'p_f7_{name}': float(fp[i]) for i, name in enumerate(GESTURES)}))
            print(f'fresh user {user}, {budget}-shot: {len(truth)} held-out trials', flush=True)
        del target, matrices
    if len(all_fresh_trials) != len(set(all_fresh_trials)):
        raise ValueError('fresh trial duplicate')
    for budget in BUDGETS:
        rows = [r for r in predictions if r['shots_per_class'] == budget]
        truth = np.asarray([r['true_label'] for r in rows], dtype=int)
        core = np.asarray([[r[f'p_core_{name}'] for name in GESTURES] for r in rows])
        f7 = np.asarray([[r[f'p_f7_{name}'] for name in GESTURES] for r in rows])
        arms = paired_arms(core, f7)
        for arm in ARMS:
            scores.append(dict(subject='ALL', shots_per_class=budget, arm=arm,
                               evaluation_trials=len(truth), tau_from_calibration='per user',
                               **score(truth, arms[arm])))
    primary = primary_guard(scores)
    complementarity = []
    for budget in BUDGETS:
        rows = [r for r in predictions if r['shots_per_class'] == budget]
        truth = np.asarray([r['true_label'] for r in rows], dtype=int)
        core = np.asarray([[r[f'p_core_{name}'] for name in GESTURES] for r in rows])
        f7 = np.asarray([[r[f'p_f7_{name}'] for name in GESTURES] for r in rows])
        core_correct = core.argmax(axis=1) == truth
        f7_correct = f7.argmax(axis=1) == truth
        complementarity.append(dict(shots_per_class=budget, trials=len(rows),
                                    f7_correct_core_wrong=int(np.sum(f7_correct & ~core_correct)),
                                    core_correct_f7_wrong=int(np.sum(core_correct & ~f7_correct)),
                                    prediction_disagreement_rate=float(np.mean(
                                        core.argmax(axis=1) != f7.argmax(axis=1)))))
    output.mkdir(parents=True)
    save_csv(output / 'scores.csv', scores)
    save_csv(output / 'trial_predictions.csv', predictions)
    save_csv(output / 'calibration_trial_ids.csv', selections)
    report = {
        'study': 'precommitted fresh EPN612 users22-31 frozen-Core/F7 trial-matched holdout',
        'protocol': 'F7_AFFINE_FRESH_PROTOCOL.md', 'fresh_users': FRESH_USERS,
        'source_users': manifest['train_users'], 'prior_validation_users': VALIDATION_USERS,
        'budgets': BUDGETS, 'weight': 0.5,
        'validation_replay_maximum_probability_error': max_error,
        'primary_five_shot': primary,
        'error_complementarity': complementarity,
        'source_sha256': EXPECTED_SHA,
        'script_sha256': sha(Path(__file__)),
        'protocol_sha256': sha(Path(__file__).with_name('F7_AFFINE_FRESH_PROTOCOL.md')),
        'output_sha256': {name: sha(output / name) for name in
                          ('scores.csv', 'trial_predictions.csv', 'calibration_trial_ids.csv')},
        'boundary': 'Fresh public heldout EPN subjects under precommitted protocol; not own-device/live-hardware evidence',
    }
    (output / 'results.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'primary five-shot guard: {primary}', flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', type=Path)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    run(args.archive, args.source, args.output)
