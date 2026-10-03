"""New-version, calibration-only affine-SPD F7 EPN612 trial screen.

This is a standalone candidate screen, not an incremental Core comparison.
The old source-fitted tangent-arm files are never overwritten.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from emgimu.datasets.epn612 import GESTURES, load_epn612_windows
from emgimu.feature_bank import AffineSpdPrototypeAnchor, document_spd_matrices
from emgimu.feature_bank.epn_study import _metrics
from emgimu.feature_bank.affine_spd_anchor import affine_spd_distance


SEED = 20261003
PHASE_USERS = {"validation": (16, 17, 18), "descriptive_final": (19, 20, 21)}
BUDGETS = (1, 2, 5)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _select(trials: np.ndarray, labels: np.ndarray, user: int) -> dict[int, set[str]]:
    truth = {}
    for trial in np.unique(trials):
        values = np.unique(labels[trials == trial])
        if len(values) != 1:
            raise ValueError(f"mixed label in native trial {trial}")
        truth[str(trial)] = int(values[0])
    for label in range(len(GESTURES)):
        members = np.asarray(sorted(t for t, value in truth.items() if value == label))
        if len(members) <= max(BUDGETS):
            raise ValueError(f"insufficient disjoint trials for user {user}, class {label}")
    # Nested budgets share a class-specific deterministic order.
    chosen = {}
    for budget in BUDGETS:
        rng = np.random.default_rng(SEED + user)
        chosen[budget] = set()
        for label in range(len(GESTURES)):
            members = np.asarray(sorted(t for t, value in truth.items() if value == label))
            chosen[budget].update(rng.permutation(members)[:budget].tolist())
    assert chosen[1] <= chosen[2] <= chosen[5]
    return chosen


def _temperature(prototypes: np.ndarray) -> float:
    between = [affine_spd_distance(prototypes[i], prototypes[j])
               for i in range(len(prototypes)) for j in range(i + 1, len(prototypes))]
    return max(float(np.median(between)), 1e-10)


def run(archive: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    scores, predictions, selections = [], [], []
    for phase, users in PHASE_USERS.items():
        print(f"[{phase}] loading users {users}", flush=True)
        data = load_epn612_windows(archive, users=users)
        for user in users:
            personal = data.take(np.flatnonzero(data.users == user))
            matrices = document_spd_matrices(personal.batch.emg)
            chosen = _select(personal.trials, personal.labels, user)
            trial_truth = {str(t): int(np.unique(personal.labels[personal.trials == t])[0])
                           for t in np.unique(personal.trials)}
            for budget in BUDGETS:
                cal_ids = chosen[budget]
                calibration = np.isin(personal.trials, list(cal_ids))
                evaluation = ~calibration
                anchor = AffineSpdPrototypeAnchor().fit(
                    matrices[calibration], personal.labels[calibration], personal.trials[calibration])
                if anchor.classes_.tolist() != list(range(len(GESTURES))):
                    raise ValueError("calibration classes do not match native ontology")
                ids, distances = anchor.transform(matrices[evaluation], personal.trials[evaluation])
                tau = _temperature(anchor.prototypes_)
                logits = -distances / tau
                logits -= logits.max(axis=1, keepdims=True)
                probability = np.exp(logits)
                probability /= probability.sum(axis=1, keepdims=True)
                truth = np.asarray([trial_truth[str(t)] for t in ids], dtype=int)
                if len(ids) != len(trial_truth) - len(cal_ids):
                    raise ValueError("trial-level evaluation count mismatch")
                metrics = _metrics(truth, probability, np.ones(len(ids)))
                scores.append(dict(phase=phase, subject=user, shots_per_class=budget,
                                   calibration_trials=len(cal_ids), evaluation_trials=len(ids),
                                   tau_from_calibration_prototype_pairs=tau, **metrics))
                for trial in sorted(cal_ids):
                    selections.append(dict(phase=phase, subject=user, shots_per_class=budget,
                                           trial_id=trial, label=trial_truth[trial]))
                for trial, label, prob in zip(ids, truth, probability):
                    predictions.append(dict(phase=phase, subject=user, shots_per_class=budget,
                                            trial_id=str(trial), true_label=int(label),
                                            predicted_label=int(np.argmax(prob)),
                                            **{f"p_{name}": float(prob[i]) for i, name in enumerate(GESTURES)}))
                print(f"[{phase}] user{user} {budget}-shot: {len(ids)} held-out trials, "
                      f"F1={metrics['macro_f1']:.4f}", flush=True)
    _write_csv(output / "subject_scores.csv", scores)
    _write_csv(output / "trial_predictions.csv", predictions)
    _write_csv(output / "calibration_trial_ids.csv", selections)
    pooled = []
    for phase in PHASE_USERS:
        for budget in BUDGETS:
            rows = [r for r in predictions if r['phase'] == phase and r['shots_per_class'] == budget]
            truth = np.asarray([r['true_label'] for r in rows])
            p = np.asarray([[r[f"p_{name}"] for name in GESTURES] for r in rows])
            pooled.append(dict(phase=phase, shots_per_class=budget, subjects=len(PHASE_USERS[phase]),
                               evaluation_trials=len(rows), **_metrics(truth, p, np.ones(len(rows)))))
    report = dict(study="F7 affine-invariant SPD personal-anchor standalone candidate",
                  source="EMG-EPN612 native labeled trainingSamples; 200 ms windows, four per trial",
                  user_phases=PHASE_USERS, budgets=BUDGETS, seed=SEED,
                  selection="classwise seeded nested complete-trial 1/2/5-shot; evaluation trials disjoint",
                  prototype="arithmetic mean of equal-weight trial F2a matrices; 1e-10 SPD ridge",
                  distance="affine-invariant matrix-log Frobenius norm",
                  temperature="median pairwise class-prototype distance from calibration only",
                  zero_shot="N/A: personal class prototypes require labeled calibration",
                  boundary="standalone candidate; final users previously inspected; no matched Core increment, source-CV temperature optimization or device claim",
                  pooled=pooled,
                  archive_sha256=_sha(archive),
                  source_sha256={"affine_spd_anchor.py": _sha(Path(__file__).resolve().parents[2] / 'src/emgimu/feature_bank/affine_spd_anchor.py'),
                                 "f7_affine_epn.py": _sha(Path(__file__))},
                  output_sha256={name: _sha(output / name) for name in
                                 ('subject_scores.csv', 'trial_predictions.csv', 'calibration_trial_ids.csv')})
    (output / 'results.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.archive, args.output)
