"""Trial-matched frozen EPN Core versus fixed Core+affine-SPD F7 fusion."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from emgimu.datasets.epn612 import GESTURES
from emgimu.feature_bank.epn_study import _metrics


PHASE_USERS = {"validation": (16, 17, 18), "descriptive_final": (19, 20, 21)}
BUDGETS = (1, 2, 5)
ARMS = ("Core", "Core_plus_Uniform", "Affine_F7", "Core_plus_Affine_F7")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _probabilities(matrix, name: str) -> np.ndarray:
    p = np.asarray(matrix, dtype=float)
    if (p.ndim != 2 or p.shape[1] != len(GESTURES) or not np.isfinite(p).all()
            or np.any(p < 0) or not np.allclose(p.sum(axis=1), 1., atol=1e-9)):
        raise ValueError(f"invalid {name} class probabilities")
    return p


def matched(core: dict, selected: list[dict], anchor: list[dict],
            phase: str, budget: int):
    if phase not in PHASE_USERS or budget not in BUDGETS:
        raise ValueError("unsupported phase/budget")
    labels = np.asarray(core["labels"], dtype=int)
    users = np.asarray(core["users"], dtype=int)
    trials = np.asarray(core["trials"]).astype(str)
    probabilities = _probabilities(core["full"], "Core")
    if not (len(labels) == len(users) == len(trials) == len(probabilities)):
        raise ValueError("Core arrays have unequal lengths")
    keys = [(int(user), trial) for user, trial in zip(users, trials)]
    if len(set(keys)) != len(keys) or set(users) != set(PHASE_USERS[phase]):
        raise ValueError("Core trial identities/users are invalid")
    calibration = {(int(row["subject"]), row["trial_id"]): int(row["label"])
                   for row in selected if row["phase"] == phase and int(row["shots_per_class"]) == budget}
    if len(calibration) != len(PHASE_USERS[phase]) * len(GESTURES) * budget:
        raise ValueError("calibration budget or duplicate trial mismatch")
    if not set(calibration) <= set(keys):
        raise ValueError("calibration trial absent from frozen Core")
    truth_map = dict(zip(keys, labels))
    if any(truth_map[key] != label for key, label in calibration.items()):
        raise ValueError("Core and calibration labels disagree")
    expected = set(keys) - set(calibration)
    anchor_map = {(int(row["subject"]), row["trial_id"]): row for row in anchor
                  if row["phase"] == phase and int(row["shots_per_class"]) == budget}
    if set(anchor_map) != expected or len(anchor_map) != len(expected):
        raise ValueError("F7 predictions do not equal noncalibration Core trials")
    indices = np.asarray([i for i, key in enumerate(keys) if key in expected])
    kept = [keys[i] for i in indices]
    if any(int(anchor_map[key]["true_label"]) != labels[i]
           for i, key in zip(indices, kept)):
        raise ValueError("Core/F7 evaluation labels disagree")
    f7 = _probabilities([[float(anchor_map[key][f"p_{name}"]) for name in GESTURES]
                         for key in kept], "Affine F7")
    core_p = probabilities[indices]
    return kept, labels[indices], users[indices], {
        "Core": core_p,
        "Core_plus_Uniform": (core_p + np.full_like(core_p, 1 / len(GESTURES))) / 2,
        "Affine_F7": f7,
        "Core_plus_Affine_F7": (core_p + f7) / 2,
    }


def run(core_validation: Path, core_final: Path, core_audit: Path,
        affine_dir: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    prior = json.loads(core_audit.read_text(encoding="utf-8"))
    affine = json.loads((affine_dir / "results.json").read_text(encoding="utf-8"))
    expected_core = {"validation": prior["source_sha256"]["heldout_predictions.npz"],
                     "descriptive_final": prior["output_sha256"]["heldout_predictions.npz"]}
    for phase, path in (("validation", core_validation), ("descriptive_final", core_final)):
        if sha(path) != expected_core[phase]:
            raise ValueError(f"{phase} Core prediction hash differs from frozen audit")
    for name, digest in affine["output_sha256"].items():
        if sha(affine_dir / name) != digest:
            raise ValueError(f"affine F7 input hash differs: {name}")
    selection = read_csv(affine_dir / "calibration_trial_ids.csv")
    anchor = read_csv(affine_dir / "trial_predictions.csv")
    rows, prediction_rows = [], []
    for phase, path in (("validation", core_validation), ("descriptive_final", core_final)):
        with np.load(path, allow_pickle=False) as loaded:
            core = {key: loaded[key].copy() for key in ("labels", "users", "trials", "full")}
        for budget in BUDGETS:
            keys, truth, users, arms = matched(core, selection, anchor, phase, budget)
            for subject in ("ALL", *PHASE_USERS[phase]):
                take = np.ones(len(truth), dtype=bool) if subject == "ALL" else users == subject
                for arm in ARMS:
                    rows.append(dict(phase=phase, subject=subject, shots_per_class=budget,
                                     arm=arm, evaluation_trials=int(take.sum()),
                                     **_metrics(truth[take], arms[arm][take], np.ones(take.sum()))))
            for (user, trial), label, core_p, f7_p, combined in zip(
                    keys, truth, arms["Core"], arms["Affine_F7"], arms["Core_plus_Affine_F7"]):
                prediction_rows.append(dict(phase=phase, subject=user, shots_per_class=budget,
                                            trial_id=trial, true_label=int(label),
                                            **{f"p_core_{name}": float(core_p[i]) for i, name in enumerate(GESTURES)},
                                            **{f"p_affine_{name}": float(f7_p[i]) for i, name in enumerate(GESTURES)},
                                            **{f"p_mix_{name}": float(combined[i]) for i, name in enumerate(GESTURES)}))
            print(f"[{phase}] {budget}-shot: {len(truth)} exact matched trials", flush=True)
    output_dir.mkdir(parents=True)
    shutil.copyfile(core_validation, output_dir / 'core_validation.npz')
    shutil.copyfile(core_final, output_dir / 'core_final.npz')
    with (output_dir / "trial_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader(); writer.writerows(prediction_rows)
    with (output_dir / "scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    increments, complementarity = [], []
    for phase in PHASE_USERS:
        for budget in BUDGETS:
            reference = next(r for r in rows if (r['phase'], r['shots_per_class'], r['subject'], r['arm']) ==
                             (phase, budget, 'ALL', 'Core'))
            candidate = next(r for r in rows if (r['phase'], r['shots_per_class'], r['subject'], r['arm']) ==
                             (phase, budget, 'ALL', 'Core_plus_Affine_F7'))
            uniform = next(r for r in rows if (r['phase'], r['shots_per_class'], r['subject'], r['arm']) ==
                           (phase, budget, 'ALL', 'Core_plus_Uniform'))
            increments.append(dict(phase=phase, shots_per_class=budget,
                                   evaluation_trials=reference['evaluation_trials'],
                                   delta_logloss=reference['log_loss'] - candidate['log_loss'],
                                   delta_macro_f1=candidate['macro_f1'] - reference['macro_f1'],
                                   delta_brier=reference['brier'] - candidate['brier'],
                                   delta_logloss_over_uniform=uniform['log_loss'] - candidate['log_loss'],
                                   delta_macro_f1_over_uniform=candidate['macro_f1'] - uniform['macro_f1'],
                                   delta_brier_over_uniform=uniform['brier'] - candidate['brier']))
            paired = [r for r in prediction_rows if r['phase'] == phase and r['shots_per_class'] == budget]
            truth = np.asarray([r['true_label'] for r in paired], dtype=int)
            core_p = np.asarray([[r[f'p_core_{name}'] for name in GESTURES] for r in paired])
            f7_p = np.asarray([[r[f'p_affine_{name}'] for name in GESTURES] for r in paired])
            core_correct = core_p.argmax(axis=1) == truth
            f7_correct = f7_p.argmax(axis=1) == truth
            core_error, f7_error = (~core_correct).astype(float), (~f7_correct).astype(float)
            correlation = (float(np.corrcoef(core_error, f7_error)[0, 1])
                           if np.std(core_error) > 0 and np.std(f7_error) > 0 else None)
            complementarity.append(dict(
                phase=phase, shots_per_class=budget, trials=len(paired),
                error_correlation=correlation,
                disagreement_rate=float(np.mean(core_p.argmax(axis=1) != f7_p.argmax(axis=1))),
                f7_correct_core_wrong=int(np.sum(f7_correct & ~core_correct)),
                f7_wrong_core_correct=int(np.sum(~f7_correct & core_correct)),
                p_f7_correct_core_wrong=float(np.mean(f7_correct & ~core_correct)),
                p_f7_wrong_core_correct=float(np.mean(~f7_correct & core_correct))))
    result = dict(study="matched frozen Core versus affine-SPD F7 fixed late-fusion increment",
                  core="source-frozen F0+F3_Ring+F2b_CSP+F6_IMU; no target-user fitting",
                  f7="calibration-only exact affine-invariant SPD class prototypes",
                  mixture="0.5 Core + 0.5 affine F7; 0.5 Core + 0.5 uniform control; weights fixed before matching or final readout",
                  evidence_boundary="same native evaluation trials per budget; validation subjects previously selected Core; descriptive final subjects previously inspected; no newly fitted concatenated Core, independent prospective holdout, or device claim",
                  input_sha256={"core_validation": sha(core_validation), "core_final": sha(core_final),
                                "core_audit": sha(core_audit), "affine_results": sha(affine_dir / 'results.json'),
                                **{f"affine_{name}": sha(affine_dir / name) for name in affine['output_sha256']}},
                  source_sha256=sha(Path(__file__)),
                  pooled_increments=increments,
                  error_complementarity=complementarity,
                  output_sha256={name: sha(output_dir / name) for name in
                                 ('core_validation.npz', 'core_final.npz',
                                  'scores.csv', 'trial_predictions.csv')})
    (output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('core_validation', 'core_final', 'core_audit', 'affine_dir', 'output_dir'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.core_validation, args.core_final, args.core_audit, args.affine_dir, args.output_dir)
