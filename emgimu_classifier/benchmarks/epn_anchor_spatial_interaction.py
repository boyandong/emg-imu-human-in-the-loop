"""Frozen EPN Personal Anchor x reference Spatial Coordination interaction.

The matched four arms are F0 population (B), F0+CSP population (B+C),
personalized F0 (B+A), and personalized F0+CSP (B+A+C).  Every population
provider uses its source-user OOF temperature.  Every personalized provider
uses the already saved calibration-only PersonalAnchor and the prespecified
``shots/(shots+2)`` blend.  No family, classifier, temperature, anchor, trial
selection, fusion weight, or hyperparameter is fit by this replay.
"""
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
from emgimu.feature_bank.force_nested_oof import temperature_probability


FAMILIES = ("F0", "F2b_CSP")
ARMS = ("B", "B_plus_CSP", "B_plus_Anchor", "B_plus_CSP_plus_Anchor")
PHASE_USERS = {"validation": (16, 17, 18), "final": (19, 20, 21)}
BUDGETS = (1, 2, 5)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def anchor_probability(anchor, temperature: float, features: np.ndarray) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("saved anchor temperature must be positive and finite")
    distances = anchor.transform(features)[:, :6].astype(np.float64)
    logits = -distances / temperature
    logits -= logits.max(axis=1, keepdims=True)
    probability = np.exp(logits)
    return probability / probability.sum(axis=1, keepdims=True)


def compositions(raw: dict[str, np.ndarray], personalized: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Fixed equal-provider four arms for the named interaction."""
    if set(raw) != set(FAMILIES) or set(personalized) != set(FAMILIES):
        raise ValueError("interaction requires exactly F0 and F2b_CSP providers")
    arrays = [*raw.values(), *personalized.values()]
    if (any(value.ndim != 2 or value.shape != arrays[0].shape for value in arrays)
            or any(not np.all(np.isfinite(value)) for value in arrays)):
        raise ValueError("provider probabilities must be finite and aligned")
    return {
        "B": raw["F0"],
        "B_plus_CSP": (raw["F0"] + raw["F2b_CSP"]) / 2,
        "B_plus_Anchor": personalized["F0"],
        "B_plus_CSP_plus_Anchor": (personalized["F0"] + personalized["F2b_CSP"]) / 2,
    }


def interaction(scores: dict[str, dict]) -> dict[str, float]:
    """Second difference: joint improvement beyond the two individual additions."""
    b, c, a, both = (scores[name] for name in ARMS)
    return {
        "S_negative_logloss": -both["log_loss"] + c["log_loss"] + a["log_loss"] - b["log_loss"],
        "S_macro_f1": both["macro_f1"] - c["macro_f1"] - a["macro_f1"] + b["macro_f1"],
        "S_negative_brier": -both["brier"] + c["brier"] + a["brier"] - b["brier"],
    }


def _eligible_source(source: Path, phase: str) -> tuple[dict, dict, list[dict]]:
    if phase not in PHASE_USERS:
        raise ValueError("phase must be validation or final")
    manifest = json.loads((source / "run_manifest.json").read_text(encoding="utf-8"))
    calibration = json.loads((source / "probability_calibration.json").read_text(encoding="utf-8"))
    splits = json.loads((source / "split_trial_ids.json").read_text(encoding="utf-8"))
    expected_users = PHASE_USERS[phase]
    if (manifest.get("phase") != phase or manifest.get("dataset") != "epn612"
            or manifest.get("classifier_or_family_fit") is not False
            or manifest.get("probability_calibration") != "source-user OOF temperature"
            or manifest.get("anchor_alpha") != "shots/(shots+2)"
            or tuple(calibration.get("fit_users", ())) != tuple(range(1, 16))
            or set(FAMILIES) - set(manifest.get("families", ()))):
        raise ValueError("source run is not eligible for frozen interaction replay")
    keys = {(int(row["user"]), int(row["shots"])) for row in splits}
    expected = {(user, shots) for user in expected_users for shots in (0, *BUDGETS)}
    if keys != expected:
        raise ValueError("source split grid is incomplete or contains unexpected cells")
    return manifest, calibration, splits


def evaluate(archive: Path, source: Path, output: Path, phase: str) -> None:
    if output.exists():
        raise FileExistsError(output)
    manifest, calibration, splits = _eligible_source(source, phase)
    users = PHASE_USERS[phase]
    print(f"[{phase} 1/4] loading native EPN users {users}", flush=True)
    target = load_epn612_windows(archive, users=users)
    states, anchors = pickle.loads((source / "fitted_states.pkl").read_bytes())
    state_before = pickle.dumps((states, anchors))
    features, raw_probability = {}, {}
    y = u = trials = None
    print(f"[{phase} 2/4] replaying frozen F0 and CSP providers", flush=True)
    for name in FAMILIES:
        family, scaler, model = states[name]
        trial_features, current_y, current_u, current_trials, _ = aggregate_trials(
            family.transform(target.batch), target)
        if y is None:
            y, u, trials = current_y, current_u, current_trials
        else:
            for actual, expected in ((current_y, y), (current_u, u), (current_trials, trials)):
                np.testing.assert_array_equal(actual, expected)
        features[name] = scaler.transform(trial_features)
        raw_probability[name] = temperature_probability(
            model.predict_proba(features[name]), float(calibration["temperatures"][name]))
    with np.load(source / "heldout_predictions.npz", allow_pickle=False) as saved:
        for key, value in (("labels", y), ("users", u), ("trials", trials)):
            np.testing.assert_array_equal(saved[key], value)
    if pickle.dumps((states, anchors)) != state_before:
        raise ValueError("target replay mutated frozen source or anchor state")

    score_rows, interaction_rows, prediction_arrays, split_output = [], [], {}, []
    print(f"[{phase} 3/4] scoring matched 1/2/5-shot four-arm cells", flush=True)
    split_by_cell = {(int(row["user"]), int(row["shots"])): row for row in splits}
    for shots in BUDGETS:
        pooled_truth, pooled_arms = [], {name: [] for name in ARMS}
        for user in users:
            split = split_by_cell[(user, shots)]
            calibration_trials = np.asarray(split["calibration"], dtype=trials.dtype)
            evaluation_trials = np.asarray(split["evaluation"], dtype=trials.dtype)
            cal = np.flatnonzero(np.isin(trials, calibration_trials))
            ev = np.flatnonzero(np.isin(trials, evaluation_trials))
            if (len(calibration_trials) != 6 * shots or len(cal) != len(calibration_trials)
                    or len(ev) != len(evaluation_trials) or not len(ev)
                    or set(calibration_trials.tolist()) & set(evaluation_trials.tolist())
                    or np.any(u[cal] != user) or np.any(u[ev] != user)
                    or set(np.unique(y[cal]).tolist()) != set(range(6))):
                raise ValueError(f"invalid saved calibration/evaluation split: {user}/{shots}")
            personalized = {}
            alpha = shots / (shots + 2)
            for name in FAMILIES:
                anchor, anchor_temperature = anchors[(user, shots, name)]
                # The saved object must still be bound to the exact selected calibration rows.
                expected_prototypes = np.stack([features[name][cal][y[cal] == label].mean(axis=0)
                                                for label in range(6)])
                # The saved run formed means before its float32 feature cache was
                # serialized; a fresh native replay can differ by one float32 ULP.
                np.testing.assert_allclose(anchor.prototypes_, expected_prototypes, rtol=1e-6, atol=1e-7)
                local = anchor_probability(anchor, anchor_temperature, features[name][ev])
                personalized[name] = (1 - alpha) * raw_probability[name][ev] + alpha * local
            arms = compositions({name: raw_probability[name][ev] for name in FAMILIES}, personalized)
            scores = {name: _metrics(y[ev], probability, np.ones(len(ev)))
                      for name, probability in arms.items()}
            common = {"phase": phase, "subject": user, "shots_per_class": shots,
                      "calibration_trials": len(cal), "evaluation_trials": len(ev),
                      "base": "F0", "spatial_family": "F2b_CSP_reference",
                      "anchor": "calibration_only_PersonalAnchor"}
            score_rows += [{**common, "arm": name, **score} for name, score in scores.items()]
            interaction_rows.append({**common, **interaction(scores)})
            split_output.append({"user": user, "shots": shots,
                                 "calibration": calibration_trials.tolist(),
                                 "evaluation": evaluation_trials.tolist()})
            prediction_arrays[f"{user}_{shots}_labels"] = y[ev]
            prediction_arrays[f"{user}_{shots}_trials"] = trials[ev]
            for name, probability in arms.items():
                prediction_arrays[f"{user}_{shots}_{name}"] = probability
                pooled_arms[name].append(probability)
            pooled_truth.append(y[ev])
        truth = np.concatenate(pooled_truth)
        pooled_scores = {name: _metrics(truth, np.concatenate(parts), np.ones(len(truth)))
                         for name, parts in pooled_arms.items()}
        common = {"phase": phase, "subject": "ALL", "shots_per_class": shots,
                  "calibration_trials": 6 * shots * len(users),
                  "evaluation_trials": len(truth), "base": "F0",
                  "spatial_family": "F2b_CSP_reference",
                  "anchor": "calibration_only_PersonalAnchor"}
        score_rows += [{**common, "arm": name, **score} for name, score in pooled_scores.items()]
        interaction_rows.append({**common, **interaction(pooled_scores)})

    output.mkdir(parents=True)
    save_csv(output / "arm_scores.csv", score_rows)
    save_csv(output / "interaction_results.csv", interaction_rows)
    np.savez_compressed(output / "heldout_predictions.npz", **prediction_arrays)
    (output / "split_trial_ids.json").write_text(json.dumps(split_output, indent=2) + "\n", encoding="utf-8")
    result_manifest = {
        "phase": phase, "source_run": str(source), "source_users": list(range(1, 16)),
        "target_users": list(users), "budgets": list(BUDGETS), "families": list(FAMILIES),
        "arms": list(ARMS), "fusion": "fixed equal-provider mean",
        "classifier_or_family_fit": False, "temperature_fit": False, "anchor_fit": False,
        "target_rule_selection": False, "source_state_immutable": True,
        "source_sha256": {name: sha(source / name) for name in
                          ("fitted_states.pkl", "run_manifest.json", "probability_calibration.json",
                           "split_trial_ids.json", "heldout_predictions.npz")},
        "output_sha256": {name: sha(output / name) for name in
                          ("arm_scores.csv", "interaction_results.csv", "heldout_predictions.npz",
                           "split_trial_ids.json")},
        "analysis_source_sha256": sha(Path(__file__)),
        "boundary": "Native EPN trial-level frozen F0/reference-CSP probability composition; saved calibration-only anchors and source-user OOF temperatures; not historical Spatial Coordination equivalence, feature concatenation, or own-device evidence.",
    }
    (output / "run_manifest.json").write_text(json.dumps(result_manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[{phase} 4/4] saved {len(score_rows)} arm rows and {len(interaction_rows)} interactions", flush=True)


def summarize(validation: Path, final: Path, output: Path) -> None:
    phases, pooled = {}, []
    for phase, root in (("validation", validation), ("final", final)):
        manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
        if manifest["phase"] != phase or manifest["analysis_source_sha256"] != sha(Path(__file__)):
            raise ValueError("phase output is not bound to the current frozen analysis")
        for name, expected in manifest["output_sha256"].items():
            if sha(root / name) != expected:
                raise ValueError(f"phase output changed: {phase}/{name}")
        with (root / "interaction_results.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        pooled += [{key: (float(value) if key.startswith("S_") else int(value) if key in
                           {"shots_per_class", "calibration_trials", "evaluation_trials"} else value)
                    for key, value in row.items()}
                   for row in rows if row["subject"] == "ALL"]
        phases[phase] = {"run_manifest_sha256": sha(root / "run_manifest.json"),
                         "output_sha256": manifest["output_sha256"]}
    artifact = {
        "completion_proven": False,
        "pair": "PersonalAnchor x reference_SpatialCoordination_CSP",
        "protocol": "Frozen source-user OOF-temperature F0/CSP providers; saved exact target calibration anchors and splits; fixed equal-provider four-arm replay",
        "analysis_source_sha256": sha(Path(__file__)), "phases": phases,
        "pooled_interactions": pooled,
        "boundary": "The four matched arms test current PersonalAnchor by reference CSP probability composition on native EPN users. They do not recover historical Spatial Coordination/RLCS, prove causal physiology, or establish own-device performance.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    modes = parser.add_subparsers(dest="mode", required=True)
    run_parser = modes.add_parser("evaluate")
    run_parser.add_argument("archive", type=Path)
    run_parser.add_argument("source", type=Path)
    run_parser.add_argument("output", type=Path)
    run_parser.add_argument("phase", choices=tuple(PHASE_USERS))
    summary_parser = modes.add_parser("summarize")
    summary_parser.add_argument("validation", type=Path)
    summary_parser.add_argument("final", type=Path)
    summary_parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.mode == "evaluate":
        evaluate(args.archive, args.source, args.output, args.phase)
    else:
        summarize(args.validation, args.final, args.output)
