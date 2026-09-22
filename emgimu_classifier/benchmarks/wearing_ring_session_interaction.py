"""Frozen reference Ring x Session Signature four-arm wearing replay.

The identifiable base is F0+CSP.  Reference Ring is the family addition and
Session Signature is the prespecified calibration-only provider reweighting.
All feature/classifier states and source OOF temperatures come from the frozen
wearing package; exact one-shot splits come from the saved session study.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from emgimu.feature_bank.calibration import SessionSignature, late_fusion
from emgimu.feature_bank.electrode_shift_study import _metrics
from emgimu.feature_bank.force_full_fusion import aggregate
from emgimu.feature_bank.force_nested_oof import temperature_probability
from emgimu.feature_bank.wearing_full_fusion import load


BASE = ("F0", "F2b_CSP")
RING = "F3_Ring"
FAMILIES = (*BASE, RING)
ARMS = ("B", "B_plus_Ring", "B_plus_Session", "B_plus_Ring_plus_Session")
PHASE_USERS = {"validation": (15, 16, 17), "final": (18, 19, 20)}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def compositions(probabilities: dict[str, np.ndarray], context: dict[str, float]) -> dict[str, np.ndarray]:
    if set(probabilities) != set(FAMILIES) or set(context) != set(FAMILIES):
        raise ValueError("interaction requires F0, CSP and reference Ring")
    arrays = list(probabilities.values())
    if any(value.shape != arrays[0].shape or value.ndim != 2 or not np.all(np.isfinite(value))
           for value in arrays) or any(not np.isfinite(value) or value <= 0 for value in context.values()):
        raise ValueError("probabilities and context scores must be finite and aligned")

    def fuse(members: tuple[str, ...], session: bool) -> np.ndarray:
        weights = np.asarray([context[name] if session else 1.0 for name in members], dtype=float)
        weights /= weights.sum()
        return late_fusion({name: probabilities[name] for name in members}, members, weights)

    return {
        "B": fuse(BASE, False),
        "B_plus_Ring": fuse(FAMILIES, False),
        "B_plus_Session": fuse(BASE, True),
        "B_plus_Ring_plus_Session": fuse(FAMILIES, True),
    }


def interaction(scores: dict[str, dict]) -> dict[str, float]:
    b, ring, session, both = (scores[name] for name in ARMS)
    return {
        "S_negative_logloss": -both["log_loss"] + ring["log_loss"] + session["log_loss"] - b["log_loss"],
        "S_macro_f1": both["macro_f1"] - ring["macro_f1"] - session["macro_f1"] + b["macro_f1"],
        "S_negative_brier": -both["brier"] + ring["brier"] + session["brier"] - b["brier"],
    }


def validate_inputs(archive: Path, parent: Path, session_run: Path, phase: str) -> tuple[dict, list[dict]]:
    if phase not in PHASE_USERS:
        raise ValueError("phase must be validation or final")
    parent_manifest = json.loads((parent / "run_manifest.json").read_text(encoding="utf-8"))
    session_manifest = json.loads((session_run / "run_manifest.json").read_text(encoding="utf-8"))
    replay = json.loads((session_run / "replay_audit.json").read_text(encoding="utf-8"))
    if (parent_manifest.get("phase") != phase or session_manifest.get("phase") != phase
            or tuple(session_manifest.get("users", ())) != PHASE_USERS[phase]
            or session_manifest.get("parent_run") != parent.name
            or set(FAMILIES) - set(parent_manifest.get("families", ()))
            or replay.get("status") != "ok" or replay.get("maximum_absolute_probability_error") != 0.0
            or not replay.get("long_term_profiles_immutable")):
        raise ValueError("saved wearing/session package is not eligible")
    for name, expected in session_manifest["parent_sha256"].items():
        if sha(parent / name) != expected:
            raise ValueError(f"frozen parent changed: {name}")
    if sha(archive) != session_manifest["raw_archive_sha256"]:
        raise ValueError("wearing raw archive changed")
    splits = json.loads((session_run / "split_trial_ids.json").read_text(encoding="utf-8"))
    selected = [row for row in splits if int(row["shots"]) == 1]
    if len(selected) != 4 * len(PHASE_USERS[phase]):
        raise ValueError("one-shot session split grid is incomplete")
    return parent_manifest, selected


def evaluate(archive: Path, parent: Path, session_run: Path, output: Path, phase: str) -> None:
    if output.exists():
        raise FileExistsError(output)
    manifest, splits = validate_inputs(archive, parent, session_run, phase)
    users = PHASE_USERS[phase]
    parent_states = pickle.loads((parent / "fitted_states.pkl").read_bytes())
    before = pickle.dumps(parent_states)
    score_rows, interaction_rows, saved, output_splits, pooled = [], [], {}, [], {}
    split_by_cell = {(int(row["user"]), row["domain"]): row for row in splits}
    for user_index, user in enumerate(users, 1):
        print(f"[{phase} {user_index}/3] user {user}: frozen F0/CSP/Ring and exact one-shot splits", flush=True)
        source = load(archive, user, ("training",))
        target = load(archive, user, ("trial_1", "trial_2", "trial_3", "trial_4"))
        source_features, target_features, probabilities, signatures = {}, {}, {}, {}
        sy = st = y = trials = None
        for name in FAMILIES:
            family, scaler, model = parent_states[(user, name)]
            a, current_sy, _, current_st = aggregate(family.transform(source.batch), source)
            b, current_y, _, current_trials = aggregate(family.transform(target.batch), target)
            if sy is None:
                sy, st, y, trials = current_sy, current_st, current_y, current_trials
            else:
                for actual, expected in ((current_sy, sy), (current_st, st),
                                         (current_y, y), (current_trials, trials)):
                    np.testing.assert_array_equal(actual, expected)
            source_features[name] = scaler.transform(a)
            target_features[name] = scaler.transform(b)
            probabilities[name] = temperature_probability(
                model.predict_proba(target_features[name]), manifest["temperatures"][f"{user}_{name}"])
            signatures[name] = SessionSignature().fit_long_term(source_features[name], sy)
        domains = np.asarray([trial.split("/")[-2] for trial in trials])
        for domain in sorted(set(domains)):
            split = split_by_cell[(user, domain)]
            cal = np.flatnonzero(np.isin(trials, split["calibration"]))
            ev = np.flatnonzero(np.isin(trials, split["evaluation"]))
            if (len(cal) != 5 or len(ev) != 5 or set(trials[cal]) & set(trials[ev])
                    or set(st) & (set(trials[cal]) | set(trials[ev]))
                    or np.any(domains[cal] != domain) or np.any(domains[ev] != domain)
                    or set(np.unique(y[cal]).tolist()) != set(range(5))):
                raise ValueError(f"invalid one-shot split: {user}/{domain}")
            context = {}
            for name in FAMILIES:
                vector = signatures[name].from_session_calibration(target_features[name][cal], y[cal])
                context[name] = float(np.clip((vector[5:10].mean() + 1) / 2, .05, 1.0))
            arms = compositions({name: probabilities[name][ev] for name in FAMILIES}, context)
            scores = {name: _metrics(y[ev], probability) for name, probability in arms.items()}
            common = {"phase": phase, "subject": user, "condition": domain,
                      "shots_per_class": 1, "calibration_trials": 5,
                      "evaluation_trials": 5, "base": "F0|F2b_CSP_reference",
                      "family_a": "F3_Ring_reference", "family_b": "F8_SessionSignature"}
            score_rows += [{**common, "arm": name, **score} for name, score in scores.items()]
            interaction_rows.append({**common, **interaction(scores)})
            output_splits.append({"user": user, "domain": domain, "shots": 1,
                                  "train": split["train"], "calibration": split["calibration"],
                                  "evaluation": split["evaluation"]})
            saved[f"{user}_{domain}_labels"] = y[ev]
            saved[f"{user}_{domain}_trials"] = trials[ev]
            saved[f"{user}_{domain}_context"] = np.asarray([context[name] for name in FAMILIES])
            for name, probability in arms.items():
                saved[f"{user}_{domain}_{name}"] = probability
                pooled.setdefault(name, []).append((y[ev], probability))
    if pickle.dumps(parent_states) != before:
        raise ValueError("target replay mutated parent fitted states")
    pooled_scores = {name: _metrics(np.concatenate([item[0] for item in values]),
                                    np.concatenate([item[1] for item in values]))
                     for name, values in pooled.items()}
    common = {"phase": phase, "subject": "ALL", "condition": "ALL",
              "shots_per_class": 1, "calibration_trials": 5 * 4 * len(users),
              "evaluation_trials": 5 * 4 * len(users), "base": "F0|F2b_CSP_reference",
              "family_a": "F3_Ring_reference", "family_b": "F8_SessionSignature"}
    score_rows += [{**common, "arm": name, **score} for name, score in pooled_scores.items()]
    interaction_rows.append({**common, **interaction(pooled_scores)})
    output.mkdir(parents=True)
    write_csv(output / "arm_scores.csv", score_rows)
    write_csv(output / "interaction_results.csv", interaction_rows)
    np.savez_compressed(output / "heldout_predictions.npz", **saved)
    (output / "split_trial_ids.json").write_text(json.dumps(output_splits, indent=2) + "\n", encoding="utf-8")
    result_manifest = {
        "phase": phase, "source_users": list(users), "target_users": list(users),
        "source_domain": "before-wearing training", "target_domains": [f"trial_{i}" for i in range(1, 5)],
        "families": list(FAMILIES), "arms": list(ARMS), "shots_per_class": 1,
        "classifier_or_family_fit": False, "source_signature_fit": True,
        "target_fit": False, "target_rule_selection": False, "source_state_immutable": True,
        "session_rule": "uniform provider weights times clipped (mean class cosine+1)/2",
        "raw_archive_sha256": sha(archive),
        "source_sha256": {"parent/fitted_states.pkl": sha(parent / "fitted_states.pkl"),
                          "parent/run_manifest.json": sha(parent / "run_manifest.json"),
                          "session/run_manifest.json": sha(session_run / "run_manifest.json"),
                          "session/split_trial_ids.json": sha(session_run / "split_trial_ids.json"),
                          "session/replay_audit.json": sha(session_run / "replay_audit.json")},
        "output_sha256": {name: sha(output / name) for name in
                          ("arm_scores.csv", "interaction_results.csv", "heldout_predictions.npz", "split_trial_ids.json")},
        "analysis_source_sha256": sha(Path(__file__)),
        "boundary": "Same-user native before/after wearing reference Ring and calibration-only Session Signature probability interaction; Ring is not historical RLCS, domains are not calendar sessions, and results are not own-device evidence.",
    }
    (output / "run_manifest.json").write_text(json.dumps(result_manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[{phase}] saved {len(score_rows)} arm rows and {len(interaction_rows)} interactions", flush=True)


def summarize(validation: Path, final: Path, output: Path) -> None:
    phases, pooled = {}, []
    for phase, root in (("validation", validation), ("final", final)):
        manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
        if manifest["phase"] != phase or manifest["analysis_source_sha256"] != sha(Path(__file__)):
            raise ValueError("phase output is not bound to current frozen analysis")
        for name, expected in manifest["output_sha256"].items():
            if sha(root / name) != expected:
                raise ValueError(f"changed phase output: {phase}/{name}")
        with (root / "interaction_results.csv").open(newline="", encoding="utf-8") as handle:
            row = next(row for row in csv.DictReader(handle) if row["subject"] == "ALL")
        pooled.append({key: (float(value) if key.startswith("S_") else int(value) if key in
                             {"shots_per_class", "calibration_trials", "evaluation_trials"} else value)
                       for key, value in row.items()})
        phases[phase] = {"run_manifest_sha256": sha(root / "run_manifest.json"),
                         "output_sha256": manifest["output_sha256"]}
    artifact = {"completion_proven": False,
                "pair": "reference_Ring x SessionSignature",
                "protocol": "Frozen F0+CSP base, fixed reference Ring addition and calibration-only Session Signature reweighting on exact saved one-shot wearing splits",
                "analysis_source_sha256": sha(Path(__file__)), "phases": phases,
                "pooled_interactions": pooled,
                "boundary": "Reference Ring is not historical RLCS; same-user wearing domains are not longitudinal sessions; fixed probability composition does not prove physiological synergy or own-device performance."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); modes = parser.add_subparsers(dest="mode", required=True)
    evaluation = modes.add_parser("evaluate")
    evaluation.add_argument("archive", type=Path); evaluation.add_argument("parent", type=Path)
    evaluation.add_argument("session_run", type=Path); evaluation.add_argument("output", type=Path)
    evaluation.add_argument("phase", choices=tuple(PHASE_USERS))
    summary = modes.add_parser("summarize")
    summary.add_argument("validation", type=Path); summary.add_argument("final", type=Path)
    summary.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.mode == "evaluate":
        evaluate(args.archive, args.parent, args.session_run, args.output, args.phase)
    else:
        summarize(args.validation, args.final, args.output)
