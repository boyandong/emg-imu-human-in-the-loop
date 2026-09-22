"""Source-frozen four-arm G5 × reference Temporal interaction on native UniBo days.

Source days1-4 fit temperature-calibration models; day5 fits temperatures;
days1-5 fit evaluation models. Day6 and days7-8 never fit any state.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from emgimu.datasets.unibo_physiology import chronological_fold, load_chronological_raw_windows
from emgimu.feature_bank.families import TemporalFormFamily
from emgimu.feature_bank.force_nested_oof import fit_temperature, temperature_probability
from emgimu.feature_bank.unibo_full_fusion import batch, classifier, weights
from emgimu.feature_bank.unibo_study import _metrics
from emgimu.feature_bank.validated_unibo import ValidatedUniBoFamily


SEED = 20260915
SPECS = {
    "B": ("G0",),
    "B_plus_G5": ("G0", "G5"),
    "B_plus_Temporal": ("G0", "Temporal"),
    "B_plus_G5_plus_Temporal": ("G0", "G5", "Temporal"),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_families(data, sample_weights):
    fitted, features = {}, {}
    for name, family in (("G0", ValidatedUniBoFamily("G0")),
                         ("G5", ValidatedUniBoFamily("G5")),
                         ("Temporal", TemporalFormFamily())):
        if isinstance(family, ValidatedUniBoFamily):
            family.fit(batch(data), data.labels, sample_weights)
        else:
            family.fit(batch(data), data.labels)
        fitted[name] = family
        features[name] = family.transform(batch(data))
    return fitted, features


def concatenate(features: dict[str, np.ndarray], members: tuple[str, ...]) -> np.ndarray:
    return np.concatenate([features[name] for name in members], axis=1)


def fit_source(archive: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    print("[source 1/4] loading native Days 1-5", flush=True)
    source = load_chronological_raw_windows(archive, (1, 2, 3, 4, 5))
    inner, calibration = chronological_fold(source, (1, 2, 3, 4), 5, 6)
    train, _ = chronological_fold(source, (1, 2, 3, 4, 5), 5, 6)
    sets = [set(data.trial_id.tolist()) for data in (inner, calibration, train)]
    if sets[0] & sets[1] or sets[0] | sets[1] != sets[2]:
        raise ValueError("source day/trial partitions are not disjoint and complete")
    print("[source 2/4] fitting G0, G5 and reference Temporal on Days 1-4", flush=True)
    inner_weights, full_weights = weights(inner), weights(train)
    inner_families, inner_features = source_families(inner, inner_weights)
    cal_features = {name: family.transform(batch(calibration)) for name, family in inner_families.items()}
    calibration_predictions, temperatures = {}, {}
    for arm, members in SPECS.items():
        scaler, model = classifier(concatenate(inner_features, members), inner.labels, inner_weights)
        raw = model.predict_proba(scaler.transform(concatenate(cal_features, members)))
        calibration_predictions[arm] = raw
        temperatures[arm] = fit_temperature(raw, calibration.labels)
    print("[source 3/4] fitting four frozen arms on Days 1-5", flush=True)
    full_families, full_features = source_families(train, full_weights)
    states = {arm: (*classifier(concatenate(full_features, members), train.labels, full_weights), members)
              for arm, members in SPECS.items()}
    dimensions = {name: int(value.shape[1]) for name, value in full_features.items()}
    output.mkdir(parents=True)
    (output / "fitted_states.pkl").write_bytes(pickle.dumps((full_families, states)))
    np.savez_compressed(output / "source_calibration_predictions.npz",
                        labels=calibration.labels, trials=calibration.trial_id,
                        subjects=calibration.subject_id, **calibration_predictions)
    (output / "source_split_trial_ids.json").write_text(json.dumps({
        "inner_train": sorted(sets[0]), "probability_calibration": sorted(sets[1]),
        "full_train": sorted(sets[2])}, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "source_days": [1, 2, 3, 4, 5], "inner_train_days": [1, 2, 3, 4],
        "probability_calibration_day": 5, "target_days_opened": False,
        "source_trial_count": len(sets[2]), "sample_rate_hz": 200, "native_emg_channels": 4,
        "source_window_counts": {"inner": len(inner), "calibration": len(calibration), "full": len(train)},
        "specs": SPECS, "dimensions": dimensions, "temperatures": temperatures,
        "model": "StandardScaler + balanced logistic regression C=1, max_iter=1000",
        "temperature_fit": "held-out source Day5 raw probabilities, unweighted windows, bounds 0.25-4",
        "source_weighting": "hierarchical segment weights for source classifiers",
        "candidate_boundary": "validated historical UniBo G0/G5 adapters plus current reference Temporal; not exact historical TemporalShape",
        "source_sha256": {name: sha(output / name) for name in
                          ("fitted_states.pkl", "source_calibration_predictions.npz", "source_split_trial_ids.json")},
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("[source 4/4] immutable source package saved", flush=True)


def interaction(scores: dict[str, dict]) -> dict[str, float]:
    b, g, t, both = (scores[name] for name in SPECS)
    return {
        "S_negative_logloss": -both["log_loss"] + g["log_loss"] + t["log_loss"] - b["log_loss"],
        "S_macro_f1": both["macro_f1"] - g["macro_f1"] - t["macro_f1"] + b["macro_f1"],
        "S_negative_brier": -both["brier"] + g["brier"] + t["brier"] - b["brier"],
    }


def evaluate(archive: Path, package: Path, output: Path, phase: str) -> None:
    if phase not in {"validation", "final"} or output.exists():
        raise ValueError("evaluation requires a valid phase and fresh output")
    manifest_path = package / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["target_days_opened"] or manifest["source_days"] != [1, 2, 3, 4, 5]:
        raise ValueError("source package is not frozen for this split")
    for name, expected in manifest["source_sha256"].items():
        if sha(package / name) != expected:
            raise ValueError(f"source package changed: {name}")
    days = (6,) if phase == "validation" else (7, 8)
    print(f"[{phase} 1/2] loading native target Days {days}", flush=True)
    target = load_chronological_raw_windows(archive, days)
    source_split = json.loads((package / "source_split_trial_ids.json").read_text(encoding="utf-8"))
    if set(source_split["full_train"]) & set(target.trial_id.tolist()):
        raise ValueError("source and evaluation native trials overlap")
    families, states = pickle.loads((package / "fitted_states.pkl").read_bytes())
    before = pickle.dumps((families, states))
    features = {name: family.transform(batch(target)) for name, family in families.items()}
    predictions = {arm: temperature_probability(
        model.predict_proba(scaler.transform(concatenate(features, members))),
        manifest["temperatures"][arm])
        for arm, (scaler, model, members) in states.items()}
    if pickle.dumps((families, states)) != before:
        raise ValueError("evaluation mutated source fitted state")
    w = weights(target)
    if not set(np.unique(target.labels)) <= set(range(predictions["B"].shape[1])):
        raise ValueError("target label outside source class support")
    cells = [("ALL", "ALL", np.ones(len(target), bool))]
    cells += [(str(user), "ALL", target.subject_id == user) for user in sorted(set(target.subject_id))]
    cells += [("ALL", f"day_{day}", target.session_id == f"d{day:02d}") for day in days]
    cells += [("ALL", f"posture_{posture}", target.posture == posture) for posture in sorted(set(target.posture))]
    score_rows, interaction_rows = [], []
    for user, condition, mask in cells:
        if not np.any(mask):
            continue
        scores = {arm: _metrics(target.labels[mask], p[mask], w[mask]) for arm, p in predictions.items()}
        common = {"phase": phase, "subject": user, "condition": condition,
                  "evaluation_windows": int(mask.sum()), "source_days": "1-5",
                  "calibration_budget": 0}
        score_rows += [{**common, "arm": arm, **score} for arm, score in scores.items()]
        interaction_rows.append({**common, "family_a": "validated_G5",
                                 "family_b": "reference_TemporalForm", **interaction(scores)})
    output.mkdir(parents=True)
    save_csv(output / "arm_scores.csv", score_rows)
    save_csv(output / "interaction_results.csv", interaction_rows)
    np.savez_compressed(output / "heldout_predictions.npz", labels=target.labels,
                        subjects=target.subject_id, trials=target.trial_id,
                        days=target.session_id, posture=target.posture, weights=w, **predictions)
    (output / "split_trial_ids.json").write_text(json.dumps({
        "source": source_split["full_train"], "evaluation": sorted(set(target.trial_id.tolist()))},
        indent=2) + "\n", encoding="utf-8")
    result_manifest = {"phase": phase, "target_days": days, "source_package_sha256": sha(manifest_path),
                       "source_package": str(package), "source_state_immutable": True,
                       "target_trials": len(set(target.trial_id.tolist())),
                       "classifier_or_family_fit": False, "target_calibration": False,
                       "output_sha256": {name: sha(output / name) for name in
                                         ("arm_scores.csv", "interaction_results.csv", "heldout_predictions.npz", "split_trial_ids.json")},
                       "boundary": "matched native windows with hierarchical trial/segment weights; correlated windows, source-day temperature calibration only"}
    (output / "run_manifest.json").write_text(json.dumps(result_manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[{phase} 2/2] {len(interaction_rows)} cells saved", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    modes = parser.add_subparsers(dest="mode", required=True)
    source = modes.add_parser("source")
    source.add_argument("archive", type=Path)
    source.add_argument("output", type=Path)
    target = modes.add_parser("evaluate")
    target.add_argument("archive", type=Path)
    target.add_argument("source_package", type=Path)
    target.add_argument("output", type=Path)
    target.add_argument("phase", choices=("validation", "final"))
    args = parser.parse_args()
    if args.mode == "source":
        fit_source(args.archive, args.output)
    else:
        evaluate(args.archive, args.source_package, args.output, args.phase)
