"""Verify G5 × reference Temporal study, including the saved Day-6 oracle."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from emgimu.feature_bank.force_nested_oof import temperature_probability


OLD_ARMS = {"B": "validated_G0", "B_plus_G5": "validated_G0_plus_validated_G5",
            "B_plus_Temporal": "validated_G0_plus_reference_F5"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run(root: Path, old_validation: Path, output: Path) -> None:
    source = root / "feature_bank_unibo_g5_temporal_source_20260922"
    source_manifest_path = source / "run_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest["target_days_opened"] or source_manifest["source_days"] != [1, 2, 3, 4, 5]:
        raise ValueError("source phase is not frozen")
    for name, expected in source_manifest["source_sha256"].items():
        if sha(source / name) != expected:
            raise ValueError(f"source artifact changed: {name}")
    split = json.loads((source / "source_split_trial_ids.json").read_text(encoding="utf-8"))
    if set(split["inner_train"]) & set(split["probability_calibration"]):
        raise ValueError("source temperature calibration overlaps its fit trials")
    phases = {}
    for phase, days in (("validation", [6]), ("final", [7, 8])):
        directory = root / f"feature_bank_unibo_g5_temporal_{phase}_20260922"
        manifest_path = directory / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest["phase"] != phase or list(manifest["target_days"]) != days
                or manifest["source_package_sha256"] != sha(source_manifest_path)
                or manifest["classifier_or_family_fit"] or manifest["target_calibration"]):
            raise ValueError("phase protocol or frozen source reference mismatches")
        for name, expected in manifest["output_sha256"].items():
            if sha(directory / name) != expected:
                raise ValueError(f"output artifact changed: {phase}/{name}")
        target_split = json.loads((directory / "split_trial_ids.json").read_text(encoding="utf-8"))
        if set(target_split["source"]) != set(split["full_train"]) or (
            set(target_split["source"]) & set(target_split["evaluation"])):
            raise ValueError("source/evaluation trial leakage")
        scores = rows(directory / "arm_scores.csv")
        interactions = rows(directory / "interaction_results.csv")
        pooled_scores = [row for row in scores if row["subject"] == row["condition"] == "ALL"]
        pooled_interactions = [row for row in interactions if row["subject"] == row["condition"] == "ALL"]
        if len(pooled_scores) != 4 or len(pooled_interactions) != 1:
            raise ValueError("incomplete pooled four-arm comparison")
        user_rows = [row for row in interactions if row["subject"].startswith("u") and row["condition"] == "ALL"]
        if len(user_rows) != 7:
            raise ValueError("expected seven native user cells")
        phases[phase] = {
            "target_days": days,
            "target_trials": manifest["target_trials"],
            "manifest_sha256": sha(manifest_path),
            "output_sha256": manifest["output_sha256"],
            "pooled_arm_scores": [{"arm": row["arm"],
                                   **{key: float(row[key]) for key in
                                      ("macro_f1", "accuracy", "log_loss", "brier", "ece")}}
                                  for row in pooled_scores],
            "pooled_interaction": {key: float(pooled_interactions[0][key]) for key in
                                   ("S_negative_logloss", "S_macro_f1", "S_negative_brier")},
            "positive_user_S_negative_logloss": sum(float(row["S_negative_logloss"]) > 0 for row in user_rows),
            "user_count": len(user_rows),
        }
    reference_path = old_validation / "heldout_predictions.npz"
    new_path = root / "feature_bank_unibo_g5_temporal_validation_20260922" / "heldout_predictions.npz"
    maximum = 0.0
    with np.load(reference_path, allow_pickle=False) as old, np.load(new_path, allow_pickle=False) as new:
        for key in ("labels", "trials", "subjects"):
            np.testing.assert_array_equal(old[key], new[key])
        for arm, old_key in OLD_ARMS.items():
            oracle = temperature_probability(old[old_key], source_manifest["temperatures"][arm])
            np.testing.assert_allclose(oracle, new[arm], rtol=0, atol=0)
            maximum = max(maximum, float(np.max(np.abs(oracle - new[arm]))))
        matched_windows = len(new["labels"])
    summary = {
        "completion_proven": False,
        "source_manifest_sha256": sha(source_manifest_path),
        "source_artifacts_sha256": source_manifest["source_sha256"],
        "source_days": source_manifest["source_days"],
        "source_calibration_day": source_manifest["probability_calibration_day"],
        "source_temperature_fit_disjoint": True,
        "validation_historical_replay": {
            "old_predictions_sha256": sha(reference_path),
            "matched_windows": matched_windows,
            "matched_arms": list(OLD_ARMS),
            "maximum_absolute_probability_error": maximum,
        },
        "phases": phases,
        "boundary": "Validated UniBo G0/G5 with current reference TemporalForm on native but correlated windows, source-day probability calibration, oracle historical comparison on Day6 only; not exact historical TemporalShape or streaming recognition.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("old_validation", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.root, args.old_validation, args.output)
