"""Reconcile the legacy EPN Anchor exclusion with frozen corrected runs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def verify(legacy, corrected_root, results, output):
    legacy_rows = rows(legacy / "calibration_curve.csv")
    if len(legacy_rows) != 16:
        raise ValueError("Unexpected legacy row count")
    excluded = [row for row in legacy_rows if row["shots_per_class"] in {"1", "2", "5"}]
    if len(excluded) != 12:
        raise ValueError("Unexpected legacy nonzero-shot rows")
    rules = json.loads((results / "scientific_exclusions.json").read_text(encoding="utf-8"))["rules"]
    if not any(rule["run_id"] == legacy.name and rule["source_artifact"] == "calibration_curve.csv"
               and set(rule["excluded_values"]) == {"1", "2", "5"} for rule in rules):
        raise ValueError("Scientific exclusion rule missing")
    runs = {}
    for phase, users in (("validation", [16, 17, 18]), ("final", [19, 20, 21])):
        run = corrected_root / f"feature_bank_epn612_anchor_temperature_{phase}_20260916_v2"
        manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
        replay = json.loads((run / "replay_audit.json").read_text(encoding="utf-8"))
        split = json.loads((run / "split_trial_ids.json").read_text(encoding="utf-8"))
        current_rows = rows(run / "calibration_curve.csv")
        if manifest["phase"] != phase or manifest["target_users"] != users or manifest["calibration_method"] != "anchor":
            raise ValueError("Corrected protocol changed")
        if manifest["anchor_temperature_fit"] != "calibration-only median prototype distance":
            raise ValueError("Corrected Anchor temperature changed")
        if len(current_rows) != 16 or replay["status"] != "ok" or replay["metric_rows_replayed"] != 16:
            raise ValueError("Corrected output replay missing")
        if set(split["target_cases"]) != {f"user{user}_shots{shots}" for user in users for shots in (0, 1, 2, 5)}:
            raise ValueError("Corrected trial cases missing")
        for key, partition in split["target_cases"].items():
            if set(partition["calibration"]) & set(partition["evaluation"]):
                raise ValueError(f"Calibration/evaluation overlap: {key}")
            if set(split["source"]) & (set(partition["calibration"]) | set(partition["evaluation"])):
                raise ValueError(f"Source/target overlap: {key}")
        files = ("calibration_curve.csv", "calibration_trial_ids.csv", "run_manifest.json",
                 "split_trial_ids.json", "heldout_predictions.npz", "fitted_source_state.pkl", "replay_audit.json")
        runs[phase] = {"run_id": run.name, "users": users, "metric_rows_replayed": 16,
                       "files_sha256": {name: digest(run / name) for name in files},
                       "pooled_results": {row["shots_per_class"]: {"macro_f1": float(row["macro_f1"]),
                           "log_loss": float(row["log_loss"])} for row in current_rows if row["subject"] == "ALL"}}
    validation_run = corrected_root / runs["validation"]["run_id"]
    selection_identical = (legacy / "calibration_trial_ids.csv").read_bytes() == (
        validation_run / "calibration_trial_ids.csv").read_bytes()
    if not selection_identical:
        raise ValueError("Validation calibration selections changed")
    audit = {"completion_proven": False, "issue": "Legacy Anchor scale depended on the evaluation batch",
             "legacy_run": legacy.name, "legacy_curve_sha256": digest(legacy / "calibration_curve.csv"),
             "legacy_nonzero_shot_rows_excluded": len(excluded), "legacy_zero_shot_rows_unaffected": 4,
             "validation_calibration_selection_byte_identical": True, "corrected_runs": runs,
             "total_saved_metric_rows_replayed": 32, "scientific_acceptance":
             "Keep legacy metrics for provenance; exclude its nonzero-shot rows from leakage-compliant claims.",
             "limitations": ["Corrected family is F0 plus current reference envelope-ring, not historical RLCS",
                             "Saved probability replay does not independently refit from raw signals",
                             "Population probabilities are raw logistic outputs, not a fully calibrated late-fusion pipeline",
                             "Validation and final users do not validate own-device four-class performance"]}
    output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"legacy_excluded": len(excluded), "corrected_phases": len(runs),
                      "saved_metric_rows_replayed": 32}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("legacy", type=Path)
    parser.add_argument("corrected_root", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    verify(args.legacy, args.corrected_root, args.results, args.output)
