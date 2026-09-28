"""Append verified public DS2 v9 calibration scores to Feature Bank source delivery."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "benchmarks/discovery/public_ds2_force_v9"
SOURCE = ROOT / "feature_bank/results/calibration_curve.csv"
INCREMENT_SOURCE = ROOT / "feature_bank/results/conditional_incremental.csv"
COMPLEMENT_SOURCE = ROOT / "feature_bank/results/error_complementarity.csv"
DISAGREEMENT_RECOVERY = ROOT / "feature_bank/results/prediction_disagreement_recovery.json"
RUN_ID = "feature_bank_public_ds2_v9_personal_force_20260928"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def append_verified(path: Path, rows: list[dict], expected_count: int) -> int:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or ())
        existing = [row for row in reader if row["run_id"] == RUN_ID]
    if len(rows) != expected_count or any(set(row) != set(fields) for row in rows):
        raise ValueError(f"Unexpected {path.name} schema or row count")
    if existing:
        if existing != [{key: str(value) for key, value in row.items()} for row in rows]:
            raise ValueError(f"Existing {path.name} DS2 delivery differs from verified results")
        return len(rows)
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=fields, lineterminator="\r\n")
    writer.writerows(rows)
    with path.open("ab") as handle:
        handle.write(text.getvalue().encode("utf-8"))
    return len(rows)


def export() -> int:
    study = json.loads((STUDY / "CALIBRATION_RESULTS.json").read_text(encoding="utf-8"))
    audit = json.loads((STUDY / "CALIBRATION_VERIFICATION.json").read_text(encoding="utf-8"))
    if audit["status"] != "pass" or audit["score_cells_recomputed"] != 336:
        raise ValueError("DS2 calibration verification is incomplete")
    for name, expected in audit["input_sha256"].items():
        if sha(STUDY / name) != expected:
            raise ValueError(f"DS2 calibration source changed: {name}")
    with SOURCE.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or ())
        existing = [row for row in reader if row["run_id"] == RUN_ID]
    rows = []
    for mode in ("unseen_high", "product_all"):
        for arm in ("F0", "F1", "F0_plus_F1"):
            for split, people in (("validation", range(13, 17)),
                                  ("final_descriptive", range(17, 21))):
                for shot in (0, 1, 2, 5):
                    grouped = study["scores"][mode][arm][split][str(shot)]
                    for subject in ("ALL", *(str(person) for person in people)):
                        score = grouped["all"] if subject == "ALL" else grouped["by_subject"][subject]
                        row = dict.fromkeys(fields, "")
                        row.update({
                            "run_id": RUN_ID, "phase": split,
                            "dataset": "public_ds2_v9_3ch_1500hz",
                            "subject": subject, "session/domain": "held_out_subject",
                            "condition": "unseen_high_force2" if mode == "unseen_high" else "product_all_force012",
                            "shots_per_class": shot, "feature_bank": arm,
                            "mode": mode, "method": "source_model_plus_fixed_personal_prototype_blend",
                            "evaluation_trials": score["trials"], "macro_f1": score["macro_f1"],
                            "accuracy": score["accuracy"], "log_loss": score["log_loss"],
                            "brier": score["brier"], "calibration_trials": shot * 4,
                            "supported": True, "aggregation": "whole_native_trial_mean",
                            "protocol": "benchmarks/discovery/public_ds2_force_v9/CALIBRATION_PROTOCOL.json",
                            "calibration_force": "low_average_only" if mode == "unseen_high" else "mixed_low_average_high",
                            "target_force_calibration": mode == "product_all",
                            "model": "source_balanced_logistic_plus_source_scaled_target_prototypes",
                        })
                        rows.append(row)
    return append_verified(SOURCE, rows, 240)


def export_incremental() -> int:
    paired = json.loads((STUDY / "CALIBRATION_PAIRED_RESULTS.json").read_text(encoding="utf-8"))
    if paired["status"] != "post_hoc_descriptive_paired_calibration_analysis" or paired["increment_rows"] != 200:
        raise ValueError("DS2 paired analysis is incomplete")
    for name, digest in paired["input_sha256"].items():
        source = (STUDY / "CALIBRATION_VERIFICATION.json" if name == "calibration_verification"
                  else STUDY / "CALIBRATION_TRIAL_PREDICTIONS.csv" if name == "calibration_predictions"
                  else ROOT / "benchmarks/discovery/scripts/public_ds2_force_v9_calibrated_pairs.py")
        if sha(source) != digest:
            raise ValueError(f"DS2 paired analysis input changed: {name}")
    input_path = STUDY / "CALIBRATION_PAIRED_INCREMENT.csv"
    if sha(input_path) != paired["increment_sha256"]:
        raise ValueError("DS2 paired increment source changed")
    with INCREMENT_SOURCE.open(encoding="utf-8", newline="") as handle:
        fields = list(csv.DictReader(handle).fieldnames or ())
    with input_path.open(encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    rows = []
    for item in source_rows:
        mode = item["training_mode"]
        shots = int(item["shots_per_gesture"])
        row = dict.fromkeys(fields, "")
        row.update({"run_id": RUN_ID, "dataset": "public_ds2_v9_3ch_1500hz",
                    "phase": item["split"], "subject": item["subject_folder"],
                    "session/domain": "held_out_subject",
                    "condition": f"{mode}:{item['condition']}",
                    "calibration_budget": shots, "shots_per_class": shots,
                    "calibration_trials": shots * 4,
                    "evaluation_unit": "whole_native_trial_mean",
                    "evaluation_trials": item["evaluation_trials"],
                    "core_definition_mode": "fixed_source_F0_with_matched_personal_prototypes",
                    "scope": "post_hoc_descriptive_frozen_predictions",
                    "core": "F0", "core_bank": "F0", "added_family": "F1",
                    "delta_log_loss": item["delta_log_loss"],
                    "delta_logloss": item["delta_log_loss"],
                    "delta_macro_f1": item["delta_macro_f1"],
                    "delta_brier": item["delta_brier"],
                    "method": "fixed_feature_concatenation_same_budget",
                    "aggregation": "whole_native_trial_mean",
                    "protocol": "benchmarks/discovery/public_ds2_force_v9/CALIBRATION_PROTOCOL.json",
                    "calibration_force": "low_average_only" if mode == "unseen_high" else "mixed_low_average_high",
                    "target_force_calibration": mode == "product_all"})
        rows.append(row)
    return append_verified(INCREMENT_SOURCE, rows, 200)


def export_complementarity() -> int:
    paired = json.loads((STUDY / "CALIBRATION_PAIRED_RESULTS.json").read_text(encoding="utf-8"))
    input_path = STUDY / "CALIBRATION_PAIRED_COMPLEMENTARITY.csv"
    if paired["complementarity_rows"] != 400 or sha(input_path) != paired["complementarity_sha256"]:
        raise ValueError("DS2 paired complementarity source changed")
    recovery = json.loads(DISAGREEMENT_RECOVERY.read_text(encoding="utf-8"))
    with COMPLEMENT_SOURCE.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or ())
        existing = [row for row in reader if row["run_id"] == RUN_ID]
    if not existing and sha(COMPLEMENT_SOURCE) != recovery["source_csv_sha256"]:
        raise ValueError("Legacy complementarity source changed before append")
    with input_path.open(encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    rows = []
    for item in source_rows:
        mode = item["training_mode"]
        shots = int(item["shots_per_gesture"])
        row = dict.fromkeys(fields, "")
        row.update({"run_id": RUN_ID, "dataset": "public_ds2_v9_3ch_1500hz",
                    "phase": item["split"], "subject": item["subject_folder"],
                    "session/domain": "held_out_subject",
                    "condition": f"{mode}:{item['condition']}",
                    "calibration_budget": shots, "shots_per_class": shots,
                    "calibration_trials": shots * 4,
                    "evaluation_unit": "whole_native_trial_mean",
                    "evaluation_trials": item["evaluation_trials"],
                    "core_definition_mode": "fixed_source_arms_with_matched_personal_prototypes",
                    "scope": "post_hoc_descriptive_frozen_predictions",
                    "family_a": item["family_a"], "family_b": item["family_b"],
                    "error_correlation": item["error_correlation"],
                    "disagreement_rate": item["disagreement_rate"],
                    "a_correct_b_wrong": item["a_correct_b_wrong"],
                    "a_wrong_b_correct": item["a_wrong_b_correct"],
                    "comparison_type": "paired_native_trial_errors",
                    "method": "fixed_same_budget_arm_comparison",
                    "aggregation": "whole_native_trial_mean",
                    "protocol": "benchmarks/discovery/public_ds2_force_v9/CALIBRATION_PROTOCOL.json",
                    "calibration_force": "low_average_only" if mode == "unseen_high" else "mixed_low_average_high",
                    "target_force_calibration": mode == "product_all"})
        rows.append(row)
    count = append_verified(COMPLEMENT_SOURCE, rows, 400)
    recovery["source_csv_sha256"] = sha(COMPLEMENT_SOURCE)
    recovery["scope"] = ("402 frozen native held-out legacy prediction-disagreement recoveries; "
                         "400 newly appended DS2 v9 paired cells have direct recorded disagreement "
                         "and do not use this recovery. No refit.")
    DISAGREEMENT_RECOVERY.write_bytes((json.dumps(recovery, indent=2) + "\n").encode("utf-8"))
    return count


if __name__ == "__main__":
    print(f"DS2 calibration source rows: {export()}, paired increment rows: {export_incremental()}, "
          f"paired complementarity rows: {export_complementarity()}")
