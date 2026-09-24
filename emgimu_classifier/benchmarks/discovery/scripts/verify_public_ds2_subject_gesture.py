"""Read back public DS2 held-out predictions and recompute every score."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, recall_score


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(actual: float, expected: float) -> bool:
    return bool(np.isclose(actual, expected, atol=1e-10, rtol=1e-10))


def verify(directory: Path, discovery: Path, runner: Path) -> dict:
    result_path = directory / "RESULTS.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result["status"] != "public_v8_gesture_only_subject_held_out_exploratory":
        raise ValueError("Study scope changed")
    if result["source_sha256"]["runner"] != sha256(runner):
        raise ValueError("Training/evaluation runner changed")
    classifier_root = runner.resolve().parents[3]
    if (result["source_sha256"]["feature_families"] != sha256(
            classifier_root / "src/emgimu/feature_bank/families.py") or
            result["source_sha256"]["feature_batch"] != sha256(
                classifier_root / "src/emgimu/feature_bank/core.py")):
        raise ValueError("Feature implementation changed")
    sources = {
        "tdms_join_csv": discovery / "DS2_TDMS_RAW_EXACT_JOIN.csv",
        "window_join_csv": discovery / "DS2_MAT_TRIAL_WINDOW_JOIN.csv",
    }
    if any(result["source_sha256"][key] != sha256(path)
           for key, path in sources.items()):
        raise ValueError("Input join index changed")
    artifacts = {
        "trial_predictions": "trial_predictions_sha256",
        "conditional_incremental_csv": "conditional_incremental_sha256",
        "error_complementarity_csv": "error_complementarity_sha256",
    }
    if any(result[hash_key] != sha256(directory / result[name])
           for name, hash_key in artifacts.items()):
        raise ValueError("Study CSV hash changed")
    tdms_rows = list(csv.DictReader(sources["tdms_join_csv"].open(encoding="utf-8", newline="")))
    window_rows = list(csv.DictReader(sources["window_join_csv"].open(encoding="utf-8", newline="")))
    expected = {int(row["raw_trial_index_zero_based"]):
                (int(row["subject_folder"]), int(row["gesture_code_if_uniform"]))
                for row in tdms_rows if row["gesture_code_if_uniform"] != "N/A"}
    if len(expected) != 2832 or result["eligible_trials"] != 2832 or \
            result["excluded_unmatched_raw_trials"] != 30 or \
            result["excluded_mixed_gesture_label_trials"] != [209]:
        raise ValueError("Eligible-trial accounting changed")
    if any(str(label) != window_rows[trial]["gesture_label_if_uniform"]
           for trial, (_, label) in expected.items()):
        raise ValueError("Gesture code differs from published window label")
    split_subjects = {split: set(ids) for split, ids in result["split_subject_folders"].items()}
    if split_subjects != {"train": set(range(1, 13)),
                          "validation": set(range(13, 17)),
                          "final": set(range(17, 21))}:
        raise ValueError("Subject partitions changed")
    arms = set(result["arms"])
    if arms != {"F0", "F0_plus_F1_scale_pattern", "F0_plus_F2a_trace_covariance",
                "F0_plus_F2c_SPD", "F0_plus_F4_spectral"}:
        raise ValueError("Experimental arms changed")
    rows = list(csv.DictReader((directory / result["trial_predictions"]).open(
        encoding="utf-8", newline="")))
    grouped = {}
    for row in rows:
        split, arm = row["split"], row["arm"]
        trial, subject, label = (int(row[key]) for key in (
            "raw_trial_index_zero_based", "subject_folder", "gesture_code"))
        if split not in ("validation", "final") or arm not in arms or \
                expected.get(trial) != (subject, label) or subject not in split_subjects[split]:
            raise ValueError("Prediction row violates source join or subject split")
        probability = np.array([float(row[f"p{code}"]) for code in range(5)])
        if not np.isfinite(probability).all() or (probability < 0).any() or \
                (probability > 1).any() or not close(float(probability.sum()), 1.0):
            raise ValueError("Invalid trial probability")
        grouped.setdefault((split, arm), {})[trial] = (subject, label, probability)
    if len(rows) != 5725 or len(grouped) != 10:
        raise ValueError("Trial prediction coverage changed")
    for split in ("validation", "final"):
        wanted = {trial for trial, (subject, _) in expected.items()
                  if subject in split_subjects[split]}
        for arm in arms:
            group = grouped[(split, arm)]
            if set(group) != wanted or len(group) != result["scores"][split][arm]["trials"]:
                raise ValueError("Held-out trial coverage changed")
            trials = sorted(group)
            truth = np.array([group[i][1] for i in trials])
            probability = np.stack([group[i][2] for i in trials])
            prediction = np.argmax(probability, axis=1)
            reference = result["scores"][split][arm]
            values = {"accuracy": accuracy_score(truth, prediction),
                      "macro_f1": f1_score(truth, prediction, labels=np.arange(5),
                                            average="macro", zero_division=0),
                      "log_loss": log_loss(truth, probability, labels=np.arange(5)),
                      "brier": np.mean(np.sum((probability - np.eye(5)[truth]) ** 2,
                                              axis=1))}
            if any(not close(float(value), reference[name]) for name, value in values.items()):
                raise ValueError(f"Saved {split}/{arm} score does not replay")
            if (reference["confusion"] != confusion_matrix(
                    truth, prediction, labels=np.arange(5)).tolist() or
                    any(not close(reference["recall_by_code"][str(code)], float(
                        recall_score(truth, prediction, labels=[code],
                                     average="macro", zero_division=0)))
                        for code in range(5))):
                raise ValueError("Saved class-level score does not replay")
            for subject in split_subjects[split]:
                selected = np.array([group[i][0] == subject for i in trials])
                sub_truth, sub_probability = truth[selected], probability[selected]
                sub_prediction = np.argmax(sub_probability, axis=1)
                sub = result["subject_scores"][split][str(subject)][arm]
                sub_brier = float(np.mean(np.sum((
                    sub_probability - np.eye(5)[sub_truth]) ** 2, axis=1)))
                if (len(sub_truth) != sub["trials"] or
                        not close(float(accuracy_score(sub_truth, sub_prediction)),
                                  sub["accuracy"]) or
                        not close(float(f1_score(sub_truth, sub_prediction,
                                                  labels=np.arange(5), average="macro",
                                                  zero_division=0)), sub["macro_f1"]) or
                        not close(float(log_loss(sub_truth, sub_probability,
                                                 labels=np.arange(5))), sub["log_loss"]) or
                        not close(sub_brier, sub["brier"]) or
                        sub["confusion"] != confusion_matrix(
                            sub_truth, sub_prediction, labels=np.arange(5)).tolist() or
                        any(not close(sub["recall_by_code"][str(code)], float(
                            recall_score(sub_truth, sub_prediction, labels=[code],
                                         average="macro", zero_division=0)))
                            for code in range(5))):
                    raise ValueError("Per-subject score does not replay")
        base = grouped[(split, "F0")]
        ids = sorted(wanted)
        base_correct = np.array([np.argmax(base[i][2]) == base[i][1] for i in ids])
        for arm in arms - {"F0"}:
            score = result["scores"][split][arm]
            base_score = result["scores"][split]["F0"]
            delta = result["increments_vs_F0"][split][arm]
            if not all(close(delta[key], value) for key, value in {
                    "delta_log_loss": base_score["log_loss"] - score["log_loss"],
                    "delta_brier": base_score["brier"] - score["brier"],
                    "delta_macro_f1": score["macro_f1"] - base_score["macro_f1"],
            }.items()):
                raise ValueError("Saved family increment changed")
            candidate_correct = np.array([
                np.argmax(grouped[(split, arm)][i][2]) == grouped[(split, arm)][i][1]
                for i in ids])
            comp = result["error_complementarity_vs_F0"][split][arm]
            if (comp["trials"] != len(ids) or
                    comp["both_correct"] != int(np.sum(base_correct & candidate_correct)) or
                    comp["F0_only_correct"] != int(np.sum(base_correct & ~candidate_correct)) or
                    comp["candidate_only_correct"] != int(np.sum(~base_correct & candidate_correct)) or
                    comp["both_wrong"] != int(np.sum(~base_correct & ~candidate_correct)) or
                    not close(comp["disagreement_rate"],
                              float(np.mean(base_correct != candidate_correct)))):
                raise ValueError("Saved error complementarity changed")
    conditional_rows = list(csv.DictReader((directory / result["conditional_incremental_csv"]).open(
        encoding="utf-8", newline="")))
    complementarity_rows = list(csv.DictReader((directory / result["error_complementarity_csv"]).open(
        encoding="utf-8", newline="")))
    wanted_pairs = {(split, arm) for split in ("validation", "final")
                    for arm in arms - {"F0"}}
    for table in (conditional_rows, complementarity_rows):
        if len(table) != 8 or {(row["split"], row["candidate_arm"])
                               for row in table} != wanted_pairs:
            raise ValueError("Incremental/error CSV coverage changed")
    for row in conditional_rows:
        split, arm = row["split"], row["candidate_arm"]
        expected_row = result["increments_vs_F0"][split][arm]
        if int(row["trials"]) != result["scores"][split][arm]["trials"] or any(
                not close(float(row[key]), expected_row[key]) for key in expected_row):
            raise ValueError("Incremental CSV differs from replayed scores")
    for row in complementarity_rows:
        split, arm = row["split"], row["candidate_arm"]
        expected_row = result["error_complementarity_vs_F0"][split][arm]
        base = grouped[(split, "F0")]
        candidate = grouped[(split, arm)]
        ids = sorted(base)
        base_error = np.array([np.argmax(base[i][2]) != base[i][1] for i in ids])
        candidate_error = np.array([np.argmax(candidate[i][2]) != candidate[i][1]
                                    for i in ids])
        correlation = float(np.corrcoef(base_error.astype(float),
                                        candidate_error.astype(float))[0, 1])
        if (not np.isfinite(correlation) or
                not close(correlation, expected_row["error_correlation"]) or
                not close(float(row["error_correlation"]), correlation) or
                not close(float(row["disagreement_rate"]),
                          expected_row["disagreement_rate"]) or
                any(int(row[key]) != expected_row[key] for key in (
                    "trials", "both_correct", "F0_only_correct",
                    "candidate_only_correct", "both_wrong"))):
            raise ValueError("Error-complementarity CSV differs from replayed predictions")
    audit = {"status": "all_held_out_trial_scores_recomputed",
             "result_sha256": sha256(result_path),
             "prediction_rows": len(rows),
             "held_out_trials_per_arm": {split: result["scores"][split]["F0"]["trials"]
                                         for split in ("validation", "final")},
             "arm_count": len(arms),
             "boundary": "Read-back verifies CSV hashes, subject/gesture joins, disjoint held-out trial coverage, aggregate and per-subject scores, increments and paired correctness. It does not independently rerun source-only fitting; the runner and source hashes bind that computation. Historical force labels and old experiment identity remain unavailable."}
    (directory / "VERIFICATION.json").write_text(json.dumps(audit, indent=2) + "\n",
                                                  encoding="utf-8")
    print(json.dumps({k: audit[k] for k in ("status", "prediction_rows", "arm_count")}))
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-dir", required=True, type=Path)
    parser.add_argument("--discovery-dir", required=True, type=Path)
    parser.add_argument("--runner", required=True, type=Path)
    args = parser.parse_args()
    verify(args.study_dir, args.discovery_dir, args.runner)
