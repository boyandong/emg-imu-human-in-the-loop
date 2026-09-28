"""Export verified v2 full-minus-family cells to the canonical source schema."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent.parent / "feature_bank" / "results" / "ablation_full_bank.csv"
RUN_ID = "feature_bank_new_v2_lofo_20260928"
FULL = "F0v2+F2a+F3c"
ARMS = {FULL: "NONE", "F2a+F3c": "F0v2", "F0v2+F3c": "F2a", "F0v2+F2a": "F3c"}
DATASETS = {"wearing": "libemg_electrode_shift", "force": "libemg_contraction_intensity"}


def _class_f1(rows: list[dict], classes: np.ndarray) -> str:
    y = np.asarray([int(row["label"]) for row in rows])
    p = np.asarray([[float(row[f"p_{c}"]) for c in classes] for row in rows])
    values = f1_score(y, p.argmax(axis=1), labels=classes, average=None, zero_division=0)
    return json.dumps({str(c): float(v) for c, v in zip(classes, values)}, sort_keys=True)


def export() -> dict:
    verification_path = ROOT / "LOFO_VERIFICATION.json"
    verify = json.loads(verification_path.read_text(encoding="utf-8"))
    if verify.get("status") != "ok" or verify.get("prediction_rows") != 5664:
        raise ValueError("independent LOFO verification unavailable")
    result_path = ROOT / "LOFO_RESULTS.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result["protocol_sha256"] != hashlib.sha256((ROOT / "LOFO_PROTOCOL.json").read_bytes()).hexdigest():
        raise ValueError("LOFO protocol hash changed")
    with (ROOT / "LOFO_TRIAL_PREDICTIONS.csv").open(newline="", encoding="utf-8") as stream:
        predictions = list(csv.DictReader(stream))
    if len(predictions) != 5664:
        raise ValueError("LOFO predictions changed")
    dimensions = {dataset: json.loads((ROOT / f"{dataset.upper()}_RESULTS.json").read_text(
        encoding="utf-8"))["feature_dimensions"] for dataset in DATASETS}
    with SOURCE.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        if not fields:
            raise ValueError("ablation source table has no header")
        old = [row for row in reader if row["run_id"] != RUN_ID]
    added = []
    for dataset, phases in result["scores"].items():
        classes = np.arange(5 if dataset == "wearing" else 7)
        label = DATASETS[dataset]
        for phase, arms in phases.items():
            for arm, scores in arms.items():
                groups = [("ALL", "ALL", scores["pooled"], "pooled")]
                groups += [(subject, "ALL", score, "subject")
                           for subject, score in scores["by_subject"].items()]
                groups += [("ALL", condition, score, "condition")
                           for condition, score in scores["by_condition"].items()]
                for subject, condition, score, scope in groups:
                    matched = [row for row in predictions if row["dataset"] == dataset and
                               row["phase"] == phase and row["arm"] == arm and
                               (subject == "ALL" or row["subject"] == subject) and
                               (condition == "ALL" or row["condition"] == condition)]
                    if len(matched) != score["trials"]:
                        raise ValueError("LOFO score and prediction cell differ")
                    full = result["scores"][dataset][phase][FULL]
                    full_score = (full["pooled"] if scope == "pooled" else
                                  full["by_subject"][subject] if scope == "subject" else
                                  full["by_condition"][condition])
                    values = {"run_id": RUN_ID, "dataset": label, "phase": phase,
                              "protocol": "new_v2_fixed_native_trial_candidate_full_leave_one_family_out",
                              "subject": subject,
                              "condition": condition if condition != "ALL" else
                              "four_wearing_domains" if dataset == "wearing" else "eleven_intensity_conditions",
                              "shots_per_class": 0, "calibration_budget": 0,
                              "method": "source_frozen_balanced_logistic",
                              "macro_f1": score["macro_f1"], "accuracy": score["accuracy"],
                              "log_loss": score["log_loss"], "brier": score["brier"],
                              "per_class_f1_json": _class_f1(matched, classes),
                              "feature_bank": FULL, "removed_family": ARMS[arm],
                              "feature_family": arm,
                              "scenario": "electrode_shift" if dataset == "wearing" else "cross_user_intensity",
                              "evaluation_trials": score["trials"], "aggregation": scope,
                              "delta_macro_f1_vs_full": score["macro_f1"] - full_score["macro_f1"],
                              "feature_dimension": sum(dimensions[dataset][name] for name in arm.split("+")),
                              "model": "trial_StandardScaler_balanced_LogisticRegression_C1"}
                    if not set(values) <= set(fields):
                        raise ValueError("ablation source schema lacks export field")
                    added.append({name: str(values.get(name, "")) for name in fields})
    if len(added) != 176:
        raise AssertionError("expected 176 pooled/subject/condition LOFO cells")
    with SOURCE.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(old + added)
    audit = {"run_id": RUN_ID, "rows": len(added),
             "source_results_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
             "source_predictions_sha256": hashlib.sha256((ROOT / "LOFO_TRIAL_PREDICTIONS.csv").read_bytes()).hexdigest(),
             "source_verification_sha256": hashlib.sha256(verification_path.read_bytes()).hexdigest(),
             "source_table_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
             "boundary": "fixed candidate bank; public matched trials; force full arm not parent-selected"}
    (ROOT / "LOFO_DELIVERY_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_id": RUN_ID, "rows": len(added)}))
    return audit


if __name__ == "__main__":
    export()
