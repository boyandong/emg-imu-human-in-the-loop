"""Idempotently deliver paired new-v1 ROAM quality screen cells."""
from __future__ import annotations

import csv
import json

from benchmarks.new_bank_v2.export_force_v1_extension import read
from benchmarks.new_bank_v2.roam_posture_run import sha256
from benchmarks.new_bank_v2.roam_v1_quality_analysis import ROOT, PREFIX

RESULTS = ROOT.parent.parent / "feature_bank" / "results"
RUN_ID = "feature_bank_v1_roam_synthetic_quality_20260929"


def context(row: dict) -> dict:
    return {"run_id": RUN_ID, "dataset": "roam_emg",
            "phase": row["phase"], "subject": row["subject"],
            "condition": row["condition"], "session/domain": "static_resting",
            "calibration_budget": "0", "shots_per_class": "0",
            "evaluation_unit": "native_label_bout",
            "evaluation_trials": row["evaluation_trials"],
            "core_definition_mode": "new_independent_v2_plus_v1",
            "scope": row["scope"], "aggregation": row["scope"],
            "protocol": "source_only_resting_users_fixed_synthetic_fault_grid",
            "method": "source_only_balanced_logistic"}


def export() -> dict:
    audit_path = ROOT / f"{PREFIX}_ANALYSIS_AUDIT.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    expected = {"FAMILY": 840, "CONDITIONAL": 672, "ERROR": 1680}
    if audit["status"] != "ok" or audit["rows"] != expected:
        raise AssertionError("ROAM new-v1 quality analysis unavailable")
    for name in expected:
        if sha256(ROOT / f"{PREFIX}_{name}.csv") != audit["sha256"][name]:
            raise AssertionError(f"quality {name} source changed")
    sources = {
        "feature_family_results.csv": ("FAMILY", lambda row: {
            **context(row), "feature_family": row["feature_family"],
            "model": "source_subject_trial_balanced_logistic_C1",
            "feature_dimension": row["feature_dimension"],
            **{name: row[name] for name in ("macro_f1", "accuracy", "log_loss",
                                            "brier", "ece", "per_class_f1_json")},
            "probability": "saved_heldout", "synthetic": "True",
            "scenario": "fixed_test_only_quality_fault"}),
        "conditional_incremental.csv": ("CONDITIONAL", lambda row: {
            **context(row), "core_bank": row["core_bank"],
            "added_family": row["added_family"],
            "delta_logloss": row["delta_logloss"],
            "delta_macro_f1": row["delta_macro_f1"],
            "delta_brier": row["delta_brier"],
            "comparison": "same_native_bout_same_fault_F0v2_vs_addition"}),
        "error_complementarity.csv": ("ERROR", lambda row: {
            **context(row), "family_a": row["family_a"],
            "family_b": row["family_b"],
            "comparison_type": "matched_source_frozen_arms_synthetic_fault",
            "error_correlation": row["error_correlation"],
            "disagreement_rate": row["disagreement_rate"],
            "a_correct_b_wrong": row["a_correct_b_wrong"],
            "a_wrong_b_correct": row["a_wrong_b_correct"],
            "both_wrong": row["both_wrong"],
            "both_correct": row["both_correct"]}),
    }
    counts, hashes = {}, {}
    for name, (source_name, transform) in sources.items():
        _, incoming = read(ROOT / f"{PREFIX}_{source_name}.csv")
        path = RESULTS / name
        fields, current = read(path)
        if not fields or len(incoming) != expected[source_name]:
            raise AssertionError("quality canonical source coverage changed")
        retained = [row for row in current if row["run_id"] != RUN_ID]
        added = []
        for row in incoming:
            values = transform(row)
            if not set(values) <= set(fields):
                raise AssertionError(f"canonical {name} missing a field")
            added.append({field: str(values.get(field, "")) for field in fields})
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(retained + added)
        counts[name] = len(added)
        hashes[name] = sha256(path)
    result = {"status": "ok", "run_id": RUN_ID, "rows": counts,
              "source_analysis_sha256": sha256(audit_path),
              "source_table_sha256": audit["sha256"],
              "output_sha256": hashes,
              "boundary": audit["boundary"]}
    (ROOT / f"{PREFIX}_DELIVERY_AUDIT.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_id": RUN_ID, "rows": counts}), flush=True)
    return result


if __name__ == "__main__":
    export()
