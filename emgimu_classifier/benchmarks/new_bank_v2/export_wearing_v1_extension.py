"""Idempotently deliver wearing new-v1 family, increment and error cells."""
from __future__ import annotations

import csv
import json

from benchmarks.new_bank_v2.export_force_v1_extension import read
from benchmarks.new_bank_v2.roam_posture_run import sha256
from benchmarks.new_bank_v2.wearing_v1_extension_analysis import ROOT, PREFIX

RESULTS = ROOT.parent.parent / "feature_bank" / "results"
RUN_ID = "feature_bank_v1_wearing_extension_20260929"


def context(row: dict) -> dict:
    return {"run_id": RUN_ID, "dataset": "libemg_electrode_shift",
            "phase": row["phase"], "subject": row["subject"],
            "condition": row["condition"], "session/domain": row["condition"]
            if row["scope"] == "domain" else "same_user_wearing",
            "calibration_budget": "0", "shots_per_class": "0",
            "evaluation_unit": "whole_native_trial_mean",
            "evaluation_trials": row["evaluation_trials"],
            "core_definition_mode": "new_independent_v2_plus_v1",
            "scope": row["scope"], "aggregation": row["scope"],
            "protocol": "same_user_training_to_four_wearing_domains",
            "method": "source_only_balanced_logistic"}


def export() -> dict:
    audit_path = ROOT / f"{PREFIX}_ANALYSIS_AUDIT.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit["status"] != "ok" or audit["rows"] != {
            "FAMILY": 80, "CONDITIONAL": 64, "ERROR": 160}:
        raise AssertionError("wearing extension analysis unavailable")
    for name in audit["rows"]:
        if sha256(ROOT / f"{PREFIX}_{name}.csv") != audit["sha256"][name]:
            raise AssertionError(f"wearing {name} source changed")
    sources = {
        "feature_family_results.csv": ("FAMILY", lambda row: {
            **context(row), "feature_family": row["feature_family"],
            "model": "personal_source_trial_balanced_logistic_C1",
            "feature_dimension": row["feature_dimension"],
            **{name: row[name] for name in ("macro_f1", "accuracy", "log_loss",
                                            "brier", "ece", "per_class_f1_json")},
            "probability": "saved_heldout", "synthetic": "False", "scenario": "wearing_domain"}),
        "conditional_incremental.csv": ("CONDITIONAL", lambda row: {
            **context(row), "core_bank": row["core_bank"],
            "added_family": row["added_family"],
            "delta_logloss": row["delta_logloss"],
            "delta_macro_f1": row["delta_macro_f1"],
            "delta_brier": row["delta_brier"],
            "comparison": "same_native_trial_F0v2_vs_addition"}),
        "error_complementarity.csv": ("ERROR", lambda row: {
            **context(row), "family_a": row["family_a"],
            "family_b": row["family_b"],
            "comparison_type": "matched_personal_source_frozen_arms",
            "error_correlation": row["error_correlation"],
            "disagreement_rate": row["disagreement_rate"],
            "a_correct_b_wrong": row["a_correct_b_wrong"],
            "a_wrong_b_correct": row["a_wrong_b_correct"],
            "both_wrong": row["both_wrong"],
            "both_correct": row["both_correct"]}),
    }
    counts, output_hashes = {}, {}
    for name, (source_name, transform) in sources.items():
        _, incoming = read(ROOT / f"{PREFIX}_{source_name}.csv")
        path = RESULTS / name
        fields, current = read(path)
        if not fields or len(incoming) != audit["rows"][source_name]:
            raise AssertionError("wearing canonical source coverage changed")
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
        output_hashes[name] = sha256(path)
    result = {"status": "ok", "run_id": RUN_ID, "rows": counts,
              "source_analysis_sha256": sha256(audit_path),
              "source_table_sha256": audit["sha256"],
              "output_sha256": output_hashes,
              "boundary": audit["boundary"]}
    (ROOT / f"{PREFIX}_DELIVERY_AUDIT.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_id": RUN_ID, "rows": counts}), flush=True)
    return result


if __name__ == "__main__":
    export()
