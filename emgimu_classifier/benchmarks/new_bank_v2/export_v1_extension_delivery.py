"""Idempotently deliver matched independent-family conditional/error rows."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from benchmarks.new_bank_v2.roam_posture_run import sha256

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parent.parent / "feature_bank" / "results"
RUN_ID = "feature_bank_v1_extension_paired_20260929"


def common(row: dict) -> dict:
    study = row["study"]
    domain = {"roam_posture": "public_200hz_posture",
              "grab_user": "public_2048hz_cross_user_day1",
              "grab_day": "public_2048hz_cross_day"}[study]
    subject = row["group"] if row["group_type"] == "subject" else "ALL"
    condition = row["group"] if row["group_type"] == "posture" else "ALL"
    return {"run_id": RUN_ID, "dataset": "roam_emg_static" if study == "roam_posture" else "grabmyo",
            "phase": row["phase"], "subject": subject, "condition": condition,
            "calibration_budget": "0", "shots_per_class": "0",
            "evaluation_unit": "native_label_bout" if study == "roam_posture" else "native_recording",
            "evaluation_trials": row["n_trials"], "core_definition_mode": "new_independent_v2_plus_v1",
            "scope": row["group_type"], "aggregation": row["group_type"],
            "session/domain": domain, "method": "source_only",
            "protocol": "three_frozen_public_screens_matched_saved_predictions"}


def export() -> dict:
    paired_path = ROOT / "V1_EXTENSION_PAIRED.csv"
    audit_path = ROOT / "V1_EXTENSION_PAIRED_AUDIT.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit["status"] != "ok" or audit["rows"] != 176 or audit["paired_csv_sha256"] != sha256(paired_path):
        raise AssertionError("matched source audit unavailable")
    with paired_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 176:
        raise AssertionError("matched source row count changed")
    exports = {}
    for name in ("conditional_incremental.csv", "error_complementarity.csv"):
        target = RESULTS / name
        with target.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            if not fields:
                raise AssertionError(f"canonical source unavailable: {name}")
            retained = [r for r in reader if r["run_id"] != RUN_ID]
        added = []
        for row in rows:
            base = common(row)
            family_arm = f"F0v2+{row['family']}"
            if name == "conditional_incremental.csv":
                values = {**base, "core": "F0v2", "core_bank": "F0v2",
                          "added_family": row["family"], "comparison": family_arm,
                          "delta_log_loss": row["log_loss_improvement"],
                          "delta_logloss": row["log_loss_improvement"],
                          "delta_brier": row["brier_improvement"],
                          "delta_macro_f1": row["delta_macro_f1"],
                          "core_per_class_f1_json": row["base_per_class_f1_json"],
                          "increment_per_class_f1_json": row["added_per_class_f1_json"]}
            else:
                disagreement_rate = str(int(row["prediction_disagreement"]) / int(row["n_trials"]))
                values = {**base, "family_a": "F0v2", "family_b": family_arm,
                          "comparison_type": "matched_independent_new_v1_addition",
                          "error_correlation": row["error_correlation"],
                          "both_wrong": row["both_wrong"], "both_correct": row["both_right"],
                          "core_wrong_increment_correct": row["base_wrong_added_right"],
                          "core_correct_increment_wrong": row["base_right_added_wrong"],
                          "a_correct_b_wrong": row["base_right_added_wrong"],
                          "a_wrong_b_correct": row["base_wrong_added_right"],
                          "disagreement": row["prediction_disagreement"],
                          "disagreement_fraction": disagreement_rate,
                          "disagreement_rate": disagreement_rate}
            if not set(values) <= set(fields):
                raise AssertionError(f"canonical schema missing fields for {name}")
            added.append({field: str(values.get(field, "")) for field in fields})
        with target.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(retained + added)
        exports[name] = {"rows": len(added), "sha256": sha256(target)}
    result = {"status": "ok", "run_id": RUN_ID, "paired_csv_sha256": sha256(paired_path),
              "paired_audit_sha256": sha256(audit_path), "exports": exports,
              "boundary": "Source tables contain 176 post-hoc matched cells per artifact; no full-bank interaction or own-device claim."}
    (ROOT / "V1_EXTENSION_DELIVERY_AUDIT.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_id": RUN_ID, "rows": {k: v["rows"] for k, v in exports.items()}}))
    return result


if __name__ == "__main__":
    export()
