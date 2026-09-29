"""Idempotently add the independent family screen to required delivery."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from benchmarks.new_bank_v2.roam_posture_run import sha256

ROOT = Path(__file__).resolve().parent
TARGET = ROOT.parent.parent / "feature_bank" / "results" / "feature_family_results.csv"
RUN_ID = "feature_bank_v1_extension_family_screen_20260929"


def export() -> dict:
    source = ROOT / "V1_EXTENSION_FAMILY_SCREEN.csv"
    audit_path = ROOT / "V1_EXTENSION_FAMILY_AUDIT.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit["status"] != "ok" or audit["rows"] != 220 or audit["table_sha256"] != sha256(source):
        raise AssertionError("five-arm source audit unavailable")
    with source.open(newline="", encoding="utf-8") as stream:
        incoming = list(csv.DictReader(stream))
    with TARGET.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        if not fields:
            raise AssertionError("family delivery source schema unavailable")
        retained = [row for row in reader if row["run_id"] != RUN_ID]
    added = []
    for row in incoming:
        values = {"run_id": RUN_ID, "dataset": row["dataset"],
                  "phase": row["phase"], "subject": row["subject"],
                  "condition": row["condition"] if row["condition"] != "ALL" else row["study"],
                  "calibration_budget": "0", "shots_per_class": "0",
                  "evaluation_unit": row["evaluation_unit"],
                  "evaluation_trials": row["evaluation_trials"],
                  "core_definition_mode": "new_independent_v2_plus_v1",
                  "scope": row["scope"], "feature_family": row["feature_family"],
                  "model": "source_native_logistic_C1_balanced" if row["study"] == "roam_posture"
                           else "source_native_logistic_C1",
                  "feature_dimension": row["feature_dimension"],
                  "macro_f1": row["macro_f1"], "accuracy": row["accuracy"],
                  "log_loss": row["log_loss"], "brier": row["brier"],
                  "ece": row["ece"], "per_class_f1_json": row["per_class_f1_json"],
                  "protocol": "three_frozen_native_target_five_arm_family_screens",
                  "method": "source_only", "aggregation": row["scope"],
                  "probability": "saved_heldout", "synthetic": "False",
                  "session/domain": row["study"], "scenario": row["study"]}
        if not set(values) <= set(fields):
            raise AssertionError("family delivery source schema lacks field")
        added.append({field: str(values.get(field, "")) for field in fields})
    if len(added) != 220:
        raise AssertionError("five-arm export count changed")
    with TARGET.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(retained + added)
    result = {"status": "ok", "run_id": RUN_ID, "rows": len(added),
              "screen_sha256": sha256(source), "screen_audit_sha256": sha256(audit_path),
              "output_sha256": sha256(TARGET),
              "boundary": "Three correlated public screens with validation/final separation; descriptive screening, not own-device model selection."}
    (ROOT / "V1_EXTENSION_FAMILY_DELIVERY_AUDIT.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_id": RUN_ID, "rows": len(added)}))
    return result


if __name__ == "__main__":
    export()
