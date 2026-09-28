"""Export new-v2 conditional family increments to the canonical source table."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent.parent / "feature_bank" / "results" / "conditional_incremental.csv"
RUN_ID = "feature_bank_new_v2_conditional_value_20260928"


def export() -> dict:
    audit_path = ROOT / "CONDITIONAL_AUDIT.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "ok" or audit["rows"] != 128:
        raise AssertionError("conditional-value audit unavailable")
    if audit["protocol_sha256"] != hashlib.sha256((ROOT / "CONDITIONAL_PROTOCOL.json").read_bytes()).hexdigest():
        raise AssertionError("conditional-value protocol changed")
    path = ROOT / "CONDITIONAL_VALUE.csv"
    if audit["family_screen_sha256"] != hashlib.sha256((ROOT / "FAMILY_SCREEN.csv").read_bytes()).hexdigest():
        raise AssertionError("family-screen source changed")
    if audit["pair_interaction_sha256"] != hashlib.sha256((ROOT / "F2A_F3C_INTERACTION.csv").read_bytes()).hexdigest():
        raise AssertionError("matched-pair source changed")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 128:
        raise AssertionError("conditional-value row count changed")
    with SOURCE.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        if not fields:
            raise AssertionError("canonical conditional schema unavailable")
        retained = [row for row in reader if row["run_id"] != RUN_ID]
    added = []
    for row in rows:
        values = {"run_id": RUN_ID, "dataset": row["dataset"], "phase": row["phase"],
                  "subject": row["subject"], "condition": row["condition"],
                  "session/domain": row["condition"], "calibration_budget": "0",
                  "shots_per_class": "0", "evaluation_unit": row["evaluation_unit"],
                  "evaluation_trials": row["evaluation_trials"],
                  "core_definition_mode": "new_independent_v2", "scope": row["scope"],
                  "core": row["core_bank"], "core_bank": row["core_bank"],
                  "added_family": row["added_family"],
                  "comparison": row["increment_bank"],
                  "delta_log_loss": row["delta_log_loss"],
                  "delta_logloss": row["delta_log_loss"],
                  "delta_brier": row["delta_brier"],
                  "delta_macro_f1": row["delta_macro_f1"],
                  "core_per_class_f1_json": row["core_per_class_f1_json"],
                  "increment_per_class_f1_json": row["increment_per_class_f1_json"],
                  "protocol": "frozen_saved_prediction_conditional_family_value",
                  "method": "source_only", "aggregation": row["scope"]}
        if not set(values) <= set(fields):
            raise AssertionError("canonical source lacks conditional field")
        added.append({field: str(values.get(field, "")) for field in fields})
    with SOURCE.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(retained + added)
    delivery = {"run_id": RUN_ID, "rows": len(added),
                "source_conditional_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "source_audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
                "source_table_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                "boundary": audit["boundary"]}
    (ROOT / "CONDITIONAL_DELIVERY_AUDIT.json").write_text(
        json.dumps(delivery, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_id": RUN_ID, "rows": len(added)}))
    return delivery


if __name__ == "__main__":
    export()
