"""Idempotently export verified v2 Stage-1 cells to required family schema."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent.parent / "feature_bank" / "results" / "feature_family_results.csv"
RUN_ID = "feature_bank_new_v2_family_screen_20260928"


def export() -> dict:
    audit_path = ROOT / "FAMILY_SCREEN_AUDIT.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "ok" or audit["rows"] != {
            "FAMILY_SCREEN.csv": 256, "ROBUSTNESS_VECTOR.csv": 24}:
        raise ValueError("frozen family-screen audit unavailable")
    if audit["protocol_sha256"] != hashlib.sha256((ROOT / "FAMILY_SCREEN_PROTOCOL.json").read_bytes()).hexdigest():
        raise ValueError("family-screen protocol hash changed")
    input_path = ROOT / "FAMILY_SCREEN.csv"
    with input_path.open(newline="", encoding="utf-8") as stream:
        summary = list(csv.DictReader(stream))
    if len(summary) != 256:
        raise ValueError("family-screen cell count changed")
    with SOURCE.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        if not fields:
            raise ValueError("family source table has no header")
        retained = [row for row in reader if row["run_id"] != RUN_ID]
    added = []
    for row in summary:
        model = ("source_window_balanced_logistic" if row["axis"] == "Song_same_day" else
                 "source_recording_balanced_logistic" if row["axis"] == "day" else
                 "source_trial_balanced_logistic")
        values = {"run_id": RUN_ID, "dataset": row["dataset"], "phase": row["phase"],
                  "subject": row["subject"], "condition": row["condition"],
                  "session/domain": row["condition"], "calibration_budget": 0,
                  "shots_per_class": 0, "feature_family": row["feature_family"],
                  "evaluation_unit": row["evaluation_unit"],
                  "evaluation_trials": row["evaluation_trials"],
                  "core_definition_mode": "new_independent_v2", "scope": row["scope"],
                  "model": model, "macro_f1": row["macro_f1"],
                  "accuracy": row["accuracy"], "log_loss": row["log_loss"],
                  "brier": row["brier"], "ece": row["ece"],
                  "per_class_f1_json": row["per_class_f1_json"],
                  "protocol": "source_only_frozen_trial_prediction_family_screen",
                  "method": "source_only", "aggregation": row["scope"],
                  "probability": "saved_heldout", "synthetic": "False"}
        if not set(values) <= set(fields):
            raise ValueError("family source schema lacks export field")
        added.append({name: str(values.get(name, "")) for name in fields})
    with SOURCE.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(retained + added)
    export_audit = {"run_id": RUN_ID, "rows": len(added),
                    "source_screen_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                    "source_robustness_vector_sha256": hashlib.sha256((ROOT / "ROBUSTNESS_VECTOR.csv").read_bytes()).hexdigest(),
                    "source_audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
                    "source_table_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                    "boundary": "four heterogeneous datasets; three observed failure axes, four explicit N/A axes"}
    (ROOT / "FAMILY_SCREEN_DELIVERY_AUDIT.json").write_text(
        json.dumps(export_audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_id": RUN_ID, "rows": len(added)}))
    return export_audit


if __name__ == "__main__":
    export()
