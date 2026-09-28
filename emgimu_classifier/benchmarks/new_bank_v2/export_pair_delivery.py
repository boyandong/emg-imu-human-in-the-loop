"""Idempotently export verified new-v2 paired errors to canonical source schema."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent.parent / "feature_bank" / "results" / "error_complementarity.csv"
RUN_ID = "feature_bank_new_v2_matched_errors_20260928"


def export() -> dict:
    audit_path = ROOT / "PAIR_AUDIT.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "ok" or audit["rows"] != {
            "PAIR_COMPLEMENTARITY.csv": 384, "F2A_F3C_INTERACTION.csv": 64}:
        raise AssertionError("paired analysis audit unavailable")
    if audit["protocol_sha256"] != hashlib.sha256((ROOT / "PAIR_PROTOCOL.json").read_bytes()).hexdigest():
        raise AssertionError("paired analysis protocol changed")
    path = ROOT / "PAIR_COMPLEMENTARITY.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        pairs = list(csv.DictReader(stream))
    if len(pairs) != 384:
        raise AssertionError("paired analysis source count changed")
    with SOURCE.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        if not fields:
            raise AssertionError("error-complementarity source schema unavailable")
        retained = [row for row in reader if row["run_id"] != RUN_ID]
    added = []
    for row in pairs:
        values = {"run_id": RUN_ID, "dataset": row["dataset"],
                  "phase": row["phase"], "subject": row["subject"],
                  "condition": row["condition"], "session/domain": row["condition"],
                  "calibration_budget": "0", "shots_per_class": "0",
                  "evaluation_unit": row["evaluation_unit"],
                  "evaluation_trials": row["evaluation_trials"],
                  "core_definition_mode": "new_independent_v2", "scope": row["scope"],
                  "family_a": row["family_a"], "family_b": row["family_b"],
                  "comparison_type": row["comparison_type"],
                  "error_correlation": row["error_correlation"],
                  "both_wrong": row["both_wrong"],
                  "disagreement_fraction": row["prediction_disagreement_rate"],
                  "disagreement_rate": row["prediction_disagreement_rate"],
                  "a_correct_b_wrong": row["a_correct_b_wrong"],
                  "a_wrong_b_correct": row["a_wrong_b_correct"],
                  "both_correct": row["both_correct"],
                  "protocol": "frozen_saved_prediction_exact_native_trial_pairing",
                  "method": "source_only", "aggregation": row["scope"]}
        if not set(values) <= set(fields):
            raise AssertionError("canonical source missing paired field")
        added.append({field: str(values.get(field, "")) for field in fields})
    with SOURCE.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(retained + added)
    delivery = {"run_id": RUN_ID, "rows": len(added),
                "source_pair_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "source_interaction_sha256": hashlib.sha256((ROOT / "F2A_F3C_INTERACTION.csv").read_bytes()).hexdigest(),
                "source_audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
                "source_table_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                "boundary": audit["boundary"]}
    (ROOT / "PAIR_DELIVERY_AUDIT.json").write_text(
        json.dumps(delivery, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_id": RUN_ID, "rows": len(added)}))
    return delivery


if __name__ == "__main__":
    export()
