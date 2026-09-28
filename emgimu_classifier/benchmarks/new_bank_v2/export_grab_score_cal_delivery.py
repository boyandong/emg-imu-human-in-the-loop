"""Idempotently export public new-v2 score-anchor calibration curve."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent.parent / "feature_bank" / "results" / "calibration_curve.csv"
RUN_ID = "feature_bank_new_v2_grab_score_anchor_20260928"


def export() -> dict:
    audit_path = ROOT / "GRAB_SCORE_CAL_AUDIT.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "ok" or audit["rows"] != {
            "GRAB_SCORE_CAL_PREDICTIONS.csv": 2048, "GRAB_SCORE_CAL_CURVE.csv": 288}:
        raise AssertionError("GRAB score calibration audit unavailable")
    if audit["protocol_sha256"] != hashlib.sha256((ROOT / "GRAB_SCORE_CAL_PROTOCOL.json").read_bytes()).hexdigest():
        raise AssertionError("frozen calibration protocol changed")
    curve_path = ROOT / "GRAB_SCORE_CAL_CURVE.csv"
    with curve_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 288:
        raise AssertionError("calibration curve row count changed")
    with SOURCE.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        if not fields:
            raise AssertionError("canonical calibration schema unavailable")
        retained = [row for row in reader if row["run_id"] != RUN_ID]
    added = []
    for row in rows:
        budget = int(row["shots_per_class"])
        phase = row["phase"]
        session = "session_2" if phase == "validation" else "session_3"
        values = {"run_id": RUN_ID, "phase": phase, "dataset": "grabmyo_forearm8",
                  "subject": row["subject"], "condition": session, "session/domain": session,
                  "shots_per_class": str(budget), "calibration_trials": str(4 * budget),
                  "feature_bank": row["arm"].replace("F0", "F0v2", 1),
                  "mode": "population_source_only" if budget == 0 else "same_day_score_space_anchor",
                  "evaluation_trials": row["trials"], "macro_f1": row["macro_f1"],
                  "accuracy": row["accuracy"], "log_loss": row["log_loss"],
                  "brier": row["brier"], "ece": row["ece"],
                  "per_class_f1_json": row["per_class_f1_json"],
                  "method": "frozen_probability_prototype_fixed_half_mix",
                  "supported": "True", "aggregation": "pooled" if row["subject"] == "ALL" else "subject",
                  "protocol": "fixed_native_repetitions_1_to_N_calibration_6_7_evaluation",
                  "model": "new_v2_source_logistic_plus_score_anchor", "feature_dimension": "4"}
        if not set(values) <= set(fields):
            raise AssertionError("canonical source lacks calibration field")
        added.append({field: str(values.get(field, "")) for field in fields})
    with SOURCE.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(retained + added)
    delivery = {"run_id": RUN_ID, "rows": len(added),
                "source_curve_sha256": hashlib.sha256(curve_path.read_bytes()).hexdigest(),
                "source_prediction_sha256": hashlib.sha256((ROOT / "GRAB_SCORE_CAL_PREDICTIONS.csv").read_bytes()).hexdigest(),
                "source_audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
                "source_table_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                "boundary": audit["boundary"]}
    (ROOT / "GRAB_SCORE_CAL_DELIVERY_AUDIT.json").write_text(
        json.dumps(delivery, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_id": RUN_ID, "rows": len(added)}))
    return delivery


if __name__ == "__main__":
    export()
