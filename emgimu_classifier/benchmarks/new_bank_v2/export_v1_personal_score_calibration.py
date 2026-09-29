"""Idempotently deliver independent-family 0/1/2/5-shot calibration cells."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from benchmarks.new_bank_v2.v1_personal_score_calibration import PROTOCOL, ROOT, digest

TARGET = ROOT.parent.parent / "feature_bank" / "results" / "calibration_curve.csv"
RUN_ID = "feature_bank_v1_personal_score_anchor_20260929"


def export() -> dict:
    curve_path = ROOT / "V1_PERSONAL_SCORE_CAL_CURVE.csv"
    verification_path = ROOT / "V1_PERSONAL_SCORE_CAL_VERIFICATION.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if (verification["status"] != "ok" or verification["verified_predictions"] != 3200
            or verification["verified_score_cells"] != 480
            or verification["curve_sha256"] != digest(curve_path)):
        raise AssertionError("personal calibration evidence not verified")
    with curve_path.open(newline="", encoding="utf-8") as stream:
        incoming = list(csv.DictReader(stream))
    with TARGET.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        retained = [row for row in reader if row["run_id"] != RUN_ID]
    if not fields or len(incoming) != 480:
        raise AssertionError("personal calibration source schema changed")
    added = []
    for row in incoming:
        budget = int(row["shots_per_class"])
        study, phase = row["study"], row["phase"]
        day = PROTOCOL["studies"][study]["days"][phase]
        values = {"run_id": RUN_ID, "phase": phase, "dataset": "grabmyo_forearm8",
                  "subject": row["subject"], "condition": f"{study}_day{day}",
                  "session/domain": f"day{day}", "shots_per_class": str(budget),
                  "calibration_trials": str(4 * budget), "feature_bank": row["arm"],
                  "mode": "population_source_only" if budget == 0 else "personal_score_anchor",
                  "evaluation_trials": row["trials"], "macro_f1": row["macro_f1"],
                  "accuracy": row["accuracy"], "log_loss": row["log_loss"],
                  "brier": row["brier"], "ece": row["ece"],
                  "per_class_f1_json": row["per_class_f1"],
                  "method": "frozen_probability_prototype_fixed_half_mix",
                  "supported": "True", "aggregation": "pooled" if row["subject"] == "ALL" else "subject",
                  "protocol": "native_reps_1_to_N_calibration_reps_6_7_evaluation",
                  "model": "new_v1_family_source_logistic_plus_score_anchor",
                  "feature_dimension": "4", "classes_present": "4|15|16|17"}
        if not set(values) <= set(fields):
            raise AssertionError("canonical calibration schema missing a field")
        added.append({field: str(values.get(field, "")) for field in fields})
    with TARGET.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(retained + added)
    result = {"status": "ok", "run_id": RUN_ID, "rows": len(added),
              "source_curve_sha256": digest(curve_path),
              "source_predictions_sha256": digest(ROOT / "V1_PERSONAL_SCORE_CAL_PREDICTIONS.csv"),
              "source_verification_sha256": digest(verification_path),
              "output_sha256": digest(TARGET),
              "boundary": PROTOCOL["scope"]}
    (ROOT / "V1_PERSONAL_SCORE_CAL_DELIVERY_AUDIT.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_id": RUN_ID, "rows": len(added)}))
    return result


if __name__ == "__main__":
    export()
