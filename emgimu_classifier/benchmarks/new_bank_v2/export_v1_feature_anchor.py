"""Idempotently add new-v1 feature-anchor cells to canonical calibration."""
from __future__ import annotations

import csv
import json

from benchmarks.new_bank_v2.roam_posture_run import sha256
from benchmarks.new_bank_v2.v1_feature_anchor import PROTOCOL, ROOT

TARGET = ROOT.parent.parent / "feature_bank" / "results" / "calibration_curve.csv"
RUN_ID = "feature_bank_v1_grab_feature_anchor_20260929"


def export() -> dict:
    cells_path = ROOT / "V1_FEATURE_ANCHOR_CURVE.csv"
    verification_path = ROOT / "V1_FEATURE_ANCHOR_VERIFICATION.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if (verification["status"] != "ok" or verification["verified_predictions"] != 2560
            or verification["verified_score_cells"] != 360
            or verification["curve_sha256"] != sha256(cells_path)):
        raise AssertionError("feature-anchor verification unavailable")
    with cells_path.open(newline="", encoding="utf-8") as stream:
        incoming = list(csv.DictReader(stream))
    with TARGET.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        retained = [row for row in reader if row["run_id"] != RUN_ID]
    if not fields or len(incoming) != 360:
        raise AssertionError("canonical calibration source changed")
    added = []
    for row in incoming:
        budget = int(row["shots_per_class"])
        day = PROTOCOL["target_days"][row["phase"]]
        values = {"run_id": RUN_ID, "phase": row["phase"], "dataset": "grabmyo_forearm8",
                  "subject": row["subject"], "condition": f"cross_day{day}",
                  "session/domain": f"day{day}", "shots_per_class": str(budget),
                  "calibration_trials": str(4 * budget), "feature_bank": row["arm"],
                  "mode": "population_source_only" if budget == 0 else "personal_feature_anchor",
                  "evaluation_trials": row["trials"], "macro_f1": row["macro_f1"],
                  "accuracy": row["accuracy"], "log_loss": row["log_loss"],
                  "brier": row["brier"], "ece": row["ece"],
                  "per_class_f1_json": row["per_class_f1"],
                  "method": "source_standardized_feature_prototype_fixed_half_mix",
                  "supported": "True", "aggregation": "pooled" if row["subject"] == "ALL" else "subject",
                  "protocol": "native_reps_1_to_N_calibration_reps_6_7_evaluation",
                  "model": "new_v1_family_source_logistic_plus_feature_anchor",
                  "classes_present": "4|15|16|17"}
        if not set(values) <= set(fields):
            raise AssertionError("canonical schema missing feature-anchor field")
        added.append({field: str(values.get(field, "")) for field in fields})
    with TARGET.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(retained + added)
    audit = {"status": "ok", "run_id": RUN_ID, "rows": len(added),
             "source_curve_sha256": sha256(cells_path),
             "source_verification_sha256": sha256(verification_path),
             "output_sha256": sha256(TARGET), "boundary": PROTOCOL["scope"]}
    (ROOT / "V1_FEATURE_ANCHOR_DELIVERY_AUDIT.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_id": RUN_ID, "rows": len(added)}))
    return audit


if __name__ == "__main__":
    export()
