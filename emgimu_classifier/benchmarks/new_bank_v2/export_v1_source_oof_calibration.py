"""Idempotently export zero-target-shot source OOF temperature comparisons."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from benchmarks.new_bank_v2.roam_posture_run import sha256

ROOT = Path(__file__).resolve().parent
TARGET = ROOT.parent.parent / "feature_bank" / "results" / "calibration_curve.csv"
RUN_ID = "feature_bank_v1_source_oof_temperature_20260929"


def export() -> dict:
    cells = ROOT / "V1_SOURCE_OOF_CAL_CELLS.csv"
    verification = ROOT / "V1_SOURCE_OOF_CAL_VERIFICATION.json"
    audit = json.loads(verification.read_text(encoding="utf-8"))
    if audit["status"] != "ok" or audit["target_score_cells"] != 440 or audit["cell_sha256"] != sha256(cells):
        raise AssertionError("source OOF calibration verification unavailable")
    with cells.open(newline="", encoding="utf-8") as stream:
        incoming = list(csv.DictReader(stream))
    with TARGET.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        if not fields:
            raise AssertionError("calibration source schema unavailable")
        retained = [row for row in reader if row["run_id"] != RUN_ID]
    added = []
    for row in incoming:
        study = row["study"]
        values = {"run_id": RUN_ID, "phase": row["phase"],
                  "dataset": "roam_emg_static" if study == "roam_posture" else "grabmyo",
                  "subject": row["subject"],
                  "condition": row["condition"] if row["condition"] != "ALL" else study,
                  "session/domain": study,
                  "shots_per_class": "0", "calibration_trials": "0",
                  "feature_bank": row["arm"], "mode": "source_only_probability_calibration",
                  "evaluation_trials": row["evaluation_trials"],
                  "macro_f1": row["macro_f1"], "accuracy": row["accuracy"],
                  "log_loss": row["log_loss"], "brier": row["brier"], "ece": row["ece"],
                  "per_class_f1_json": row["per_class_f1_json"],
                  "method": row["method"], "supported": "True",
                  "aggregation": row["scope"],
                  "protocol": "source_subject_held_out_OOF_temperature_no_target_shots",
                  "model": "new_independent_v2_plus_v1_source_logistic",
                  "reliability_temperature": row["temperature"]}
        if not set(values) <= set(fields):
            raise AssertionError("calibration source schema missing field")
        added.append({field: str(values.get(field, "")) for field in fields})
    if len(added) != 440:
        raise AssertionError("calibration target cell count changed")
    with TARGET.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(retained + added)
    result = {"status": "ok", "run_id": RUN_ID, "rows": len(added),
              "source_cells_sha256": sha256(cells),
              "source_verification_sha256": sha256(verification),
              "output_sha256": sha256(TARGET),
              "boundary": "Zero target shots only, source-subject OOF-selected temperature; not personal 1/2/5-shot calibration or device setup time."}
    (ROOT / "V1_SOURCE_OOF_CAL_DELIVERY_AUDIT.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_id": RUN_ID, "rows": len(added)}))
    return result


if __name__ == "__main__":
    export()
