"""Idempotently export matched ring/spectral candidate LOFO cells."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from benchmarks.new_bank_v2.roam_posture_run import sha256

ROOT = Path(__file__).resolve().parent
TARGET = ROOT.parent.parent / "feature_bank" / "results" / "ablation_full_bank.csv"
RUN_ID = "feature_bank_ring_freq_lofo_20260929"
FULL = "F0v2+ring_lag+frequency_direction"
PARENTS = {"roam_posture": "ROAM_V1_EXTENSION", "grab_user": "GRAB_V1_EXTENSION",
           "grab_day": "GRAB_DAY_V1_EXTENSION"}


def export() -> dict:
    source = ROOT / "RING_FREQ_LOFO_CELLS.csv"
    verification = ROOT / "RING_FREQ_LOFO_VERIFICATION.json"
    audit = json.loads(verification.read_text(encoding="utf-8"))
    if audit["status"] != "ok" or audit["score_cells"] != 176 or audit["cell_sha256"] != sha256(source):
        raise AssertionError("LOFO independent verification unavailable")
    dimensions = {study: json.loads((ROOT / f"{prefix}_RESULTS.json").read_text(
        encoding="utf-8"))["feature_dimensions"] for study, prefix in PARENTS.items()}
    with source.open(newline="", encoding="utf-8") as stream:
        incoming = list(csv.DictReader(stream))
    with TARGET.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        if not fields:
            raise AssertionError("canonical ablation source table unavailable")
        retained = [row for row in reader if row["run_id"] != RUN_ID]
    added = []
    for row in incoming:
        study = row["study"]
        values = {"run_id": RUN_ID,
                  "dataset": "roam_emg_static" if study == "roam_posture" else "grabmyo",
                  "phase": row["phase"],
                  "protocol": "new_v2_independent_ring_frequency_candidate_lofo_matched_native_trials",
                  "subject": row["group"] if row["group_type"] == "subject" else "ALL",
                  "condition": row["group"] if row["group_type"] == "posture" else study,
                  "shots_per_class": "0", "calibration_budget": "0", "method": "source_only_logistic",
                  "macro_f1": row["macro_f1"], "accuracy": row["accuracy"],
                  "log_loss": row["log_loss"], "brier": row["brier"],
                  "per_class_f1_json": row["per_class_f1_json"],
                  "feature_bank": FULL, "removed_family": row["removed_family"],
                  "scenario": study, "evaluation_trials": row["n_trials"],
                  "aggregation": row["group_type"],
                  "delta_macro_f1_vs_full": -float(row["full_minus_removed_macro_f1"]),
                  "feature_dimension": sum(dimensions[study][part] for part in row["arm"].split("+")),
                  "model": "native_trial_StandardScaler_LogisticRegression_C1",
                  "feature_family": row["arm"]}
        if not set(values) <= set(fields):
            raise AssertionError("canonical ablation schema lacks export field")
        added.append({field: str(values.get(field, "")) for field in fields})
    if len(added) != 176:
        raise AssertionError("LOFO export cell count changed")
    with TARGET.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(retained + added)
    result = {"status": "ok", "run_id": RUN_ID, "rows": len(added),
              "source_sha256": sha256(source), "verification_sha256": sha256(verification),
              "output_sha256": sha256(TARGET),
              "boundary": "Fixed candidate full bank failed prior validation, final descriptive; not a selected product bank."}
    (ROOT / "RING_FREQ_LOFO_DELIVERY_AUDIT.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_id": RUN_ID, "rows": len(added)}))
    return result


if __name__ == "__main__":
    export()
