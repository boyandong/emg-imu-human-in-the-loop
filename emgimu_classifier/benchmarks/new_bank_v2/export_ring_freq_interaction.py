"""Idempotently add finite independent-family interaction to source results."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from benchmarks.new_bank_v2.roam_posture_run import sha256

ROOT = Path(__file__).resolve().parent
RUN_ID = "feature_bank_ring_freq_interaction_20260929"
TARGET = ROOT.parent.parent / "feature_bank" / "results" / "interaction_results.csv"


def export() -> dict:
    source = ROOT / "RING_FREQ_INTERACTION.csv"
    audit_path = ROOT / "RING_FREQ_INTERACTION_AUDIT.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit["status"] != "ok" or audit["rows"] != 44 or audit["table_sha256"] != sha256(source):
        raise AssertionError("interaction source audit changed")
    with source.open(newline="", encoding="utf-8") as stream:
        incoming = list(csv.DictReader(stream))
    with TARGET.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        if not fields:
            raise AssertionError("interaction source table unavailable")
        retained = [row for row in reader if row["run_id"] != RUN_ID]
    added = []
    for row in incoming:
        study = row["study"]
        values = {"run_id": RUN_ID, "phase": row["phase"],
                  "subject": row["group"] if row["group_type"] == "subject" else "ALL",
                  "condition": row["group"] if row["group_type"] == "posture" else study,
                  "shots_per_class": "0", "calibration_budget": "0",
                  "calibration_trials": "0", "evaluation_trials": row["n_trials"],
                  "base": "F0v2", "family_a": "ring_lag", "family_b": "frequency_direction",
                  "S_negative_logloss": row["S_negative_logloss"],
                  "S_macro_f1": row["S_macro_f1"],
                  "S_negative_brier": row["S_negative_brier"],
                  "dataset": "roam_emg_static" if study == "roam_posture" else "grabmyo",
                  "evaluation": f"{study}:{row['group_type']}:{row['group']}:matched_native_trials"}
        if not set(values) <= set(fields):
            raise AssertionError("source interaction schema changed")
        added.append({field: str(values.get(field, "")) for field in fields})
    with TARGET.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(retained + added)
    result = {"status": "ok", "run_id": RUN_ID, "rows": len(added),
              "source_sha256": sha256(source), "source_audit_sha256": sha256(audit_path),
              "output_sha256": sha256(TARGET),
              "boundary": "Positive synergy is separate from absolute joint-versus-base gain; final-only evidence cannot select."}
    (ROOT / "RING_FREQ_INTERACTION_DELIVERY_AUDIT.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_id": RUN_ID, "rows": len(added)}))
    return result


if __name__ == "__main__":
    export()
