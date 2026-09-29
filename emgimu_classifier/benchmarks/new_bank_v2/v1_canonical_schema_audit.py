"""Check required canonical fields for the versioned seven-axis family delivery."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from benchmarks.new_bank_v2.roam_posture_run import sha256

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "feature_bank" / "results"
OUTPUT = Path(__file__).resolve().parent / "V1_CANONICAL_SCHEMA_AUDIT.json"
RUNS = (
    "feature_bank_v1_extension_family_screen_20260929",
    "feature_bank_v1_force_intensity_extension_20260929",
    "feature_bank_v1_wearing_extension_20260929",
    "feature_bank_v1_manus_speed_external_rest_20260929",
    "feature_bank_v1_roam_synthetic_quality_20260929",
)
FIELDS = {
    "feature_family_results.csv": ("dataset", "subject", "session/domain", "feature_family",
                                   "calibration_budget", "condition", "macro_f1", "accuracy",
                                   "log_loss", "brier", "ece"),
    "conditional_incremental.csv": ("dataset", "subject", "session/domain", "core_bank",
                                    "added_family", "condition", "delta_logloss",
                                    "delta_macro_f1", "delta_brier"),
    "error_complementarity.csv": ("dataset", "subject", "session/domain", "family_a",
                                  "family_b", "error_correlation", "disagreement_rate",
                                  "a_correct_b_wrong", "a_wrong_b_correct"),
}
EXPECTED = {"feature_family_results.csv": 1560,
            "conditional_incremental.csv": 1072,
            "error_complementarity.csv": 2680}


def build() -> dict:
    artifacts = {}
    for name, fields in FIELDS.items():
        path = RESULTS / name
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if not set(fields) <= set(reader.fieldnames or []):
                raise AssertionError(f"missing new-v1 canonical column: {name}")
            rows = [row for row in reader if row["run_id"] in RUNS]
        if len(rows) != EXPECTED[name]:
            raise AssertionError(f"new-v1 canonical run coverage changed: {name}")
        missing = {field: sum(not row[field] for row in rows) for field in fields}
        if any(missing.values()):
            raise AssertionError(f"new-v1 required values missing: {name}: {missing}")
        artifacts[name] = {"rows": len(rows), "required_fields": fields,
                           "missing_field_counts": missing, "source_sha256": sha256(path),
                           "run_ids": sorted({row["run_id"] for row in rows})}
    audit = {"status": "new_v1_required_fields_complete", "artifacts": artifacts,
             "total_checked_rows": sum(item["rows"] for item in artifacts.values()),
             "boundary": "Only the versioned new-v1 seven-axis family/conditional/error delivery is assessed. Older canonical rows retain explicit N/A fields and are not silently filled with inferred values. Calibration and ablation tables have separate source audits."}
    OUTPUT.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "rows": audit["total_checked_rows"]}), flush=True)
    return audit


if __name__ == "__main__":
    build()
