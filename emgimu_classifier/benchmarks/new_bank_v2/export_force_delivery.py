"""Idempotently export verified new-v2 force paired cells to canonical sources."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parent.parent / "feature_bank" / "results"
RUN_ID = "feature_bank_new_v2_force_paired_20260928"
DATASET = "libemg_contraction_intensity"


def _read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"missing header: {path}")
        return reader.fieldnames, list(reader)


def _context(row: dict[str, str]) -> dict[str, str]:
    scope = row["scope"]
    cell = row["cell"]
    return {"run_id": RUN_ID, "dataset": DATASET, "phase": row["phase"],
            "subject": cell if scope == "subject" else "ALL",
            "condition": cell if scope == "condition" else "ALL",
            "session/domain": cell if scope == "condition" else "cross_user_intensity",
            "calibration_budget": "0", "shots_per_class": "0",
            "evaluation_unit": "whole_native_trial_mean", "evaluation_trials": row["trials"],
            "core_definition_mode": "new_independent_v2", "scope": scope,
            "protocol": "new_v2_source_ramp_cross_user_11_conditions",
            "aggregation": "pooled trials" if scope == "pooled" else f"{scope} cell",
            "method": "source_frozen_balanced_logistic"}


def export() -> dict:
    verify = json.loads((ROOT / "FORCE_VERIFICATION.json").read_text(encoding="utf-8"))
    audit = json.loads((ROOT / "FORCE_PAIRED_AUDIT.json").read_text(encoding="utf-8"))
    if verify.get("status") != "ok" or verify.get("prediction_rows") != 5880:
        raise ValueError("force trial verification is unavailable")
    if audit.get("status") != "ok" or audit.get("rows") != {
            "FORCE_CONDITIONAL.csv": 84, "FORCE_COMPLEMENTARITY.csv": 28,
            "FORCE_INTERACTION.csv": 28}:
        raise ValueError("paired force audit is unavailable")
    source_predictions = ROOT / "FORCE_TRIAL_PREDICTIONS.csv"
    if audit["source_predictions_sha256"] != hashlib.sha256(source_predictions.read_bytes()).hexdigest():
        raise ValueError("paired force audit refers to different predictions")
    _, increments = _read(ROOT / "FORCE_CONDITIONAL.csv")
    _, pairs = _read(ROOT / "FORCE_COMPLEMENTARITY.csv")
    if len(increments) != 84 or len(pairs) != 28:
        raise ValueError("paired force table row count changed")
    specifications = (
        ("conditional_incremental.csv", increments, lambda r: {
            **_context(r), "core_bank": r["core_bank"], "added_family": r["added_family"],
            "delta_logloss": r["delta_logloss_improvement"],
            "delta_macro_f1": r["delta_macro_f1"],
            "delta_brier": r["delta_brier_improvement"],
            "comparison": "same_native_trials_core_vs_added_family"}),
        ("error_complementarity.csv", pairs, lambda r: {
            **_context(r), "family_a": "F0v2+F2a", "family_b": "F0v2+F3c",
            "comparison_type": "matched_incremental_arms",
            "error_correlation": r["error_correlation"],
            "disagreement_rate": r["prediction_disagreement_rate"],
            "a_correct_b_wrong": r["a_correct_b_wrong"],
            "a_wrong_b_correct": r["a_wrong_b_correct"]}),
    )
    rows_added = {}
    for name, source, transform in specifications:
        path = RESULTS / name
        fields, existing = _read(path)
        retained = [row for row in existing if row["run_id"] != RUN_ID]
        added = []
        for row in source:
            values = transform(row)
            if not set(values) <= set(fields):
                raise ValueError(f"unrecognized delivery field in {name}")
            added.append({field: str(values.get(field, "")) for field in fields})
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(retained + added)
        rows_added[name] = len(added)
    delivery_audit = {"run_id": RUN_ID, "rows_added": rows_added,
                      "source_predictions_sha256": audit["source_predictions_sha256"],
                      "source_conditional_sha256": hashlib.sha256((ROOT / "FORCE_CONDITIONAL.csv").read_bytes()).hexdigest(),
                      "source_complementarity_sha256": hashlib.sha256((ROOT / "FORCE_COMPLEMENTARITY.csv").read_bytes()).hexdigest(),
                      "source_tables_sha256": {name: hashlib.sha256((RESULTS / name).read_bytes()).hexdigest()
                                               for name in rows_added},
                      "boundary": "public cross-user intensity; final comparisons descriptive; no target fit"}
    (ROOT / "FORCE_DELIVERY_AUDIT.json").write_text(json.dumps(delivery_audit, indent=2) + "\n",
                                                      encoding="utf-8")
    print(json.dumps({"run_id": RUN_ID, "rows_added": rows_added}))
    return delivery_audit


if __name__ == "__main__":
    export()
