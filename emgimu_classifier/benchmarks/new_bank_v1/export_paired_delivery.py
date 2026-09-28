"""Idempotently export matched independent new-v1 family results."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parent.parent / "feature_bank" / "results"
RUN_ID = "feature_bank_new_v1_matched_pairs_20260928"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def update(name: str, rows: list[dict], convert) -> tuple[int, str]:
    path = RESULTS / name
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        if not fields:
            raise AssertionError(f"canonical source unavailable: {name}")
        retained = [row for row in reader if row["run_id"] != RUN_ID]
    added = []
    for row in rows:
        values = convert(row)
        if not set(values) <= set(fields):
            raise AssertionError(f"canonical source lacks new-v1 field: {name}")
        added.append({field: str(values.get(field, "")) for field in fields})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(retained + added)
    return len(added), sha(path)


def common(row: dict) -> dict:
    return {"run_id": RUN_ID, "dataset": row["dataset"], "phase": row["phase"],
            "subject": row["subject"], "condition": row["condition"],
            "session/domain": row["condition"], "calibration_budget": "0",
            "shots_per_class": "0", "evaluation_unit": row["evaluation_unit"],
            "evaluation_trials": row["evaluation_trials"],
            "core_definition_mode": "new_independent_v1", "scope": row["scope"],
            "method": "source_only", "aggregation": row["scope"],
            "protocol": "frozen_saved_prediction_exact_native_trial_pairing"}


def family(row: dict) -> dict:
    return {**common(row), "feature_family": row["feature_family"],
            "model": "source_trial_balanced_logistic", "macro_f1": row["macro_f1"],
            "accuracy": row["accuracy"], "log_loss": row["log_loss"],
            "brier": row["brier"], "ece": row["ece"],
            "per_class_f1_json": row["per_class_f1_json"],
            "probability": "saved_heldout", "synthetic": "False"}


def conditional(row: dict) -> dict:
    return {**common(row), "core": row["core_bank"], "core_bank": row["core_bank"],
            "added_family": row["added_family"], "comparison": row["increment_bank"],
            "delta_log_loss": row["delta_log_loss"], "delta_logloss": row["delta_log_loss"],
            "delta_brier": row["delta_brier"], "delta_macro_f1": row["delta_macro_f1"],
            "core_per_class_f1_json": row["core_per_class_f1_json"],
            "increment_per_class_f1_json": row["increment_per_class_f1_json"]}


def errors(row: dict) -> dict:
    return {**common(row), "family_a": row["family_a"], "family_b": row["family_b"],
            "comparison_type": row["comparison_type"],
            "error_correlation": row["error_correlation"],
            "both_wrong": row["both_wrong"], "both_correct": row["both_correct"],
            "a_correct_b_wrong": row["a_correct_b_wrong"],
            "a_wrong_b_correct": row["a_wrong_b_correct"],
            "disagreement_fraction": row["prediction_disagreement_rate"],
            "disagreement_rate": row["prediction_disagreement_rate"]}


def export() -> dict:
    audit_path = ROOT / "PAIRED_AUDIT.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    expected = {"PAIRED_SCREEN.csv": 176, "PAIRED_CONDITIONAL.csv": 88,
                "PAIRED_COMPLEMENTARITY.csv": 44, "PAIRED_INTERACTION.csv": 44}
    if audit.get("status") != "ok" or audit["rows"] != expected or audit["protocol_sha256"] != sha(ROOT / "PAIRED_PROTOCOL.json"):
        raise AssertionError("new-v1 paired analysis audit unavailable")
    mappings = (("PAIRED_SCREEN.csv", "feature_family_results.csv", family),
                ("PAIRED_CONDITIONAL.csv", "conditional_incremental.csv", conditional),
                ("PAIRED_COMPLEMENTARITY.csv", "error_complementarity.csv", errors))
    exported = {}
    for input_name, output_name, convert in mappings:
        with (ROOT / input_name).open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != expected[input_name]:
            raise AssertionError(f"source row count changed: {input_name}")
        count, digest = update(output_name, rows, convert)
        exported[output_name] = {"rows": count, "sha256": digest,
                                 "source_sha256": sha(ROOT / input_name)}
    delivery = {"run_id": RUN_ID, "exports": exported,
                "source_interaction_sha256": sha(ROOT / "PAIRED_INTERACTION.csv"),
                "source_audit_sha256": sha(audit_path), "boundary": audit["boundary"]}
    (ROOT / "PAIRED_DELIVERY_AUDIT.json").write_text(
        json.dumps(delivery, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_id": RUN_ID,
                      "rows": {name: value["rows"] for name, value in exported.items()}}))
    return delivery


if __name__ == "__main__":
    export()
