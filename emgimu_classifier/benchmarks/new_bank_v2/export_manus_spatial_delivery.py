"""Idempotently add reduced new-v2 MANUS spatial evidence to canonical tables."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from benchmarks.new_bank_v1.export_paired_delivery import conditional, errors, family

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parent.parent / "feature_bank" / "results"
RUN_ID = "feature_bank_manus_reduced_spatial_v2_20260928"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export() -> dict:
    audit_path = ROOT / "MANUS_SPATIAL_PAIRED_AUDIT.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (audit["status"] != "ok" or audit["matched_cells"] != 20
            or audit["parent_score_groups_replayed"] != 80
            or audit["analysis_protocol_sha256"] != sha(ROOT / "MANUS_SPATIAL_PAIRED_PROTOCOL.json")):
        raise AssertionError("reduced MANUS paired analysis is not verified")
    mapping = (("MANUS_SPATIAL_SCREEN.csv", "feature_family_results.csv", family),
               ("MANUS_SPATIAL_CONDITIONAL.csv", "conditional_incremental.csv", conditional),
               ("MANUS_SPATIAL_COMPLEMENTARITY.csv", "error_complementarity.csv", errors))
    exports = {}
    for source_name, target_name, convert in mapping:
        source = ROOT / source_name
        with source.open(newline="", encoding="utf-8") as stream:
            incoming = list(csv.DictReader(stream))
        if len(incoming) != audit["rows"][source_name]:
            raise AssertionError("source cell count changed")
        target = RESULTS / target_name
        with target.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            if not fields:
                raise AssertionError("canonical source table missing")
            retained = [row for row in reader if row["run_id"] != RUN_ID]
        added = []
        for row in incoming:
            values = convert(row)
            values.update(run_id=RUN_ID, core_definition_mode="new_independent_v2_reduced_no_rest",
                          protocol="source_session1_trial_model_exact_native_trial_pairing")
            if not set(values) <= set(fields):
                raise AssertionError(f"canonical schema lacks MANUS field: {target_name}")
            added.append({field: str(values.get(field, "")) for field in fields})
        with target.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(retained + added)
        exports[target_name] = {"rows": len(added), "source_sha256": sha(source),
                                "output_sha256": sha(target)}
    delivery = {"status": "ok", "run_id": RUN_ID, "exports": exports,
                "interaction_sha256": sha(ROOT / "MANUS_SPATIAL_INTERACTION.csv"),
                "paired_audit_sha256": sha(audit_path), "boundary": audit["boundary"]}
    (ROOT / "MANUS_SPATIAL_DELIVERY_AUDIT.json").write_text(
        json.dumps(delivery, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_id": RUN_ID,
                      "rows": {name: value["rows"] for name, value in exports.items()}}))
    return delivery


if __name__ == "__main__":
    export()
