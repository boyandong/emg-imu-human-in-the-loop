"""Idempotently export source-only external-Rest MANUS speed evidence."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from benchmarks.new_bank_v1.export_paired_delivery import conditional, errors, family

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parent.parent / "feature_bank" / "results"
RUN_ID = "feature_bank_manus_external_rest_speed_v2_20260929"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export() -> dict:
    audit_path = ROOT / "MANUS_REST_TRANSFER_PAIRED_AUDIT.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (audit["status"] != "ok" or audit["matched_cells"] != 56
            or audit["parent_score_groups_replayed"] != 80
            or audit["analysis_protocol_sha256"] != sha(ROOT / "MANUS_REST_TRANSFER_PAIRED_PROTOCOL.json")):
        raise AssertionError("external-Rest MANUS paired audit unavailable")
    envelope = json.loads((ROOT / "ENVELOPE_SPEED_AUDIT.json").read_text(encoding="utf-8"))
    if envelope["status"] != "ok" or envelope["rows"] != 8:
        raise AssertionError("qualified seven-axis envelope unavailable")
    mapping = (("MANUS_REST_TRANSFER_SCREEN.csv", "feature_family_results.csv", family),
               ("MANUS_REST_TRANSFER_CONDITIONAL.csv", "conditional_incremental.csv", conditional),
               ("MANUS_REST_TRANSFER_COMPLEMENTARITY.csv", "error_complementarity.csv", errors))
    exports = {}
    for source_name, target_name, convert in mapping:
        source = ROOT / source_name
        with source.open(newline="", encoding="utf-8") as stream:
            incoming = list(csv.DictReader(stream))
        if len(incoming) != audit["rows"][source_name]:
            raise AssertionError("MANUS source row count changed")
        target = RESULTS / target_name
        with target.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            if not fields:
                raise AssertionError("canonical table unavailable")
            retained = [row for row in reader if row["run_id"] != RUN_ID]
        added = []
        for row in incoming:
            values = convert(row)
            values.update(run_id=RUN_ID,
                          core_definition_mode="new_independent_v2_external_source_rest_speed",
                          protocol="manus_session1_source_roam_rest_prior_exact_trial_pairing")
            if not set(values) <= set(fields):
                raise AssertionError(f"canonical schema lacks speed field: {target_name}")
            added.append({field: str(values.get(field, "")) for field in fields})
        with target.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(retained + added)
        exports[target_name] = {"rows": len(added), "source_sha256": sha(source),
                                "output_sha256": sha(target)}
    delivery = {"status": "ok", "run_id": RUN_ID, "exports": exports,
                "interaction_sha256": sha(ROOT / "MANUS_REST_TRANSFER_INTERACTION.csv"),
                "envelope_sha256": sha(ROOT / "ROBUSTNESS_ENVELOPE_7AXIS_QUALIFIED.csv"),
                "paired_audit_sha256": sha(audit_path), "boundary": audit["boundary"]}
    (ROOT / "MANUS_REST_TRANSFER_DELIVERY_AUDIT.json").write_text(
        json.dumps(delivery, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_id": RUN_ID,
                      "rows": {name: value["rows"] for name, value in exports.items()}}))
    return delivery


if __name__ == "__main__":
    export()
