"""Idempotently export frozen native-bout ROAM posture evidence."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from benchmarks.new_bank_v1.export_paired_delivery import conditional, errors, family

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parent.parent / "feature_bank" / "results"
RUN_ID = "feature_bank_roam_posture_v2_20260929"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export() -> dict:
    audit_path = ROOT / "ROAM_POSTURE_PAIRED_AUDIT.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (audit["status"] != "ok" or audit["matched_cells"] != 60
            or audit["saved_score_groups_replayed"] != 40
            or audit["analysis_protocol_sha256"] != sha(ROOT / "ROAM_POSTURE_PAIRED_PROTOCOL.json")):
        raise AssertionError("ROAM paired audit unavailable")
    envelope = json.loads((ROOT / "ENVELOPE_POSTURE_AUDIT.json").read_text(encoding="utf-8"))
    if envelope["status"] != "ok" or envelope["rows"] != 8:
        raise AssertionError("five-axis envelope unavailable")
    mapping = (("ROAM_POSTURE_SCREEN.csv", "feature_family_results.csv", family),
               ("ROAM_POSTURE_CONDITIONAL.csv", "conditional_incremental.csv", conditional),
               ("ROAM_POSTURE_COMPLEMENTARITY.csv", "error_complementarity.csv", errors))
    exports = {}
    for source_name, target_name, convert in mapping:
        source = ROOT / source_name
        with source.open(newline="", encoding="utf-8") as stream:
            incoming = list(csv.DictReader(stream))
        if len(incoming) != audit["rows"][source_name]:
            raise AssertionError("ROAM source row count changed")
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
                          core_definition_mode="new_independent_v2_public_posture_bout",
                          protocol="source_users1to18_resting_exact_bout_pairing")
            if not set(values) <= set(fields):
                raise AssertionError(f"canonical schema lacks ROAM field: {target_name}")
            added.append({field: str(values.get(field, "")) for field in fields})
        with target.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(retained + added)
        exports[target_name] = {"rows": len(added), "source_sha256": sha(source),
                                "output_sha256": sha(target)}
    delivery = {"status": "ok", "run_id": RUN_ID, "exports": exports,
                "interaction_sha256": sha(ROOT / "ROAM_POSTURE_INTERACTION.csv"),
                "envelope_sha256": sha(ROOT / "ROBUSTNESS_ENVELOPE_5AXIS.csv"),
                "paired_audit_sha256": sha(audit_path), "boundary": audit["boundary"]}
    (ROOT / "ROAM_POSTURE_DELIVERY_AUDIT.json").write_text(
        json.dumps(delivery, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_id": RUN_ID,
                      "rows": {name: value["rows"] for name, value in exports.items()}}))
    return delivery


if __name__ == "__main__":
    export()
