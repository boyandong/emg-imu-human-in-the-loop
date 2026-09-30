"""Independently read back the added document-F2a three-axis guard."""
import csv
import hashlib
import json
from pathlib import Path

import pytest


ROOT=Path(__file__).resolve().parents[1]/"benchmarks"/"new_bank_v2"


def test_document_f2a_decision_uses_validation_only_and_frozen_sources():
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    audit=json.loads((ROOT/"F2A_DOCUMENT_CROSS_AXIS_AUDIT.json").read_text(encoding="utf-8"))
    rows_path=ROOT/"F2A_DOCUMENT_CROSS_AXIS_CELLS.csv"
    assert audit["cells_sha256"]==digest(rows_path)
    assert audit["wearing_protocol_sha256"]==digest(ROOT/"F2A_DOCUMENT_WEARING_PROTOCOL.json")
    assert audit["transfer_protocol_sha256"]==digest(ROOT/"F2A_DOCUMENT_TRANSFER_PROTOCOL.json")
    assert audit["existing_guard_protocol_sha256"]==digest(ROOT/"V1_CROSS_AXIS_DECISION_PROTOCOL.json")
    for name,expected in audit["source_results_sha256"].items():
        assert expected==digest(ROOT/name)
    with rows_path.open(newline="",encoding="utf-8") as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==audit["cells"]==6
    axes={"wearing_shift":120,"manus_session":108,"grab_unseen_user":56}
    assert {(row["phase"],row["axis"]) for row in rows}=={
        (phase,axis) for phase in ("validation","final") for axis in axes}
    for row in rows:
        assert int(row["trials"])==axes[row["axis"]]
        assert float(row["delta_macro_f1"])==pytest.approx(
            float(row["candidate_macro_f1"])-float(row["f0_macro_f1"]),abs=1e-12)
        assert float(row["delta_log_loss"])==pytest.approx(
            float(row["candidate_log_loss"])-float(row["f0_log_loss"]),abs=1e-12)
    violations=[f"{row['axis']}:{metric}" for row in rows if row["phase"]=="validation"
                for metric,failed in (("macro_f1",float(row["delta_macro_f1"])<-1e-12),
                                      ("log_loss",float(row["delta_log_loss"])>1e-12)) if failed]
    assert violations==audit["validation_violations"]
    assert audit["candidate_eligible_for_public_default"] is False
    assert audit["three_axis_default"]=="F0v2"
