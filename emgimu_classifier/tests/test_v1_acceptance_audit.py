"""The reviewed new-version clause matrix remains bound to its evidence."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from benchmarks.new_bank_v2.v1_acceptance_audit import sha

ROOT = Path(__file__).resolve().parents[1]


def test_versioned_acceptance_matrix_evidence_and_open_scope() -> None:
    output = ROOT / "feature_bank"
    audit = json.loads((output / "NEW_VERSION_ACCEPTANCE_AUDIT.json").read_text(encoding="utf-8"))
    table = output / "NEW_VERSION_ACCEPTANCE_AUDIT.csv"
    assert audit["csv_sha256"] == sha(table)
    with table.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == audit["clauses"] == 32
    assert len({row["requirement_id"] for row in rows}) == 32
    assert audit["completion_proven"] is False
    assert set(audit["open_items"]) == {"GOAL-03", "GOAL-10", "GOAL-20", "GOAL-22"}
    for row in rows:
        assert int(row["line_start"]) <= int(row["line_end"])
        assert row["document_sha256"] == audit["documents"][row["document"]]
        hashes = json.loads(row["evidence_sha256_json"])
        assert hashes and all(sha(ROOT / path) == digest for path, digest in hashes.items())
