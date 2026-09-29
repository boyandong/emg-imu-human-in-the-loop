"""F0–F9 scope decisions remain attached to current public evidence."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from benchmarks.formula_family_scope_audit import sha

ROOT = Path(__file__).resolve().parents[1]


def test_family_scope_audit_has_all_ten_families_and_hashed_evidence() -> None:
    directory = ROOT / "feature_bank"
    table = directory / "FORMULA_FAMILY_SCOPE_AUDIT.csv"
    audit = json.loads((directory / "FORMULA_FAMILY_SCOPE_AUDIT.json").read_text(encoding="utf-8"))
    with table.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert audit["completion_proven"] is False
    assert audit["families"] == len(rows) == 10
    assert audit["csv_sha256"] == sha(table)
    assert audit["formula_review_sha256"] == sha(directory / "FORMULA_IMPLEMENTATION_AUDIT.csv")
    assert [r["family"] for r in rows] == [f"F{i}" for i in range(10)]
    for row in rows:
        evidence = json.loads(row["evidence_sha256_json"])
        assert evidence
        assert all(sha(ROOT / path) == digest for path, digest in evidence.items())
        assert row["supported_conclusion"] and row["remaining_boundary"]
