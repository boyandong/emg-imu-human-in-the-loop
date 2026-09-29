"""The new-version canonical deliveries have no missing required values."""
from __future__ import annotations

from benchmarks.new_bank_v2.v1_canonical_schema_audit import build


def test_new_v1_required_fields_and_runs() -> None:
    audit = build()
    assert audit["status"] == "new_v1_required_fields_complete"
    assert audit["total_checked_rows"] == 5312
    assert all(not any(item["missing_field_counts"].values())
               for item in audit["artifacts"].values())
