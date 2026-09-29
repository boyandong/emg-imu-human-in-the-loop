"""The complete five-family study retains matched, replayable native evidence."""
from __future__ import annotations

import csv
from pathlib import Path

from benchmarks.new_bank_v2.full_v1_lofo_audit import audit
from benchmarks.new_bank_v2.full_v1_lofo_run import ARMS, PROTOCOL

ROOT = Path(__file__).resolve().parents[1]


def test_full_v1_lofo_all_axes_and_frozen_controls() -> None:
    result = audit()
    assert result["cells"] == len(PROTOCOL["axes"]) * len(ARMS) * 2 == 98
    assert sum(v["prediction_rows"] for v in result["axis_audits"].values()) == 26684
    assert all(v["f0v2_parent_replay_max_abs_error"] <= 1e-8
               for v in result["axis_audits"].values())
    with (ROOT / "benchmarks/new_bank_v2/FULL_V1_LOFO_CELLS.csv").open(
            newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert {(r["axis"], r["phase"], r["arm"]) for r in rows} == {
        (axis, phase, arm) for axis in PROTOCOL["axes"]
        for phase in ("validation", "final") for arm in ARMS}
    assert set(axis for axis, delta in result["validation_full_minus_F0v2"].items()
               if delta > 0) == {"observed_speed"}
