"""Per-subject effects are paired and sourced from frozen family screens."""
from __future__ import annotations

import csv
from pathlib import Path

from benchmarks.new_bank_v2.roam_posture_run import sha256
from benchmarks.new_bank_v2.v1_cross_axis_subjects import build

ROOT = Path(__file__).resolve().parents[1] / "benchmarks" / "new_bank_v2"


def test_v1_subject_effects_replay_all_axes() -> None:
    prior = {name: sha256(ROOT / f"V1_CROSS_AXIS_SUBJECT_{name}.csv")
             for name in ("CELLS", "SUMMARY")}
    audit = build()
    assert audit["subject_cells"] == 310
    assert audit["summary_cells"] == 70
    assert audit["subject_cells_sha256"] == prior["CELLS"]
    assert audit["summary_cells_sha256"] == prior["SUMMARY"]
    for filename, digest in audit["source_sha256"].items():
        assert sha256(ROOT / filename) == digest
    with (ROOT / "V1_CROSS_AXIS_SUBJECT_SUMMARY.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len({(r["axis"], r["phase"], r["arm"]) for r in rows}) == 70
    assert all(int(r["improved_subjects"]) + int(r["worsened_subjects"])
               + int(r["unchanged_subjects"]) == int(r["subjects"]) for r in rows)
