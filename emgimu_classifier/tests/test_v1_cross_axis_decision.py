"""Seven-axis default-bank decision is bound to saved validation evidence."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from benchmarks.new_bank_v2.roam_posture_run import sha256
from benchmarks.new_bank_v2.v1_cross_axis_decision import build

ROOT = Path(__file__).resolve().parents[1] / "benchmarks" / "new_bank_v2"


def test_v1_cross_axis_saved_source_and_decision() -> None:
    prior = sha256(ROOT / "V1_CROSS_AXIS_DECISION_CELLS.csv")
    audit = build()
    assert audit["saved_cells_sha256"] == prior
    assert audit["deployment_default"] == "F0v2"
    assert audit["eligible_additions"] == []
    assert audit["selection_uses_final"] is False
    assert set(audit["validation_violations"]) == {
        "F0v2+scale_pattern", "F0v2+ring_lag",
        "F0v2+correlation_spectrum", "F0v2+frequency_direction"}
    assert all(audit["validation_violations"].values())
    vector = audit["robustness_vectors"]["validation"]
    assert len(vector["F0v2"]["axis_macro_f1"]) == 7
    assert abs(vector["F0v2"]["mean_available_macro_f1"] - 0.7475529747820113) < 1e-12
    assert abs(vector["F0v2"]["minimum_available_macro_f1"] - 0.3913744923084004) < 1e-12
    protocol = json.loads((ROOT / "V1_CROSS_AXIS_DECISION_PROTOCOL.json").read_text(encoding="utf-8"))
    assert audit["protocol_sha256"] == sha256(ROOT / "V1_CROSS_AXIS_DECISION_PROTOCOL.json")
    for source, digest in protocol["source_results_sha256"].items():
        assert sha256(ROOT / f"{source}_RESULTS.json") == digest


def test_v1_cross_axis_complete_matched_cells() -> None:
    with (ROOT / "V1_CROSS_AXIS_DECISION_CELLS.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 70
    assert len({(r["axis"], r["phase"], r["arm"]) for r in rows}) == 70
    assert {r["axis"] for r in rows} == {
        "posture", "unseen_user", "cross_day", "force_intensity",
        "wearing_shift", "observed_speed", "synthetic_quality"}
    assert all(float(r["delta_macro_f1"]) == 0 and float(r["delta_log_loss"]) == 0
               for r in rows if r["arm"] == "F0v2")
