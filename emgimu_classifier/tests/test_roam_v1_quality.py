"""Frozen synthetic-fault pairing and canonical evidence checks."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from benchmarks.new_bank_v2.roam_posture_run import sha256
from benchmarks.new_bank_v2.roam_v1_quality_analysis import analyze

ROOT = Path(__file__).resolve().parents[1] / "benchmarks" / "new_bank_v2"
RUN_ID = "feature_bank_v1_roam_synthetic_quality_20260929"


def test_roam_v1_quality_parent_replay_and_disjoint_users() -> None:
    result = json.loads((ROOT / "ROAM_V1_QUALITY_RESULTS.json").read_text(encoding="utf-8"))
    protocol = result["protocol"]
    assert result["protocol_sha256"] == sha256(ROOT / "ROAM_V1_QUALITY_PROTOCOL.json")
    assert result["prediction_sha256"] == sha256(ROOT / "ROAM_V1_QUALITY_PREDICTIONS.csv")
    assert result["baseline_replay_max_abs_error"] == 0
    assert result["validation_selected_arm"] == "F0v2"
    assert len(protocol["conditions"]) == 14
    source = set(protocol["source_subjects"])
    validation = set(protocol["validation_subjects"])
    final = set(protocol["final_subjects"])
    assert not source & validation and not source & final and not validation & final
    for phase in ("validation", "final"):
        ids = result["split_trial_ids"][phase]
        assert len(set(ids)) == 45
        assert set(ids).isdisjoint(result["source_trial_ids"])


def test_roam_v1_quality_matched_fault_cells_delivered() -> None:
    prior = {name: sha256(ROOT / f"ROAM_V1_QUALITY_{name}.csv")
             for name in ("FAMILY", "CONDITIONAL", "ERROR")}
    audit = analyze()
    assert audit["rows"] == {"FAMILY": 840, "CONDITIONAL": 672, "ERROR": 1680}
    assert audit["sha256"] == prior
    assert audit["saved_score_groups_replayed"] == 840
    results = ROOT.parents[1] / "feature_bank" / "results"
    for name, expected in (("feature_family_results.csv", 840),
                           ("conditional_incremental.csv", 672),
                           ("error_complementarity.csv", 1680)):
        with (results / name).open(newline="", encoding="utf-8") as stream:
            rows = [row for row in csv.DictReader(stream) if row["run_id"] == RUN_ID]
        assert len(rows) == expected
    with (ROOT / "ROAM_V1_QUALITY_ERROR.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len({(r["phase"], r["condition"], r["scope"], r["subject"],
                 r["family_a"], r["family_b"]) for r in rows}) == 1680
