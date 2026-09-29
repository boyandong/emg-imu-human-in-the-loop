"""Qualified native MANUS speed-screen provenance and paired-delivery checks."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from benchmarks.new_bank_v2.manus_v1_speed_analysis import analyze
from benchmarks.new_bank_v2.roam_posture_run import sha256

ROOT = Path(__file__).resolve().parents[1] / "benchmarks" / "new_bank_v2"
RUN_ID = "feature_bank_v1_manus_speed_external_rest_20260929"


def test_manus_v1_speed_frozen_source_and_sessions() -> None:
    result = json.loads((ROOT / "MANUS_V1_SPEED_RESULTS.json").read_text(encoding="utf-8"))
    protocol = result["protocol"]
    assert result["protocol_sha256"] == sha256(ROOT / "MANUS_V1_SPEED_PROTOCOL.json")
    assert result["prediction_sha256"] == sha256(ROOT / "MANUS_V1_SPEED_PREDICTIONS.csv")
    assert result["baseline_replay_max_abs_error"] <= 1e-8
    assert result["validation_selected_arm"] == "F0v2+ring_lag"
    assert protocol["rest_prior"].startswith("MANUS has no labelled Rest")
    for phase, session in (("validation", 2), ("final", 3)):
        split = result["split_trial_ids"][phase]
        assert len(set(split["source_trials"])) == 108
        assert len(set(split["target_trials"])) == 108
        assert not set(split["source_trials"]) & set(split["target_trials"])
        assert all(f"/s_{session}/" in trial.lower() for trial in split["target_trials"])


def test_manus_v1_speed_all_matched_cells_delivered() -> None:
    prior = {name: sha256(ROOT / f"MANUS_V1_SPEED_{name}.csv")
             for name in ("FAMILY", "CONDITIONAL", "ERROR")}
    audit = analyze()
    assert audit["rows"] == {"FAMILY": 280, "CONDITIONAL": 224, "ERROR": 560}
    assert audit["sha256"] == prior
    results = ROOT.parents[1] / "feature_bank" / "results"
    for name, expected in (("feature_family_results.csv", 280),
                           ("conditional_incremental.csv", 224),
                           ("error_complementarity.csv", 560)):
        with (results / name).open(newline="", encoding="utf-8") as stream:
            rows = [row for row in csv.DictReader(stream) if row["run_id"] == RUN_ID]
        assert len(rows) == expected
    with (ROOT / "MANUS_V1_SPEED_ERROR.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len({(r["phase"], r["scope"], r["subject"], r["condition"],
                 r["family_a"], r["family_b"]) for r in rows}) == 560
