"""Frozen force-v1 trial pairing and canonical evidence read-back."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from benchmarks.new_bank_v2.force_v1_extension_analysis import analyze
from benchmarks.new_bank_v2.roam_posture_run import sha256


ROOT = Path(__file__).resolve().parents[1] / "benchmarks" / "new_bank_v2"
RUN_ID = "feature_bank_v1_force_intensity_extension_20260929"


def test_force_v1_extension_preserves_parent_and_held_subjects() -> None:
    result = json.loads((ROOT / "FORCE_V1_EXTENSION_RESULTS.json").read_text(encoding="utf-8"))
    assert result["protocol_sha256"] == sha256(ROOT / "FORCE_V1_EXTENSION_PROTOCOL.json")
    assert result["prediction_sha256"] == sha256(ROOT / "FORCE_V1_EXTENSION_PREDICTIONS.csv")
    assert result["baseline_replay_max_abs_error"] == 0
    assert result["validation_selected_arm"] == "F0v2"
    source = set(result["split_trial_ids"]["validation"]["source_trials"])
    assert len(source) == 168
    for phase in ("validation", "final"):
        partition = result["split_trial_ids"][phase]
        assert set(partition["source_subjects"]) == {1, 2, 3, 4, 5, 6}
        assert not set(partition["target_subjects"]) & set(partition["source_subjects"])
        assert len(set(partition["target_trials"])) == 588
        assert not source & set(partition["target_trials"])


def test_force_v1_all_pair_cells_and_delivery() -> None:
    before = {name: sha256(ROOT / f"FORCE_V1_EXTENSION_{name}.csv")
              for name in ("FAMILY", "CONDITIONAL", "ERROR")}
    audit = analyze()
    assert audit["rows"] == {"FAMILY": 140, "CONDITIONAL": 112, "ERROR": 280}
    assert audit["sha256"] == before
    results = ROOT.parents[1] / "feature_bank" / "results"
    for name, expected in (("feature_family_results.csv", 140),
                           ("conditional_incremental.csv", 112),
                           ("error_complementarity.csv", 280)):
        with (results / name).open(newline="", encoding="utf-8") as stream:
            rows = [row for row in csv.DictReader(stream) if row["run_id"] == RUN_ID]
        assert len(rows) == expected
    with (ROOT / "FORCE_V1_EXTENSION_ERROR.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len({(r["phase"], r["scope"], r["subject"], r["condition"],
                 r["family_a"], r["family_b"]) for r in rows}) == 280
