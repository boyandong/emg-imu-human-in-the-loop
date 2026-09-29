"""Held-wearing-domain split and matched prediction delivery checks."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from benchmarks.new_bank_v2.roam_posture_run import sha256
from benchmarks.new_bank_v2.wearing_v1_extension_analysis import analyze

ROOT = Path(__file__).resolve().parents[1] / "benchmarks" / "new_bank_v2"
RUN_ID = "feature_bank_v1_wearing_extension_20260929"


def test_wearing_v1_source_and_target_are_disjoint() -> None:
    result = json.loads((ROOT / "WEARING_V1_EXTENSION_RESULTS.json").read_text(encoding="utf-8"))
    assert result["protocol_sha256"] == sha256(ROOT / "WEARING_V1_EXTENSION_PROTOCOL.json")
    assert result["prediction_sha256"] == sha256(ROOT / "WEARING_V1_EXTENSION_PREDICTIONS.csv")
    assert result["baseline_replay_max_abs_error"] == 0
    assert result["validation_selected_arm"] == "F0v2+correlation_spectrum"
    assert len(result["split_trial_ids"]) == 6
    for phase in ("validation", "final"):
        for subject in result["protocol"][f"{phase}_subjects"]:
            split = result["split_trial_ids"][f"{phase}_{subject}"]
            assert len(set(split["target"])) == 40
            assert not set(split["source"]) & set(split["target"])
            assert all(f"/subject{subject}/training/" in trial for trial in split["source"])
            assert all(f"/subject{subject}/trial_" in trial for trial in split["target"])


def test_wearing_v1_all_native_pair_cells_are_delivered() -> None:
    prior = {name: sha256(ROOT / f"WEARING_V1_EXTENSION_{name}.csv")
             for name in ("FAMILY", "CONDITIONAL", "ERROR")}
    audit = analyze()
    assert audit["rows"] == {"FAMILY": 80, "CONDITIONAL": 64, "ERROR": 160}
    assert audit["sha256"] == prior
    results = ROOT.parents[1] / "feature_bank" / "results"
    for name, expected in (("feature_family_results.csv", 80),
                           ("conditional_incremental.csv", 64),
                           ("error_complementarity.csv", 160)):
        with (results / name).open(newline="", encoding="utf-8") as stream:
            rows = [row for row in csv.DictReader(stream) if row["run_id"] == RUN_ID]
        assert len(rows) == expected
    with (ROOT / "WEARING_V1_EXTENSION_ERROR.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len({(r["phase"], r["scope"], r["subject"], r["condition"],
                 r["family_a"], r["family_b"]) for r in rows}) == 160
