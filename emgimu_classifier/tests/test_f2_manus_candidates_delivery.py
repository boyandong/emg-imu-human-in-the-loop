"""Frozen MANUS F2b/F2c paired-trial delivery readback."""
import csv
import json
from pathlib import Path

import numpy as np

from benchmarks.new_bank_v2.manus_spatial_run import score
from benchmarks.new_bank_v2.roam_posture_run import sha256


ROOT = Path(__file__).resolve().parents[1] / "benchmarks/new_bank_v2"


def test_f2_manus_candidate_trial_readback():
    protocol_path = ROOT / "F2_MANUS_CANDIDATES_PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    result = json.loads((ROOT / "F2_MANUS_CANDIDATES_RESULTS.json").read_text(encoding="utf-8"))
    parent = json.loads((ROOT / "MANUS_REST_TRANSFER_RESULTS.json").read_text(encoding="utf-8"))
    prediction_path = ROOT / "F2_MANUS_CANDIDATES_PREDICTIONS.csv"
    with prediction_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert result["protocol_sha256"] == sha256(protocol_path)
    assert result["parent_result_sha256"] == sha256(ROOT / "MANUS_REST_TRANSFER_RESULTS.json")
    assert result["parent_prediction_sha256"] == sha256(ROOT / "MANUS_REST_TRANSFER_PREDICTIONS.csv")
    assert result["prediction_sha256"] == sha256(prediction_path)
    assert result["prediction_rows"] == len(rows) == 648
    assert result["f0v2_parent_replay_max_abs_error"] == 0.0
    assert result["feature_dimensions"] == {"F0v2": 48, "F2b_document": 24, "F2c_spd": 36}
    for phase in ("validation", "final"):
        expected = set(parent["split_trial_ids"][phase]["target_trials"])
        assert len(expected) == 108
        for arm in protocol["arms"]:
            part = [row for row in rows if row["phase"] == phase and row["arm"] == arm]
            assert len(part) == 108
            assert {row["trial_id"] for row in part} == expected
            probabilities = np.asarray([[float(row[f"p_{c}"]) for c in range(6)]
                                        for row in part])
            np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-10)
            recalculated = score(rows, phase, arm)
            saved = result["scores"][phase][arm]
            for key in ("macro_f1", "log_loss", "brier"):
                assert abs(recalculated["pooled"][key] - saved["pooled"][key]) < 1e-12
            assert abs(recalculated["minimum_user_macro_f1"] - saved["minimum_user_macro_f1"]) < 1e-12
    validation = result["scores"]["validation"]
    baseline = validation["F0v2"]["pooled"]
    for arm in protocol["arms"][1:]:
        assert validation[arm]["pooled"]["macro_f1"] > baseline["macro_f1"]
        assert validation[arm]["pooled"]["log_loss"] > baseline["log_loss"]
