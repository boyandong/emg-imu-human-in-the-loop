"""Read back the frozen native label-free session prediction evidence."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1] / "benchmarks" / "new_bank_v2"


def test_native_session_prediction_rows_match_frozen_protocol():
    protocol_path = ROOT / "SESSION_UNLABELED_PROTOCOL.json"
    result = json.loads((ROOT / "SESSION_UNLABELED_RESULTS.json").read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    rows_path = ROOT / "SESSION_UNLABELED_PREDICTIONS.csv"
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest(protocol_path) == result["protocol_sha256"]
    assert digest(ROOT / "WEARING_PROTOCOL.json") == protocol["parent_wearing_protocol_sha256"]
    assert digest(rows_path) == result["prediction_rows_sha256"]
    source = set(result["source_trial_ids"])
    calibration = set(result["calibration_trial_ids"])
    evaluation = set(result["evaluation_trial_ids"])
    assert (len(source), len(calibration), len(evaluation)) == (25, 5, 5)
    assert not (source & calibration or source & evaluation or calibration & evaluation)
    assert all("/training/" in trial for trial in source)
    assert all("/trial_1/R_0_C_" in trial for trial in calibration)
    assert all("/trial_1/R_1_C_" in trial for trial in evaluation)
    with rows_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == result["prediction_rows"] == 60
    expected = {(branch, family, trial)
                for branch in result["branches"] for family in result["families"]
                for trial in evaluation}
    assert {(row["branch"], row["family"], row["trial_id"]) for row in rows} == expected
    for row in rows:
        label = int(row["trial_id"].rsplit("_C_", 1)[1].removesuffix(".csv"))
        assert int(row["offline_truth_for_audit"]) == label
        probabilities = np.array([float(row[f"p_{i}"]) for i in range(5)])
        assert np.isfinite(probabilities).all()
        assert (probabilities >= 0).all()
        np.testing.assert_allclose(probabilities.sum(), 1., atol=1e-12)
    assert result["max_abs_probability_difference"] <= 1e-12
    assert len(result["source_profile_id"]) == 64
    assert result["session_bound_to_source_profile"] is True
    assert result["wrong_offline_truth_changed_predictions"] is False
    assert result["source_and_session_immutable"] is True
