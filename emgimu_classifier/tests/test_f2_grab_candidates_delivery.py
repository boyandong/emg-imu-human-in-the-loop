"""Unseen-user GRAB F2 candidate delivery and parent replay readback."""
import csv
import json
from pathlib import Path

import numpy as np

from benchmarks.grabmyo_crossday import run as grab
from benchmarks.new_bank_v2.roam_posture_run import sha256


ROOT = Path(__file__).resolve().parents[1] / "benchmarks/new_bank_v2"


def test_f2_grab_candidate_readback():
    protocol_path = ROOT / "F2_GRAB_CANDIDATES_PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    result = json.loads((ROOT / "F2_GRAB_CANDIDATES_RESULTS.json").read_text(encoding="utf-8"))
    parent = json.loads((ROOT / "GRAB_USER_RESULTS.json").read_text(encoding="utf-8"))
    path = ROOT / "F2_GRAB_CANDIDATES_PREDICTIONS.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert result["protocol_sha256"] == sha256(protocol_path)
    assert result["parent_result_sha256"] == sha256(ROOT / "GRAB_USER_RESULTS.json")
    assert result["parent_prediction_sha256"] == sha256(ROOT / "GRAB_USER_PREDICTIONS.csv")
    assert result["prediction_sha256"] == sha256(path)
    assert result["prediction_rows"] == len(rows) == 336
    assert result["f0v2_parent_replay_max_abs_error"] == 0.0
    assert result["feature_dimensions"] == {"F0v2": 48, "F2b_document": 16, "F2c_spd": 36}
    classes = np.asarray(grab.GESTURES)
    for phase in ("validation", "final"):
        expected = set(parent[f"{phase}_trial_ids"])
        assert len(expected) == 56
        for arm in protocol["arms"]:
            part = [row for row in rows if row["phase"] == phase and row["arm"] == arm]
            assert len(part) == 56
            assert {row["trial_id"] for row in part} == expected
            labels = np.asarray([int(row["gesture"]) for row in part])
            subjects = np.asarray([int(row["subject"]) for row in part])
            probabilities = np.asarray([[float(row[f"p_{c}"]) for c in classes]
                                        for row in part])
            np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-10)
            observed = grab.score(labels, probabilities, classes, subjects)
            for key in ("macro_f1", "log_loss", "brier", "minimum_subject_macro_f1"):
                assert abs(observed[key] - result["scores"][arm][phase][key]) < 1e-12
    for phase in ("validation", "final"):
        baseline = result["scores"]["F0v2"][phase]["macro_f1"]
        for arm in protocol["arms"][1:]:
            assert result["scores"][arm][phase]["macro_f1"] < baseline
        assert parent["scores"]["F0v2+F2a"][phase]["macro_f1"] < baseline
