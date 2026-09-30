"""Native full-bout F5c paired predictions and frozen G5 replay."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.new_bank_v2.f5c_unibo_bout_run import _score
from benchmarks.new_bank_v2.roam_posture_run import sha256


ROOT = Path(__file__).resolve().parents[1] / "benchmarks/new_bank_v2"


def _read():
    protocol_path = ROOT / "F5C_UNIBO_BOUT_PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    result = json.loads((ROOT / "F5C_UNIBO_BOUT_RESULTS.json").read_text(encoding="utf-8"))
    prediction_path = ROOT / "F5C_UNIBO_BOUT_PREDICTIONS.csv"
    with prediction_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return protocol_path, protocol, result, prediction_path, rows


def test_f5c_unibo_saved_trial_and_score_readback():
    protocol_path, protocol, result, prediction_path, raw = _read()
    assert result["protocol_sha256"] == sha256(protocol_path)
    assert result["prediction_sha256"] == sha256(prediction_path)
    assert result["prediction_rows"] == len(raw) == 3*(1700+3391)
    assert result["bout_counts"] == {"validation": 1700, "final": 3391}
    assert result["parent_g5_replay_max_abs_error"] == {"validation": 0., "final": 0.}
    assert (result["g5_dimension"], result["signature_dimension"]) == (20, 20)
    rows = [{**row, "day": int(row["day"]), "posture": int(row["posture"]),
             "label": int(row["label"]), "weight": float(row["weight"]),
             **{f"p_{c}": float(row[f"p_{c}"]) for c in range(4)}} for row in raw]
    for phase, count in result["bout_counts"].items():
        identities = None
        for arm in protocol["arms"]:
            selected = [row for row in rows if row["phase"] == phase and row["arm"] == arm]
            assert len(selected) == count
            current = {row["bout_id"] for row in selected}
            assert len(current) == count
            if identities is None:
                identities = current
            else:
                assert current == identities
            probabilities = np.asarray([[row[f"p_{c}"] for c in range(4)] for row in selected])
            np.testing.assert_allclose(probabilities.sum(axis=1), 1., atol=1e-10)
            recalculated = _score(rows, phase, arm)
            stored = result["scores"][phase][arm]
            for key in ("macro_f1", "log_loss", "brier"):
                assert abs(recalculated["pooled"][key] - stored["pooled"][key]) < 1e-10
            assert abs(recalculated["minimum_user_macro_f1"] - stored["minimum_user_macro_f1"]) < 1e-10
    assert result["scores"]["validation"]["G5+F5c"]["pooled"]["macro_f1"] > result["scores"]["validation"]["G5"]["pooled"]["macro_f1"]
    assert result["scores"]["final"]["G5+F5c"]["pooled"]["macro_f1"] < result["scores"]["final"]["G5"]["pooled"]["macro_f1"]


def test_f5c_unibo_replay_against_external_frozen_parent():
    _, protocol, result, _, raw = _read()
    for phase, parent_key in (("validation", "frozen_parent_source"),
                              ("final", "frozen_parent_final")):
        parent = Path(protocol[parent_key]) / "heldout_predictions.npz"
        if not parent.exists():
            pytest.skip("frozen external UniBo parent is unavailable on this machine")
        expected_sha_key = ("parent_validation_prediction_sha256" if phase == "validation"
                            else "parent_final_prediction_sha256")
        assert sha256(parent) == protocol[expected_sha_key] == result[expected_sha_key]
        with np.load(parent, allow_pickle=False) as saved:
            rows = [row for row in raw if row["phase"] == phase and row["arm"] == "G5"]
            np.testing.assert_array_equal([row["bout_id"] for row in rows], saved["bout_ids"])
            np.testing.assert_array_equal([int(row["label"]) for row in rows], saved["labels"])
            np.testing.assert_allclose([[float(row[f"p_{c}"]) for c in range(4)] for row in rows],
                                       saved["G5"], atol=1e-12, rtol=0)
    splits = result["split_trial_ids"]
    for user, groups in splits.items():
        sets = [set(value) for value in groups.values()]
        assert all(group for group in sets), user
        assert not any(first & second for index, first in enumerate(sets)
                       for second in sets[index+1:]), user
