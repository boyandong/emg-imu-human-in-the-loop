"""The MANUS F4d correction shares fixed source models and matched targets."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from benchmarks.grabmyo_crossday.run import score
from benchmarks.new_bank_v2.roam_posture_run import sha256

ROOT = Path(__file__).resolve().parents[1] / "benchmarks/new_bank_v2"


def test_f4d_predictive_trial_readback_and_scope() -> None:
    protocol_path = ROOT / "F4D_MANUS_PREDICTIVE_PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    parent = json.loads((ROOT / "F4D_MANUS_SESSION_RESULTS.json").read_text(encoding="utf-8"))
    result = json.loads((ROOT / "F4D_MANUS_PREDICTIVE_RESULTS.json").read_text(encoding="utf-8"))
    path = ROOT / "F4D_MANUS_PREDICTIVE_PREDICTIONS.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert result["protocol_sha256"] == sha256(protocol_path)
    assert result["parent_result_sha256"] == sha256(ROOT / "F4D_MANUS_SESSION_RESULTS.json")
    assert result["prediction_sha256"] == sha256(path)
    assert result["prediction_rows"] == len(rows) == 432
    assert result["rest_windows"] == 1828
    classes = np.arange(6)
    for phase in ("validation", "final"):
        expected = {trial for user in parent["users"].values()
                    for group in user["sessions"][phase]["held_out"].values()
                    for trial in group["native_trials"]}
        assert len(expected) == 72
        for arm in protocol["arms"]:
            part = [r for r in rows if r["phase"] == phase and r["arm"] == arm]
            assert len(part) == 72
            assert {r["trial_id"] for r in part} == expected
            y = np.asarray([int(r["label"]) for r in part])
            p = np.asarray([[float(r[f"p_{c}"]) for c in classes] for r in part])
            users = np.asarray([int(r["user"]) for r in part])
            np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-10)
            observed = score(y, p, classes, users)
            for key in ("macro_f1", "log_loss", "brier"):
                assert abs(observed[key] - result["scores"][phase][arm]["pooled"][key]) < 1e-12
    validation = result["scores"]["validation"]
    assert (validation["F0v2+F4d_session"]["pooled"]["macro_f1"]
            > validation["F0v2"]["pooled"]["macro_f1"])
    final = result["scores"]["final"]
    assert (final["F0v2+F4d_session"]["pooled"]["minimum_subject_macro_f1"]
            < final["F0v2"]["pooled"]["minimum_subject_macro_f1"])
