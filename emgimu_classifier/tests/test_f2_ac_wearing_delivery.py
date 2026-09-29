"""Three spatial candidates share one frozen wearing-domain comparison."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from benchmarks.new_bank_v2 import wearing_v1_extension_run as wearing
from benchmarks.new_bank_v2.roam_posture_run import sha256

ROOT = Path(__file__).resolve().parents[1] / "benchmarks/new_bank_v2"


def test_trace_spd_wearing_readback_matches_frozen_f2b_targets() -> None:
    protocol_path = ROOT / "F2_AC_WEARING_PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    result = json.loads((ROOT / "F2_AC_WEARING_RESULTS.json").read_text(encoding="utf-8"))
    path = ROOT / "F2_AC_WEARING_PREDICTIONS.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    with (ROOT / "F2B_WEARING_PREDICTIONS.csv").open(newline="", encoding="utf-8") as stream:
        previous = list(csv.DictReader(stream))
    assert result["protocol_sha256"] == sha256(protocol_path)
    assert result["prediction_sha256"] == sha256(path)
    assert result["prediction_rows"] == len(rows) == 480
    assert result["parent_prediction_sha256"] == sha256(ROOT / "F2B_WEARING_PREDICTIONS.csv")
    assert result["feature_dimensions"] == {"F0v2": 48, "F2a_trace": 36, "F2c_spd": 36}
    for phase in ("validation", "final"):
        expected = {r["trial_id"] for r in previous
                    if r["phase"] == phase and r["arm"] == "F0v2"}
        assert len(expected) == 120
        for arm in protocol["arms"]:
            part = [{**r, "subject": int(r["subject"]), "label": int(r["label"]),
                     **{f"p_{c}": float(r[f"p_{c}"]) for c in range(5)}}
                    for r in rows if r["phase"] == phase and r["arm"] == arm]
            assert {r["trial_id"] for r in part} == expected
            p = np.asarray([[r[f"p_{c}"] for c in range(5)] for r in part])
            np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-10)
            readback = wearing._score(part, phase, arm)
            for key in ("macro_f1", "log_loss", "brier"):
                assert abs(readback["pooled"][key]
                           - result["scores"][phase][arm]["pooled"][key]) < 1e-12
    parent = json.loads((ROOT / "F2B_WEARING_RESULTS.json").read_text(encoding="utf-8"))
    baseline = parent["scores"]["validation"]["F0v2"]["pooled"]["macro_f1"]
    assert result["scores"]["validation"]["F0v2+F2a_trace"]["pooled"]["macro_f1"] > baseline
    assert result["scores"]["final"]["F0v2+F2a_trace"]["pooled"]["macro_f1"] \
        < parent["scores"]["final"]["F0v2"]["pooled"]["macro_f1"]
