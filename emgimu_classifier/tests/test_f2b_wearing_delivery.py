"""The isolated document CSP study keeps paired, source-frozen native trials."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from benchmarks.new_bank_v2 import wearing_v1_extension_run as wearing
from benchmarks.new_bank_v2.roam_posture_run import sha256

ROOT = Path(__file__).resolve().parents[1] / "benchmarks/new_bank_v2"


def test_document_csp_wearing_predictions_and_selection_boundary() -> None:
    protocol = json.loads((ROOT / "F2B_WEARING_PROTOCOL.json").read_text(encoding="utf-8"))
    result = json.loads((ROOT / "F2B_WEARING_RESULTS.json").read_text(encoding="utf-8"))
    path = ROOT / "F2B_WEARING_PREDICTIONS.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert result["protocol_sha256"] == sha256(ROOT / "F2B_WEARING_PROTOCOL.json")
    assert result["prediction_sha256"] == sha256(path)
    assert result["prediction_rows"] == len(rows) == 480
    assert result["f0v2_parent_replay_max_abs_error"] == 0.0
    assert result["feature_dimensions"] == {"F0v2": 48, "F2b_document": 20}
    for phase in ("validation", "final"):
        keys = {arm: {(r["subject"], r["domain"], r["trial_id"])
                      for r in rows if r["phase"] == phase and r["arm"] == arm}
                for arm in protocol["arms"]}
        assert keys["F0v2"] == keys["F0v2+F2b_document"]
        assert len(keys["F0v2"]) == 120
        for arm in protocol["arms"]:
            part = [{**r, "subject": int(r["subject"]), "label": int(r["label"]),
                     **{f"p_{c}": float(r[f"p_{c}"]) for c in protocol["classes"]}}
                    for r in rows if r["phase"] == phase and r["arm"] == arm]
            probabilities = np.asarray([[r[f"p_{c}"] for c in protocol["classes"]]
                                        for r in part])
            np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-10)
            observed = wearing._score(part, phase, arm)
            # _score selects by phase and arm; the saved subset has both fixed.
            for key in ("macro_f1", "log_loss", "brier"):
                assert abs(observed["pooled"][key] - result["scores"][phase][arm]["pooled"][key]) < 1e-12
    validation = result["scores"]["validation"]
    assert (validation["F0v2+F2b_document"]["pooled"]["macro_f1"]
            < validation["F0v2"]["pooled"]["macro_f1"])
