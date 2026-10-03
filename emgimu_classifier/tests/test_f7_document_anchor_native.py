"""Read back the frozen F7 public-trial delivery without requiring raw GRAB files."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "benchmarks" / "new_bank_v3"


def test_f7_document_anchor_trial_delivery():
    protocol_path = HERE / "F7_DOCUMENT_ANCHOR_PROTOCOL.json"
    result = json.loads((HERE / "F7_DOCUMENT_ANCHOR_RESULTS.json").read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    prediction_path = HERE / "F7_DOCUMENT_ANCHOR_PREDICTIONS.csv"
    if result["protocol_sha256"] != hashlib.sha256(protocol_path.read_bytes()).hexdigest():
        pytest.fail("protocol changed after native replay")
    if result["prediction_sha256"] != hashlib.sha256(prediction_path.read_bytes()).hexdigest():
        pytest.fail("predictions changed after native replay")
    with prediction_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == result["row_count"] == 1152
    keys = [(r["phase"], int(r["subject"]), int(r["budget"]), r["arm"], r["trial_id"]) for r in rows]
    assert len(set(keys)) == len(keys)
    assert Counter((r["phase"], int(r["budget"]), r["arm"]) for r in rows) == {
        (phase, budget, arm): 64 for phase in protocol["target_days"]
        for budget in protocol["shots_per_class"] for arm in ("F0v2", "F7_document", "F0v2+F7_document")
    }
    classes = protocol["classes"]
    for row in rows:
        probs = np.array([float(row[f"p_{label}"]) for label in classes])
        assert np.isfinite(probs).all() and np.all(probs >= 0)
        assert np.isclose(probs.sum(), 1, atol=1e-8)
    for cell in result["calibration_cells"]:
        budget = cell["budget"]
        assert len(cell["calibration_trial_ids"]) == 4 * budget
        assert len(cell["evaluation_trial_ids"]) == 8
        assert not set(cell["calibration_trial_ids"]) & set(cell["evaluation_trial_ids"])
        assert {int(s.split("_trial")[-1]) for s in cell["calibration_trial_ids"]} == set(range(1, budget + 1))
        assert {int(s.split("_trial")[-1]) for s in cell["evaluation_trial_ids"]} == {6, 7}
        assert cell["calibration_fitted_tau"] > 0
    for phase in protocol["target_days"]:
        for budget in protocol["shots_per_class"]:
            cell_rows = [r for r in rows if r["phase"] == phase and int(r["budget"]) == budget]
            for trial_id in {r["trial_id"] for r in cell_rows}:
                same = {r["arm"]: r for r in cell_rows if r["trial_id"] == trial_id}
                p0 = np.array([float(same["F0v2"][f"p_{label}"]) for label in classes])
                p7 = np.array([float(same["F7_document"][f"p_{label}"]) for label in classes])
                mix = np.array([float(same["F0v2+F7_document"][f"p_{label}"]) for label in classes])
                np.testing.assert_allclose(mix, 0.5 * (p0 + p7), atol=1e-12)
