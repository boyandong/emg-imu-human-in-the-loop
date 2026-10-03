"""Portable native D/E replay checks against bundled public-derived assets."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from emgimu.feature_bank.document_reliability_v2 import DocumentReliabilityWeightsV2


ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "benchmarks" / "new_bank_v3"


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_document_reliability_grab_native_readback():
    protocol_path = HERE / "DOCUMENT_RELIABILITY_GRAB_PROTOCOL.json"
    prediction_path = HERE / "DOCUMENT_RELIABILITY_GRAB_PREDICTIONS.csv"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    result = json.loads((HERE / "DOCUMENT_RELIABILITY_GRAB_RESULTS.json").read_text(encoding="utf-8"))
    assert result["protocol_sha256"] == _hash(protocol_path)
    assert result["prediction_sha256"] == _hash(prediction_path)
    for key, filename in (
        ("source_fitted_f0_features_sha256", "F7_DOCUMENT_ANCHOR_F0v2.npy"),
        ("source_fitted_f0_scale_features_sha256", "DOCUMENT_RELIABILITY_F0_SCALE.npy"),
    ):
        assert protocol[key] == _hash(HERE / filename)
    parent = ROOT / "benchmarks" / "new_bank_v2"
    assert protocol["parent_result_sha256"] == _hash(parent / "V1_FEATURE_ANCHOR_RESULTS.json")
    assert protocol["parent_prediction_sha256"] == _hash(parent / "GRAB_DAY_V1_EXTENSION_PREDICTIONS.csv")

    with prediction_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == result["prediction_rows"] == 576
    keys = [(r["day"], r["subject"], r["budget"], r["arm"], r["trial_id"]) for r in rows]
    assert len(set(keys)) == len(keys)
    assert Counter((int(r["budget"]), r["arm"]) for r in rows) == {
        (budget, arm): 64 for budget in protocol["shots_per_class"] for arm in protocol["arms"]
    }
    for row in rows:
        p = np.array([float(row[f"p_{label}"]) for label in protocol["classes"]])
        assert int(row["day"]) == 3 and np.isfinite(p).all()
        assert np.all(p >= 0) and np.isclose(p.sum(), 1, atol=1e-8)
    assert len(result["source_cv_grid"]) == 9
    assert result["selected_source_policy"] == min(
        result["source_cv_grid"], key=lambda r: (r["mean_log_loss"], r["n0"], r["temperature"])
    )
    assert result["source_population_trial_count"] == 64
    assert len(result["descriptive_cells"]) == 24
    for cell in result["descriptive_cells"]:
        cal, evaluation = cell["calibration_trial_ids"], cell["evaluation_trial_ids"]
        assert len(cal) == 4 * cell["budget"] and len(evaluation) == 8
        assert not set(cal) & set(evaluation)
        assert {int(s.rsplit("_trial", 1)[1]) for s in cal} == set(range(1, cell["budget"] + 1))
        assert {int(s.rsplit("_trial", 1)[1]) for s in evaluation} == {6, 7}
        assert np.isclose(sum(cell["exact_weights"]), 1)
        assert np.isclose(sum(cell["legacy_weights"]), 1)

    # Independently replay one native calibration cell through the delivered formula API.
    trial_ids = json.loads((parent / "V1_FEATURE_ANCHOR_RESULTS.json").read_text(encoding="utf-8"))["target_trial_ids"]
    features = {
        "F0v2": np.load(HERE / "F7_DOCUMENT_ANCHOR_F0v2.npy", allow_pickle=False),
        "F0v2+scale_pattern": np.load(HERE / "DOCUMENT_RELIABILITY_F0_SCALE.npy", allow_pickle=False),
    }
    first = result["descriptive_cells"][0]
    cal = {}
    for provider, x in features.items():
        by_id = dict(zip(trial_ids, x))
        identities = first["calibration_trial_ids"]
        cal[provider] = (np.stack([by_id[trial_id] for trial_id in identities]),
                         np.array([int(trial_id.split("_gesture")[1].split("_trial")[0])
                                   for trial_id in identities]), identities)
    selected = result["selected_source_policy"]
    detail = DocumentReliabilityWeightsV2(
        tuple(protocol["classes"]), tuple(protocol["providers"]),
        tuple(result["source_population"]), selected["n0"], selected["temperature"],
        f"day2-source-cv:{_hash(protocol_path)}",
    ).calculate(cal)
    np.testing.assert_allclose(detail["final"], first["exact_weights"], atol=1e-12)
    np.testing.assert_allclose(detail["between"], first["between"], atol=1e-12)
    np.testing.assert_allclose(detail["within"], first["within"], atol=1e-12)
