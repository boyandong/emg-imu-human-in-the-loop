"""Independent F2c native probability, metric and paired-error readback."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, recall_score


ROOT = Path(__file__).resolve().parents[1] / "benchmarks" / "new_bank_v3"
CLASSES = np.asarray([4, 15, 16, 17])


def test_spec_f2c_grab_readback() -> None:
    protocol = ROOT / "SPEC_F2C_GRAB_PROTOCOL.json"
    prediction_path = ROOT / "SPEC_F2C_GRAB_PREDICTIONS.csv"
    parent_path = ROOT / "SPEC_SPATIAL_GRAB_PREDICTIONS.csv"
    result = json.loads((ROOT / "SPEC_F2C_GRAB_RESULTS.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(protocol.read_bytes()).hexdigest() == result["protocol_sha256"]
    assert hashlib.sha256(prediction_path.read_bytes()).hexdigest() == result["prediction_sha256"]
    assert hashlib.sha256(parent_path.read_bytes()).hexdigest() == result["parent_prediction_sha256"]
    assert result["source_reference_windows"] == 4480
    assert result["feature_dimension"] == 168  # F0 96 + F2c mean/std 72.
    with prediction_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    with parent_path.open(newline="", encoding="utf-8") as stream:
        parent = {(r["split"], r["trial_id"]): r for r in csv.DictReader(stream)
                  if r["arm"] == "F0"}
    assert len(rows) == result["prediction_rows"] == len(parent) == 448
    assert {(r["split"], r["trial_id"]) for r in rows} == set(parent)
    for split, corrections, new_errors in (("validation", 6, 3), ("final", 7, 11)):
        selected = [r for r in rows if r["split"] == split]
        assert len(selected) == 224
        labels = np.asarray([int(r["gesture"]) for r in selected])
        assert {int(c): int((labels == c).sum()) for c in CLASSES} == {
            int(c): 56 for c in CLASSES
        }
        probabilities = np.asarray([[float(r[f"p_{c}"]) for c in CLASSES]
                                    for r in selected])
        assert np.isfinite(probabilities).all() and (probabilities >= 0).all()
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-12, rtol=0)
        prediction = CLASSES[np.argmax(probabilities, axis=1)]
        baseline = np.asarray([CLASSES[np.argmax(
            [float(parent[(r["split"], r["trial_id"])][f"p_{c}"]) for c in CLASSES])]
            for r in selected])
        assert int(np.sum((baseline != labels) & (prediction == labels))) == corrections
        assert int(np.sum((baseline == labels) & (prediction != labels))) == new_errors
        score = result["scores"][split]
        np.testing.assert_allclose(score["macro_f1"], f1_score(
            labels, prediction, labels=CLASSES, average="macro"))
        np.testing.assert_allclose(score["accuracy"], accuracy_score(labels, prediction))
        np.testing.assert_allclose(score["log_loss"], log_loss(
            labels, probabilities, labels=CLASSES))
        recalls = recall_score(labels, prediction, labels=CLASSES, average=None)
        for c, value in zip(CLASSES, recalls):
            np.testing.assert_allclose(score["per_class_recall"][str(c)], value)
