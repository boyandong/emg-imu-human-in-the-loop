"""Independent trial and metric readback of the centered-formula GRAB replay."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, recall_score


ROOT = Path(__file__).resolve().parents[1] / "benchmarks" / "new_bank_v3"
CLASSES = np.asarray([4, 15, 16, 17])


def test_spec_spatial_grab_readback() -> None:
    protocol = ROOT / "SPEC_SPATIAL_GRAB_PROTOCOL.json"
    predictions = ROOT / "SPEC_SPATIAL_GRAB_PREDICTIONS.csv"
    result = json.loads((ROOT / "SPEC_SPATIAL_GRAB_RESULTS.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(protocol.read_bytes()).hexdigest() == result["protocol_sha256"]
    assert hashlib.sha256(predictions.read_bytes()).hexdigest() == result["prediction_sha256"]
    with predictions.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == result["prediction_rows"] == 1344
    assert len({(r["arm"], r["trial_id"]) for r in rows}) == 1344
    # Per-trial mean and standard deviation double each window-level dimension.
    assert result["dimensions"] == {"F0": 96, "F2a_spec": 72, "F3c_spec": 48}
    for arm in ("F0", "F0+F2a_spec", "F0+F3c_spec"):
        for split in ("validation", "final"):
            selected = [r for r in rows if r["arm"] == arm and r["split"] == split]
            assert len(selected) == 224
            labels = np.asarray([int(r["gesture"]) for r in selected])
            assert {int(c): int((labels == c).sum()) for c in CLASSES} == {
                int(c): 56 for c in CLASSES
            }
            assert {int(r["subject"]) for r in selected} == set(range(1, 9))
            probabilities = np.asarray([[float(r[f"p_{c}"]) for c in CLASSES]
                                        for r in selected])
            assert np.isfinite(probabilities).all() and (probabilities >= 0).all()
            np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-12, rtol=0)
            predictions_labels = CLASSES[np.argmax(probabilities, axis=1)]
            score = result["scores"][arm][split]
            np.testing.assert_allclose(score["macro_f1"], f1_score(
                labels, predictions_labels, labels=CLASSES, average="macro"))
            np.testing.assert_allclose(score["accuracy"], accuracy_score(labels, predictions_labels))
            np.testing.assert_allclose(score["log_loss"], log_loss(
                labels, probabilities, labels=CLASSES))
            recalls = recall_score(labels, predictions_labels, labels=CLASSES, average=None)
            for c, value in zip(CLASSES, recalls):
                np.testing.assert_allclose(score["per_class_recall"][str(c)], value)
