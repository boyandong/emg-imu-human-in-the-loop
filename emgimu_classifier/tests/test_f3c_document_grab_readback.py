"""Independent metric and identity readback for the document F3c GRAB screen."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, recall_score


ROOT = Path(__file__).resolve().parents[1] / "benchmarks" / "new_bank_v2"
CLASSES = np.asarray([4, 15, 16, 17])


def test_document_f3c_grab_readback() -> None:
    result = json.loads((ROOT / "F3C_DOCUMENT_GRAB_RESULTS.json").read_text(encoding="utf-8"))
    protocol = ROOT / "F3C_DOCUMENT_GRAB_PROTOCOL.json"
    predictions = ROOT / "F3C_DOCUMENT_GRAB_PREDICTIONS.csv"
    assert hashlib.sha256(protocol.read_bytes()).hexdigest() == result["protocol_sha256"]
    assert hashlib.sha256(predictions.read_bytes()).hexdigest() == result["prediction_sha256"]
    with predictions.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == result["prediction_rows"] == 896
    assert len({(r["arm"], r["trial_id"]) for r in rows}) == 896
    for arm in ("F0", "F0+F3c_document"):
        for split in ("validation", "final"):
            selected = [r for r in rows if r["arm"] == arm and r["split"] == split]
            assert len(selected) == 224
            assert {int(r["subject"]) for r in selected} == set(range(1, 9))
            labels = np.asarray([int(r["gesture"]) for r in selected])
            assert {int(c): int((labels == c).sum()) for c in CLASSES} == {
                int(c): 56 for c in CLASSES
            }
            proba = np.asarray([[float(r[f"p_{c}"]) for c in CLASSES] for r in selected])
            np.testing.assert_allclose(proba.sum(axis=1), 1, atol=1e-12, rtol=0)
            assert np.isfinite(proba).all() and (proba >= 0).all()
            predicted = CLASSES[np.argmax(proba, axis=1)]
            score = result["scores"][arm][split]
            np.testing.assert_allclose(score["accuracy"], accuracy_score(labels, predicted))
            np.testing.assert_allclose(score["macro_f1"], f1_score(labels, predicted,
                                                                    labels=CLASSES, average="macro"))
            np.testing.assert_allclose(score["log_loss"], log_loss(labels, proba, labels=CLASSES))
            recalls = recall_score(labels, predicted, labels=CLASSES, average=None)
            for c, recall in zip(CLASSES, recalls):
                np.testing.assert_allclose(score["per_class_recall"][str(c)], recall)
            for subject in range(1, 9):
                mask = np.asarray([int(r["subject"]) == subject for r in selected])
                np.testing.assert_allclose(
                    score["per_subject_macro_f1"][str(subject)],
                    f1_score(labels[mask], predicted[mask], labels=CLASSES, average="macro"),
                )
