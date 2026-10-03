"""Independent trial-level readback for matched Song F0 threshold classifiers."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score


HERE = Path(__file__).resolve().parents[1] / "benchmarks" / "song_real8"


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_f0_rest_model_paired_trial_delivery():
    protocol_path = HERE / "F0_REST_MODEL_PROTOCOL.json"
    parent_path = HERE / "F0_REST_NOISE_PROTOCOL.json"
    prediction_path = HERE / "F0_REST_MODEL_TRIAL_PREDICTIONS.csv"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    result = json.loads((HERE / "F0_REST_MODEL_RESULTS.json").read_text(encoding="utf-8"))
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    native = json.loads((HERE / "F9_DOCUMENT_V3_PROTOCOL.json").read_text(encoding="utf-8"))
    assert result["protocol_sha256"] == _sha(protocol_path)
    assert result["parent_f0_protocol_sha256"] == protocol["parent_f0_protocol_sha256"] == _sha(parent_path)
    assert result["prediction_csv_sha256"] == _sha(prediction_path)
    assert result["source_hdf5_sha256"] == {
        s: native["expected_session_sha256"][s] for s in parent["source_sessions"]
    }
    with prediction_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == result["prediction_rows"] == 568
    assert Counter((r["session"], r["arm"]) for r in rows) == {
        ("S03", arm): 140 for arm in protocol["arms"]
    } | {("S04", arm): 144 for arm in protocol["arms"]}
    keys = [(r["session"], r["arm"], r["trial_id"]) for r in rows]
    assert len(keys) == len(set(keys))
    classes = protocol["classes"]
    for session in protocol["read_only_sessions"]:
        pairs = {arm: {r["trial_id"]: r for r in rows if r["session"] == session and r["arm"] == arm}
                 for arm in protocol["arms"]}
        assert set(pairs[protocol["arms"][0]]) == set(pairs[protocol["arms"][1]])
        for trial_id in pairs[protocol["arms"][0]]:
            assert len({pairs[arm][trial_id]["hand"] for arm in protocol["arms"]}) == 1
            assert len({pairs[arm][trial_id]["windows"] for arm in protocol["arms"]}) == 1
        for arm in protocol["arms"]:
            selected = list(pairs[arm].values())
            y = np.asarray([r["hand"] for r in selected])
            p = np.asarray([[float(r[f"p_{label}"]) for label in classes] for r in selected])
            assert np.isfinite(p).all() and np.all(p >= 0) and np.allclose(p.sum(axis=1), 1, atol=1e-8)
            assert all(1 <= int(r["windows"]) <= 3 for r in selected)
            pred = np.asarray(classes)[p.argmax(axis=1)]
            target = np.asarray([classes.index(label) for label in y])
            saved = result["metrics"][session][arm]
            assert saved["trials"] == len(y)
            assert np.isclose(saved["macro_f1"], f1_score(y, pred, labels=classes,
                                                           average="macro", zero_division=0))
            assert np.isclose(saved["log_loss"], -np.log(np.clip(p[np.arange(len(y)), target],
                                                               1e-15, 1)).mean())
            assert np.isclose(saved["brier"], np.mean(np.sum(
                (p - (y[:, None] == np.asarray(classes))) ** 2, axis=1)))
