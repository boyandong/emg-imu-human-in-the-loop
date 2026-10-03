"""Portable readback for the source-frozen Song F0 threshold comparison."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parents[1] / "benchmarks" / "song_real8"


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_f0_rest_noise_native_feature_readback():
    protocol_path = HERE / "F0_REST_NOISE_PROTOCOL.json"
    parent_path = HERE / "F9_DOCUMENT_V3_PROTOCOL.json"
    csv_path = HERE / "F0_REST_NOISE_DIFFERENCES.csv"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    result = json.loads((HERE / "F0_REST_NOISE_RESULTS.json").read_text(encoding="utf-8"))
    assert result["protocol_sha256"] == _sha(protocol_path)
    assert result["parent_f9_protocol_sha256"] == protocol["parent_f9_protocol_sha256"] == _sha(parent_path)
    assert result["difference_csv_sha256"] == _sha(csv_path)
    assert result["source_hdf5_sha256"] == {
        s: parent["expected_session_sha256"][s] for s in protocol["source_sessions"]
    }
    assert result["source_windows"] == 849 and result["source_rest_windows"] == 213
    base = np.asarray(result["pooled_source_thresholds"])
    rest = np.asarray(result["rest_only_thresholds"])
    assert base.shape == rest.shape == (8,) and np.isfinite(base).all() and np.isfinite(rest).all()
    assert np.all(rest > 0) and np.all(rest < base)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == result["difference_rows"] == 847
    assert Counter(row["session"] for row in rows) == {"S03": 416, "S04": 431}
    for session, item in result["evaluation"].items():
        selected = [row for row in rows if row["session"] == session]
        assert len(selected) == item["windows"] == parent["expected_formal_windows"][session]
        assert len(set(row["trial_id"] for row in selected)) == item["unique_trials"]
        changed = []
        for row in selected:
            differences = []
            for metric in ("zc", "ssc", "wamp"):
                a, b = int(row[f"base_{metric}"]), int(row[f"rest_{metric}"])
                assert 0 <= a <= b <= 8 * protocol["window_samples"]
                differences.append(b - a)
            changed.append(any(differences))
        assert np.isclose(np.mean(changed), item["threshold_sensitive_fraction"])
        for metric in ("zc", "ssc", "wamp"):
            delta = np.mean([int(row[f"rest_{metric}"]) - int(row[f"base_{metric}"])
                             for row in selected])
            assert np.isclose(delta, item["mean_rest_minus_base"][metric])
