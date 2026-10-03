"""Descriptive 8-channel F0 Rest-only versus pooled-source threshold replay."""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from benchmarks.song_real8_study import load_session
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.document_signal import RestNoiseLocalDetailFamily
from emgimu.feature_bank.families import LocalDetailFamily


HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "F0_REST_NOISE_PROTOCOL.json"
PARENT = HERE / "F9_DOCUMENT_V3_PROTOCOL.json"
CSV = HERE / "F0_REST_NOISE_DIFFERENCES.csv"
RESULT = HERE / "F0_REST_NOISE_RESULTS.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(source: Path) -> dict:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    parent = json.loads(PARENT.read_text(encoding="utf-8"))
    if sha(PARENT) != protocol["parent_f9_protocol_sha256"]:
        raise ValueError("parent Song session inventory changed")
    if (protocol["source_sessions"] != parent["source_sessions"]
            or protocol["evaluation_sessions"] != parent["evaluation_sessions"]
            or protocol["sample_rate_hz"] != parent["emg_sample_rate_hz"]
            or protocol["window_samples"] != parent["window_samples"]
            or protocol["channels"] != parent["channels"]):
        raise ValueError("Song native protocol diverged from frozen parent")
    sessions = {}
    for name in protocol["source_sessions"] + protocol["evaluation_sessions"]:
        data = load_session(source / f"2026-09-18_{name}", name, filter_mode="causal")
        if (data["audit"]["sha256"] != parent["expected_session_sha256"][name]
                or len(data["trial"]) != parent["expected_formal_windows"][name]):
            raise ValueError(f"Song source or trial inventory changed: {name}")
        sessions[name] = data
        print(f"F0 loaded {name}: {len(data['trial'])} windows", flush=True)
    source_ids = set(np.concatenate([sessions[s]["trial"] for s in protocol["source_sessions"]]).tolist())
    target_ids = set(np.concatenate([sessions[s]["trial"] for s in protocol["evaluation_sessions"]]).tolist())
    if source_ids & target_ids:
        raise ValueError("source and evaluation trial identities overlap")
    x = np.concatenate([sessions[s]["batch"].emg for s in protocol["source_sessions"]])
    y = np.concatenate([sessions[s]["hand"] for s in protocol["source_sessions"]])
    if x.shape[1:] != (protocol["window_samples"], protocol["channels"]):
        raise ValueError("native feature shape changed")
    batch = FeatureBatch(x, protocol["sample_rate_hz"])
    reference = LocalDetailFamily().fit(batch)
    document = RestNoiseLocalDetailFamily(rest_label=protocol["rest_label"]).fit(batch, y)
    if document.rest_windows_ != int(np.sum(y == protocol["rest_label"])):
        raise ValueError("source Rest count changed")
    frozen = (pickle.dumps(reference), pickle.dumps(document))
    rows = []
    summary = {}
    fields = ["session", "trial_id", "base_zc", "rest_zc", "base_ssc", "rest_ssc",
              "base_wamp", "rest_wamp"]
    for name in protocol["evaluation_sessions"]:
        data = sessions[name]
        base = reference.transform(data["batch"])
        rest = document.transform(data["batch"])
        if base.shape != rest.shape or base.shape[1] != 48 or not np.isfinite(rest).all():
            raise ValueError("native F0 feature dimension or finiteness changed")
        if not np.array_equal(base[:, :24], rest[:, :24]):
            raise ValueError("threshold-free F0 coordinates changed")
        for index, trial_id in enumerate(data["trial"]):
            row = {"session": name, "trial_id": trial_id}
            for metric, lo, hi in (("zc", 24, 32), ("ssc", 32, 40), ("wamp", 40, 48)):
                row[f"base_{metric}"] = int(np.sum(base[index, lo:hi]))
                row[f"rest_{metric}"] = int(np.sum(rest[index, lo:hi]))
            rows.append(row)
        sub = [r for r in rows if r["session"] == name]
        summary[name] = {"windows": len(sub), "unique_trials": len(set(r["trial_id"] for r in sub)),
                         "threshold_sensitive_fraction": float(np.mean([any(
                             r[f"base_{metric}"] != r[f"rest_{metric}"]
                             for metric in ("zc", "ssc", "wamp")) for r in sub])),
                         "mean_rest_minus_base": {metric: float(np.mean([
                             r[f"rest_{metric}"] - r[f"base_{metric}"] for r in sub]))
                             for metric in ("zc", "ssc", "wamp")}}
        print(f"F0 summarized {name}: {len(sub)} windows", flush=True)
    if frozen != (pickle.dumps(reference), pickle.dumps(document)):
        raise ValueError("target transform mutated source-fitted family")
    with CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    result = {"status": "descriptive_feature_only", "protocol_sha256": sha(PROTOCOL),
              "parent_f9_protocol_sha256": sha(PARENT),
              "source_hdf5_sha256": {s: parent["expected_session_sha256"][s]
                                     for s in protocol["source_sessions"]},
              "source_windows": len(x), "source_rest_windows": document.rest_windows_,
              "pooled_source_thresholds": reference.thresholds_.tolist(),
              "rest_only_thresholds": document.thresholds_.tolist(),
              "difference_rows": len(rows), "difference_csv_sha256": sha(CSV),
              "evaluation": summary, "boundary": protocol["boundary"]}
    RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    run(args.source)
