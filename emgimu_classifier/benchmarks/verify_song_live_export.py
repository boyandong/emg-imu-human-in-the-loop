"""Verify exported Song runtime probabilities and chunked causal replay."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmarks.song_real8_study import _filter_emg, _join_batches, load_session
from emgimu.feature_bank.families import LocalDetailFamily, SpdTangentFamily


def verify(source: Path, bundle: Path, collection_root: Path, output: Path):
    sys.path.insert(0, str(collection_root.resolve()))
    from emgforce.inference.song_local import SongLocalRuntime

    data = {sid: load_session(source / f"2026-09-18_{sid}", sid, "causal")
            for sid in ("S01", "S02", "S03")}
    source_batch = _join_batches([data["S01"], data["S02"]])
    family = LocalDetailFamily().fit(source_batch)
    runtime = SongLocalRuntime(bundle)
    spd = SpdTangentFamily().fit(source_batch) if runtime.with_spd else None

    def matrix(sid):
        pieces = [family.transform(data[sid]["batch"])]
        if spd is not None:
            pieces.append(spd.transform(data[sid]["batch"]))
        return np.concatenate(pieces, axis=1)

    train_x = np.concatenate([matrix(sid) for sid in ("S01", "S02")])
    train_y = np.concatenate([data[sid]["hand"] for sid in ("S01", "S02")])
    source_model = make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=2000, random_state=0))
    source_model.fit(train_x, train_y)
    if runtime.manifest["source_hdf5_sha256"] != {sid: data[sid]["audit"]["sha256"] for sid in data}:
        raise ValueError("exported Song model source hashes differ from recordings")
    if not np.allclose(runtime.thresholds, family.thresholds_, rtol=0, atol=1e-12):
        raise ValueError("exported F0 thresholds differ from source fit")
    if spd is not None:
        reference = np.ascontiguousarray(spd.reference_)
        digest = hashlib.sha256(reference.tobytes()).hexdigest()
        if runtime.manifest.get("source_spd_reference_sha256") != digest:
            raise ValueError("exported SPD source-reference digest differs from source fit")
        exported = json.loads(runtime.artifact.read_text(encoding="utf-8"))["spd_reference"]
        if not np.allclose(exported, reference, rtol=0, atol=1e-12):
            raise ValueError("exported SPD source reference differs from source fit")
    validation = data["S03"]["batch"].emg
    expected = source_model.predict_proba(matrix("S03"))
    actual = np.stack([runtime.predict_filtered_window(window) for window in validation])
    probability_error = float(np.max(np.abs(expected - actual)))
    if probability_error > 2e-5 or not np.array_equal(np.argmax(expected, axis=1), np.argmax(actual, axis=1)):
        raise ValueError(f"exported runtime differs from source model: max error {probability_error}")
    test_path = source / "2026-09-18_S04" / "session.h5"
    with h5py.File(test_path) as handle:
        raw = handle["streams/emg/raw"][:25000]
    offline_filtered = _filter_emg(raw, "causal")
    runtime.reset()
    frames = []
    chunk_ms = []
    for start in range(0, len(raw), 37):
        began = time.perf_counter()
        gap, output_frames = runtime.ingest(raw[start:start + 37], np.arange(start, min(start + 37, len(raw))))
        chunk_ms.append((time.perf_counter() - began) * 1000.0)
        if gap:
            raise ValueError("continuous source stream reported a gap")
        frames.extend(output_frames)
    replay_error = max(float(np.max(np.abs(
        probability - runtime.predict_filtered_window(offline_filtered[index - 49:index + 1]))))
        for index, probability in frames)
    if replay_error > 2e-5 or len(frames) != 999:
        raise ValueError(f"chunked Song replay differs from offline causal filtering: {replay_error}")
    result = {"status": "export_and_chunked_causal_replay_verified",
              "model_kind": "F0_plus_SPD" if runtime.with_spd else "F0",
              "artifact_sha256": runtime.sha256, "source_hdf5_sha256": runtime.manifest["source_hdf5_sha256"],
              "S03_validation_windows": len(validation),
              "max_probability_error_vs_source_sklearn": probability_error,
              "S04_replayed_samples": len(raw), "S04_replayed_200ms_frames": len(frames),
              "max_probability_error_chunked_vs_offline_filter": replay_error,
              "local_chunk_processing_p95_ms": float(np.percentile(chunk_ms, 95)),
              "local_chunk_processing_max_ms": float(np.max(chunk_ms)),
              "nominal_37_sample_chunk_duration_ms": 148.0,
              "boundary": "Exact same stable-window and first-100-second causal replay semantics. Chunk timings are local CPU processing only; no USB/UI age, continuous-event accuracy or end-to-end latency claim."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "S03_validation_windows", "S04_replayed_200ms_frames",
                                                "max_probability_error_vs_source_sklearn",
                                                "max_probability_error_chunked_vs_offline_filter")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--collection-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    verify(args.source, args.bundle, args.collection_root, args.output)
