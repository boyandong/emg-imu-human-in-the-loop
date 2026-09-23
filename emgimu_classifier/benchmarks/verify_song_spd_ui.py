"""Offscreen project-page smoke check with the real local Song SPD bundle."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np

from benchmarks.song_real8_study import _hash


def verify(source: Path, collection_root: Path, bundle: Path, output: Path) -> dict:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    sys.path.insert(0, str(collection_root.resolve()))
    from PySide6.QtWidgets import QApplication
    from emgforce.inference.song_local import SongLocalRuntime
    from emgforce.ui.realtime_inference_page import RealtimeInferencePage

    bundle = bundle.resolve()
    models_root = collection_root.resolve() / "models"
    if bundle.parent != models_root:
        raise ValueError("bundle must be in the project app's actual models directory")
    source_file = source / "2026-09-18_S04/session.h5"
    readiness = json.loads((source_file.parent / "SESSION_COLLECTION_READINESS.json").read_text(
        encoding="utf-8"))
    source_sha = _hash(source_file)
    if source_sha != readiness["hdf5_sha256"]:
        raise ValueError("S04 recording differs from saved readiness digest")
    with h5py.File(source_file) as handle:
        raw = handle["streams/emg/raw"][:1000]
        indices = handle["streams/emg/sample_index"][:1000]
    if raw.shape != (1000, 8) or not np.array_equal(indices, np.arange(1000)):
        raise ValueError("Song UI smoke input is not a contiguous 8-channel stream")

    expected_runtime = SongLocalRuntime(bundle)
    _, expected_frames = expected_runtime.ingest(raw, indices)
    app = QApplication.instance() or QApplication([])
    page = RealtimeInferencePage(models_root)
    observed = []
    try:
        page.refresh_models()
        key = f"song::{bundle}"
        selected = page.model_combo.findData(key)
        if selected < 0:
            raise AssertionError("project page did not discover the F0+SPD bundle")
        page.model_combo.setCurrentIndex(selected)
        page.set_connected(True)
        page.load_selected_model()
        deadline = time.monotonic() + 5
        while page.bundle is None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        if page.bundle is None or page.bundle.runtime_backend != "song_real8_f0_spd":
            raise AssertionError("project page did not load the SPD runtime")
        page.worker.prediction_ready.connect(observed.append)
        page.start_recognition()
        page.ingest_emg(raw, indices)
        deadline = time.monotonic() + 5
        while not observed and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        app.processEvents()
        if not observed:
            raise AssertionError("project page did not emit a prediction")
        frame = observed[-1]
        expected_index, expected_probability = expected_frames[-1]
        error = float(np.max(np.abs(frame.probabilities - expected_probability)))
        if frame.output_sample_index != expected_index or error > 2e-5:
            raise AssertionError("UI worker prediction differs from direct Song runtime")
        if "数据龄未测" not in page.current_probability.text():
            raise AssertionError("UI should not claim a measured device-data age")
        result = {
            "status": "offscreen_project_page_real_model_replay_verified",
            "source_s04_sha256": source_sha,
            "bundle_sha256": expected_runtime.sha256,
            "bundle_dir": str(bundle),
            "ui_model_id": page.bundle.model_id,
            "runtime_backend": page.bundle.runtime_backend,
            "input_samples": len(raw),
            "direct_runtime_frames": len(expected_frames),
            "ui_prediction_frames_emitted": len(observed),
            "latest_output_sample_index": int(frame.output_sample_index),
            "max_latest_probability_error": error,
            "physical_usb_tested": False,
            "boundary": "Actual project model selector and Qt page loaded the local F0+SPD bundle and processed saved raw S04 samples. Offscreen replay does not validate USB acquisition, wall-clock pacing, screen visibility to a person, live gesture accuracy or end-to-end latency.",
        }
    finally:
        if not page.shutdown():
            raise RuntimeError("Song Qt worker did not stop cleanly")
        page.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "status", "runtime_backend", "direct_runtime_frames",
        "ui_prediction_frames_emitted", "max_latest_probability_error")}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--collection-root", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    verify(args.source, args.collection_root, args.bundle, args.output)
