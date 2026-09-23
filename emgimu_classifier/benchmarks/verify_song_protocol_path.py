"""Replay saved Song EMG through protocol parser, controller and real Qt page.

Packet bytes are reconstructed from saved raw channel values. This validates
software channel order and fan-out; it is not a physical USB acquisition test.
"""
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
    from emgforce.protocol import EMG_TYPE, SYNC, FrameParser
    from emgforce.ui.main_window import MainWindow

    bundle = bundle.resolve()
    collection_root = collection_root.resolve()
    if bundle.parent != collection_root / "models":
        raise ValueError("bundle must belong to this project's collection app")
    session_file = source / "2026-09-18_S04/session.h5"
    readiness = json.loads((session_file.parent / "SESSION_COLLECTION_READINESS.json").read_text(
        encoding="utf-8"))
    source_sha = _hash(session_file)
    if source_sha != readiness["hdf5_sha256"]:
        raise ValueError("source differs from collection readiness digest")
    with h5py.File(session_file) as handle:
        meta = handle["meta"].attrs
        nominal_rate = int(meta["emg_nominal_rate_hz"])
        measured_rate = float(meta["measured_emg_rate_hz"])
        channels = int(meta["num_emg_channels"])
        raw = handle["streams/emg/raw"][:1000]
        indices = handle["streams/emg/sample_index"][:1000]
    if nominal_rate != 250 or channels != 8 or not 245 <= measured_rate <= 255:
        raise ValueError("recording channel/rate metadata incompatible with Song model")
    if raw.shape != (1000, 8) or not np.array_equal(indices, np.arange(1000)):
        raise ValueError("source prefix is not contiguous 8-channel EMG")
    if np.min(raw) < -(1 << 23) or np.max(raw) >= (1 << 23):
        raise ValueError("source values cannot be reconstructed as signed 24-bit packets")

    runtime = SongLocalRuntime(bundle)
    _, expected = runtime.ingest(raw, indices)
    app = QApplication.instance() or QApplication([])
    window = MainWindow(project_root=collection_root)
    page = window.realtime_inference_page
    observed = []
    parser = FrameParser()
    decoded_rows = []
    try:
        page.refresh_models()
        selected = page.model_combo.findData(f"song::{bundle}")
        if selected < 0:
            raise AssertionError("project window cannot discover the Song bundle")
        page.model_combo.setCurrentIndex(selected)
        page.set_connected(True)
        page.load_selected_model()
        deadline = time.monotonic() + 8
        while page.bundle is None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        if page.bundle is None or page.bundle.runtime_backend != "song_real8_f0_spd":
            raise AssertionError("project window did not load the SPD backend")
        page.worker.prediction_ready.connect(observed.append)
        page.start_recognition()
        for start in range(0, len(raw), 37):
            payload = bytearray()
            for sample_number, row in enumerate(raw[start:start + 37], start):
                payload.extend(SYNC)
                payload.extend((EMG_TYPE, sample_number & 255))
                for value in row:
                    payload.extend(int(value).to_bytes(3, "big", signed=True))
            packets = parser.feed(bytes(payload))
            decoded_rows.extend(packet.emg_uv for packet in packets)
            window.acquisition.ingest_packets(packets)
            app.processEvents()
        deadline = time.monotonic() + 8
        while (not observed or observed[-1].output_sample_index != 999) and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        app.processEvents()
        if parser.stats.emg_frames != 1000 or parser.stats.lost_frames:
            raise AssertionError("reconstructed protocol packets did not decode contiguously")
        if not np.array_equal(np.asarray(decoded_rows, dtype=np.int32), raw):
            raise AssertionError("parser changed signed channel values or channel order")
        if window.acquisition.clock.total != 1000:
            raise AssertionError("controller did not forward every unique EMG sample")
        if not observed or observed[-1].output_sample_index != expected[-1][0]:
            raise AssertionError("project Qt page did not finish the continuous prefix")
        probability_error = float(np.max(np.abs(observed[-1].probabilities - expected[-1][1])))
        if probability_error > 2e-5:
            raise AssertionError("protocol-to-UI prediction differs from direct runtime")
        result = {
            "status": "reconstructed_protocol_to_project_ui_verified",
            "source_s04_sha256": source_sha,
            "model_sha256": runtime.sha256,
            "nominal_emg_rate_hz": nominal_rate,
            "recorded_measured_emg_rate_hz": measured_rate,
            "channels": channels,
            "reconstructed_emg_packets": parser.stats.emg_frames,
            "decoded_channel_rows_equal_saved_raw": True,
            "controller_samples": window.acquisition.clock.total,
            "direct_runtime_frames": len(expected),
            "ui_prediction_batches": len(observed),
            "latest_output_sample_index": int(observed[-1].output_sample_index),
            "max_latest_probability_error": probability_error,
            "physical_usb_tested": False,
            "boundary": "Saved EMG values were re-encoded as protocol packets, parsed and passed through the project's AcquisitionController and MainWindow signal wiring into the Song worker. This proves the software path and channel order for these values, not original packet-byte identity, a connected device, wall-clock pacing, channel placement on a new donning, gesture accuracy or measured screen latency.",
        }
    finally:
        window.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "status", "reconstructed_emg_packets", "latest_output_sample_index",
        "max_latest_probability_error")}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--collection-root", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    verify(args.source, args.collection_root, args.bundle, args.output)
