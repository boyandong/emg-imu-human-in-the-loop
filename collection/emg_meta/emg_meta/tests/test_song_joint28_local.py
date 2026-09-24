from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import butter, iirnotch, sosfilt, tf2sos

from emgforce.inference.song_joint28_local import SongJoint28Stream, SongJoint28WindowRuntime


BUNDLE = Path(__file__).resolve().parents[1] / "model_assets/song_joint28_window"


def test_tracked_bundle_predicts_finite_28_state_probabilities():
    runtime = SongJoint28WindowRuntime(BUNDLE)
    rng = np.random.default_rng(20260925)
    emg = rng.normal(size=(50, 8)).astype(np.float32)
    imu = rng.normal(size=(22, 6)).astype(np.float32)
    hand, arm, joint = runtime.predict_filtered_window(emg, imu)
    assert (hand.shape, arm.shape, joint.shape) == ((4,), (7,), (28,))
    assert np.isfinite(joint).all()
    np.testing.assert_allclose(joint.sum(), 1, atol=1e-12)
    for i, (a, h) in enumerate(runtime.joint_indices):
        np.testing.assert_allclose(joint[i], arm[a] * hand[h], atol=1e-12)
    with pytest.raises(ValueError, match="50×8"):
        runtime.predict_filtered_window(emg[:-1], imu)
    with pytest.raises(ValueError, match="22×6"):
        runtime.predict_filtered_window(emg, imu[:-1])


def test_bundle_rejects_tampered_model_and_invalid_label_order(tmp_path):
    directory = tmp_path / "bundle"
    shutil.copytree(BUNDLE, directory)
    artifact = directory / "song_joint28_model.json"
    manifest_path = directory / "song_joint28_manifest.json"
    with artifact.open("a", encoding="utf-8") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        SongJoint28WindowRuntime(directory)
    model = json.loads(artifact.read_text(encoding="utf-8"))
    model["joint_indices"][0] = model["joint_indices"][1]
    artifact.write_text(json.dumps(model), encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="label mapping"):
        SongJoint28WindowRuntime(directory)


def test_indexed_stream_matches_offline_windows_across_packet_chunking():
    model = SongJoint28WindowRuntime(BUNDLE)
    rng = np.random.default_rng(18)
    raw = rng.integers(-300, 300, size=(600, 8), dtype=np.int32)
    imu_indices = np.rint(np.arange(270) * 250 / 112).astype(np.int64)
    accel = rng.normal(size=(len(imu_indices), 3)).astype(np.float32)
    gyro = rng.normal(size=(len(imu_indices), 3)).astype(np.float32)

    def replay(split_emg: bool):
        stream = SongJoint28Stream(model)
        result = []
        imu_offset = 0
        for start in range(0, len(raw), 25):
            end = start + 25
            if split_emg:
                for left in range(start, end, 7):
                    right = min(left + 7, end)
                    gap, frames = stream.ingest_emg(raw[left:right], np.arange(left, right))
                    assert not gap
                    result.extend(frames)
            else:
                gap, frames = stream.ingest_emg(raw[start:end], np.arange(start, end))
                assert not gap
                result.extend(frames)
            imu_end = int(np.searchsorted(imu_indices, end, side="right"))
            result.extend(stream.ingest_imu(accel[imu_offset:imu_end], gyro[imu_offset:imu_end],
                                            imu_indices[imu_offset:imu_end]))
            imu_offset = imu_end
        assert stream.dropped_frames == 0
        return result

    whole, chunked = replay(False), replay(True)
    assert [index for index, _ in whole] == [index for index, _ in chunked]
    assert len(whole) == 23
    np.testing.assert_allclose([value for _, value in whole],
                               [value for _, value in chunked], atol=1e-7)
    filtered = raw.astype(np.float64)
    filtered = sosfilt(butter(4, 40, btype="highpass", fs=250, output="sos"), filtered, axis=0)
    for frequency in (50, 100):
        b, a = iirnotch(frequency, Q=30, fs=250)
        filtered = sosfilt(tf2sos(b, a), filtered, axis=0)
    filtered = filtered.astype(np.float32)
    for end, joint in whole:
        last = np.searchsorted(imu_indices, end, side="right")
        native_imu = np.column_stack((accel[last - 22:last], gyro[last - 22:last]))
        expected = model.predict_filtered_window(filtered[end - 49:end + 1], native_imu)[2]
        np.testing.assert_allclose(joint, expected, atol=1e-6)


def test_collection_page_loads_opt_in_joint_model_and_consumes_both_sensors(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from emgforce.ui.realtime_inference_page import RealtimeInferencePage

    app = QApplication.instance() or QApplication([])
    page = RealtimeInferencePage(tmp_path / "models")
    try:
        page.refresh_models()
        key = f"song28::{BUNDLE.resolve()}"
        index = page.model_combo.findData(key)
        assert index >= 0
        page.model_combo.setCurrentIndex(index)
        page.set_connected(True)
        page.load_selected_model()
        deadline = time.monotonic() + 5
        while page.bundle is None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.01)
        assert page.bundle is not None
        assert page.bundle.output_channels == 28
        assert page.start_button.isEnabled() and not page.calibrate_button.isEnabled()
        frames = []
        page.worker.prediction_ready.connect(frames.append)
        page.start_recognition()
        rng = np.random.default_rng(19)
        page.ingest_emg(rng.integers(-200, 200, size=(250, 8), dtype=np.int32),
                        np.arange(250, dtype=np.int64))
        imu_indices = np.rint(np.arange(113) * 250 / 112).astype(np.int64)
        page.ingest_imu(rng.normal(size=(113, 3)).astype(np.float32),
                        rng.normal(size=(113, 3)).astype(np.float32),
                        np.arange(113, dtype=np.int64), imu_indices)
        deadline = time.monotonic() + 5
        while not frames and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.01)
        assert frames and frames[-1].probabilities.shape == (28,)
        assert "数据龄未测" in page.current_probability.text()
    finally:
        assert page.shutdown()
        page.close()


def test_main_window_controller_routes_batched_imu_into_joint_worker(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from emgforce.protocol import Packet
    from emgforce.ui import MainWindow

    app = QApplication.instance() or QApplication([])
    (tmp_path / "protocols").mkdir()
    window = MainWindow(tmp_path)
    page = window.realtime_inference_page
    try:
        page.refresh_models()
        page.model_combo.setCurrentIndex(page.model_combo.findData(f"song28::{BUNDLE.resolve()}"))
        page.set_connected(True)
        page.load_selected_model()
        deadline = time.monotonic() + 5
        while page.bundle is None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.01)
        assert page.bundle is not None
        frames = []
        page.worker.prediction_ready.connect(frames.append)
        page.start_recognition()
        for batch in range(50):
            packets = [Packet("EMG", (batch * 7 + offset) % 256,
                              (batch * 5 + offset) * 4_000_000, b"",
                              emg_uv=tuple(100 + offset for _ in range(8)))
                       for offset in range(5)]
            packets += [Packet("IMU", (batch * 7 + 5 + offset) % 256,
                               batch * 20_000_000 + offset * 8_000_000, b"",
                               gyro_rad_s=(.1, .2, .3), accel_m_s2=(.0, .0, 9.81))
                        for offset in range(2)]
            window.acquisition.ingest_packets(packets)
        deadline = time.monotonic() + 5
        while not frames and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.01)
        assert frames and frames[-1].output_sample_index >= 74
        assert frames[-1].probabilities.shape == (28,)
    finally:
        assert window.close()
