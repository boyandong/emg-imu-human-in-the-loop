import csv
import json

import h5py
import numpy as np
import pytest

from emgforce.inference.engine import PredictionFrame
from emgforce.inference.live_diagnostic import LiveDiagnosticRecorder
from emgforce.controller import AcquisitionController
from emgforce.protocol import Packet


def test_live_capture_preserves_raw_indices_imu_and_displayed_probabilities(tmp_path):
    capture = LiveDiagnosticRecorder(
        tmp_path, model_id="song-test", model_sha256="a" * 64,
        labels=("fist", "neutral"), sample_rate_hz=250,
        threshold=0.5, hand="right")
    capture.record_emg(np.arange(24, dtype=np.int32).reshape(3, 8), np.array([8, 9, 10]),
                       np.array([80, 90, 100]))
    capture.record_emg(np.full((2, 8), 4, dtype=np.int32), np.array([12, 13]))
    capture.record_imu(np.ones((2, 3)), np.zeros((2, 3)), np.array([100, 200]))
    capture.record_packet_loss(2)
    capture.record_annotation("neutral", "start", 13)
    capture.record_annotation("neutral", "end", 13)
    capture.record_prediction(PredictionFrame(
        probabilities=np.array([0.2, 0.8]), labels=("fist", "neutral"), events=(),
        output_sample_index=13, output_age_ms=0, fixed_lag_ms=0,
        inference_ms=3.5, scale_counts_per_unit=1.0, active_label="neutral"), 0.55)
    with pytest.raises(ValueError, match="indices must increase"):
        capture.record_emg(np.zeros((1, 8)), np.array([13]))
    directory = capture.close()
    assert capture.close() == directory
    with h5py.File(directory / "signals.h5") as handle:
        assert handle["emg/raw"].shape == (5, 8)
        np.testing.assert_array_equal(handle["emg/sample_index"][:], [8, 9, 10, 12, 13])
        np.testing.assert_array_equal(handle["emg/received_ns"][:], [80, 90, 100, -1, -1])
        assert handle["imu/gyro_rad_s"].shape == (2, 3)
        np.testing.assert_array_equal(handle["imu/received_ns"][:], [100, 200])
    with (directory / "predictions.csv").open(newline="", encoding="utf-8") as handle:
        predictions = list(csv.DictReader(handle))
    assert len(predictions) == 1
    assert predictions[0]["active_label"] == "neutral"
    assert predictions[0]["output_sample_index"] == "13"
    assert float(predictions[0]["p_neutral"]) == pytest.approx(0.8)
    with (directory / "manual_annotations.csv").open(newline="", encoding="utf-8") as handle:
        annotations = list(csv.DictReader(handle))
    assert [(row["action"], row["event"], row["latest_emg_sample_index"])
            for row in annotations] == [("neutral", "start", "13"), ("neutral", "end", "13")]
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "closed"
    assert (manifest["raw_samples"], manifest["imu_samples"], manifest["predictions"],
            manifest["reported_lost_packets"]) == (5, 2, 1, 2)
    assert manifest["ground_truth_available"] is False
    assert manifest["manual_annotations"] == 2


def test_live_inference_fanout_keeps_original_receive_timestamps():
    controller = AcquisitionController()
    captured = []
    controller.emg_inference_ready.connect(lambda raw, indices, stamps:
                                           captured.append((raw.copy(), indices.copy(), stamps.copy())))
    controller.ingest_packets([
        Packet("EMG", 1, 123_000, b"", emg_uv=(1,) * 8),
        Packet("EMG", 2, 127_000, b"", emg_uv=(2,) * 8),
    ])
    assert len(captured) == 1
    raw, indices, received = captured[0]
    np.testing.assert_array_equal(raw[:, 0], [1, 2])
    np.testing.assert_array_equal(indices, [0, 1])
    np.testing.assert_array_equal(received, [123_000, 127_000])


def test_imu_inference_fanout_uses_global_emg_boundary():
    controller = AcquisitionController()
    captured = []
    controller.imu_inference_ready.connect(
        lambda gyro, accel, received, indices: captured.append(
            (gyro.copy(), accel.copy(), received.copy(), indices.copy())))
    controller.ingest_packets([
        Packet("EMG", 1, 123_000, b"", emg_uv=(1,) * 8),
        Packet("EMG", 2, 127_000, b"", emg_uv=(2,) * 8),
        Packet("IMU", 3, 129_000, b"", gyro_rad_s=(.1, .2, .3),
               accel_m_s2=(1., 2., 3.)),
    ])
    assert len(captured) == 1
    gyro, accel, received, indices = captured[0]
    np.testing.assert_allclose(gyro, [[.1, .2, .3]])
    np.testing.assert_allclose(accel, [[1., 2., 3.]])
    np.testing.assert_array_equal(received, [129_000])
    np.testing.assert_array_equal(indices, [2])
