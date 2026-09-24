from __future__ import annotations

import hashlib
import json
import os
import shutil
import time

import numpy as np
import pytest

from emgforce.inference.song_local import SongLocalRuntime, SongOnlineDecision, discover_song_models


def test_online_decision_requires_three_confident_frames_and_resets_on_gap():
    decision = SongOnlineDecision()
    open_hand = np.array([0.02, 0.02, 0.06, 0.90])
    neutral = np.array([0.03, 0.02, 0.92, 0.03])
    for _ in range(2):
        assert decision.step(open_hand, 0.5) == (None, False)
    assert decision.step(open_hand, 0.5) == ("open_hand", True)
    assert decision.step(neutral, 0.5) == ("open_hand", False)
    assert decision.step(open_hand, 0.5) == ("open_hand", False)
    assert decision.step(neutral, 0.5) == ("open_hand", False)
    assert decision.step(neutral, 0.5) == ("open_hand", False)
    assert decision.step(neutral, 0.5) == ("neutral", True)
    decision.reset()
    assert decision.step(open_hand, 0.5) == (None, False)
    with pytest.raises(ValueError, match="threshold"):
        decision.step(open_hand, 1.5)


def _bundle(tmp_path):
    directory = tmp_path / "models" / "song"
    directory.mkdir(parents=True)
    model = {"format_version": 1, "model_kind": "song_real8_causal_f0_logistic",
             "sample_rate_hz": 250, "channels": 8, "window_samples": 50, "hop_samples": 25,
             "filter": {"highpass_hz": 40.0, "highpass_order": 4,
                        "notches_hz": [50.0, 100.0], "notch_q": 30.0,
                        "implementation": "causal_sosfilt_zero_initial_state_continuous"},
             "classes": ["fist", "index_pinch", "neutral", "open_hand"],
             "f0_thresholds": [3.0] * 8,
             "standard_scaler_mean": [0.0] * 48,
             "standard_scaler_scale": [1.0] * 48,
             "logistic_coef": np.random.default_rng(7).normal(0, 0.01, size=(4, 48)).tolist(),
             "logistic_intercept": [0.0] * 4}
    artifact = directory / "song_f0_model.json"
    artifact.write_text(json.dumps(model), encoding="utf-8")
    manifest = {"format_version": 1, "algorithm_id": "song_real8_local_v1",
                "model_id": "test Song", "artifact": artifact.name,
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "model_status": "exploratory_one_person_one_day_not_formal_frozen",
                "validation_trial_accuracy": 0.7, "validation_trial_macro_f1": 0.6}
    (directory / "song_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory


def test_streaming_chunks_match_whole_stream(tmp_path):
    directory = _bundle(tmp_path)
    assert len(discover_song_models(directory.parent)) == 1
    raw = np.random.default_rng(8).integers(-3000, 3000, size=(1000, 8), dtype=np.int32)
    indices = np.arange(1000, dtype=np.int64)
    whole = SongLocalRuntime(directory)
    gap, reference = whole.ingest(raw, indices)
    assert not gap and len(reference) == 39
    chunked = SongLocalRuntime(directory)
    received = []
    for start in range(0, 1000, 17):
        gap, frames = chunked.ingest(raw[start:start + 17], indices[start:start + 17])
        assert not gap
        received.extend(frames)
    assert [index for index, _ in received] == [index for index, _ in reference]
    np.testing.assert_allclose([p for _, p in received], [p for _, p in reference], atol=1e-6, rtol=1e-6)
    assert all(abs(float(probabilities.sum()) - 1) < 1e-6 for _, probabilities in received)


def test_gap_resets_filter_and_window(tmp_path):
    runtime = SongLocalRuntime(_bundle(tmp_path))
    raw = np.ones((100, 8), dtype=np.int32)
    assert runtime.ingest(raw, np.arange(100))[0] is False
    gap, frames = runtime.ingest(raw[:49], np.arange(150, 199))
    assert gap and not frames
    gap, frames = runtime.ingest(raw[:1], np.array([199]))
    assert not gap and len(frames) == 1 and frames[0][0] == 199


def test_hash_mismatch_is_rejected(tmp_path):
    directory = _bundle(tmp_path)
    with (directory / "song_f0_model.json").open("a", encoding="utf-8") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="SHA-256"):
        SongLocalRuntime(directory)


def test_spd_bundle_validates_reference_and_outputs_four_probabilities(tmp_path):
    directory = _bundle(tmp_path)
    artifact = directory / "song_f0_model.json"
    manifest_path = directory / "song_manifest.json"
    model = json.loads(artifact.read_text(encoding="utf-8"))
    model["model_kind"] = "song_real8_causal_f0_spd_logistic"
    model["spd_shrinkage"] = 0.05
    model["spd_reference"] = np.eye(8).tolist()
    model["standard_scaler_mean"] += [0.0] * 36
    model["standard_scaler_scale"] += [1.0] * 36
    model["logistic_coef"] = np.random.default_rng(11).normal(0, 0.01, (4, 84)).tolist()

    def save(updated):
        artifact.write_text(json.dumps(updated), encoding="utf-8")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    save(model)
    runtime = SongLocalRuntime(directory)
    assert runtime.with_spd and runtime.coef.shape == (4, 84)
    probability = runtime.predict_filtered_window(np.random.default_rng(12).normal(size=(50, 8)))
    assert probability.shape == (4,) and np.isclose(probability.sum(), 1.0)
    model["spd_reference"][0][0] = -1.0
    save(model)
    with pytest.raises(ValueError, match="positive definite"):
        SongLocalRuntime(directory)


def test_song_only_page_prefers_valid_spd_but_keeps_manual_selection(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from emgforce.ui.realtime_inference_page import RealtimeInferencePage

    app = QApplication.instance() or QApplication([])
    base = _bundle(tmp_path)
    spd = base.parent / "song_real8_f0_spd"
    shutil.copytree(base, spd)
    artifact = spd / "song_f0_model.json"
    model = json.loads(artifact.read_text(encoding="utf-8"))
    model["model_kind"] = "song_real8_causal_f0_spd_logistic"
    model["spd_shrinkage"] = 0.05
    model["spd_reference"] = np.eye(8).tolist()
    model["standard_scaler_mean"] += [0.0] * 36
    model["standard_scaler_scale"] += [1.0] * 36
    model["logistic_coef"] = np.zeros((4, 84)).tolist()
    artifact.write_text(json.dumps(model), encoding="utf-8")
    manifest_path = spd / "song_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model_id"] = "test Song SPD"
    manifest["sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    page = RealtimeInferencePage(base.parent, prefer_song_spd=True)
    try:
        page.refresh_models()
        assert page.model_combo.currentData() == f"song::{spd}"
        page.model_combo.setCurrentIndex(page.model_combo.findData(f"song::{base}"))
        page.refresh_models()
        assert page.model_combo.currentData() == f"song::{base}"
        app.processEvents()
    finally:
        assert page.shutdown()
        page.close()


def test_realtime_page_loads_song_and_emits_probability_without_device(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from emgforce.inference.engine import PredictionFrame
    from emgforce.ui.realtime_inference_page import RealtimeInferencePage

    app = QApplication.instance() or QApplication([])
    directory = _bundle(tmp_path)
    page = RealtimeInferencePage(directory.parent)
    try:
        page.refresh_models()
        index = page.model_combo.findData(f"song::{directory}")
        assert index >= 0
        page.model_combo.setCurrentIndex(index)
        page.set_connected(True)
        page.load_selected_model()
        deadline = time.monotonic() + 5
        while page.bundle is None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        assert page.bundle is not None
        assert page.start_button.isEnabled()
        assert not page.calibrate_button.isEnabled()
        assert page.start_diagnostic_button.isEnabled()
        page.start_diagnostic()
        assert page._diagnostic is not None
        diagnostic_directory = page._diagnostic.directory
        predictions = []
        page.worker.prediction_ready.connect(predictions.append)
        page.start_recognition()
        raw = np.random.default_rng(13).integers(-2000, 2000, size=(250, 8), dtype=np.int32)
        page.ingest_emg(raw[:100], np.arange(100, dtype=np.int64))
        page.annotation_action.setCurrentIndex(page.annotation_action.findData("open_hand"))
        page._mark_diagnostic_action("start")
        page.ingest_emg(raw[100:], np.arange(100, 250, dtype=np.int64))
        page._mark_diagnostic_action("end")
        deadline = time.monotonic() + 5
        while not predictions and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        assert predictions and predictions[-1].probabilities.shape == (4,)
        assert "数据龄未测" in page.current_probability.text()
        steady = PredictionFrame(
            probabilities=np.array([0.02, 0.03, 0.05, 0.90]),
            labels=("fist", "index_pinch", "neutral", "open_hand"), events=(),
            output_sample_index=250, output_age_ms=0, fixed_lag_ms=0,
            inference_ms=1, scale_counts_per_unit=1, active_label="open_hand")
        page._prediction_ready(steady)
        assert page.current_gesture.text() == "张开手"
        assert not page._gesture_reset_timer.isActive()
        page._prediction_ready(PredictionFrame(
            probabilities=np.array([0.2, 0.3, 0.3, 0.2]),
            labels=steady.labels, events=(), output_sample_index=275,
            output_age_ms=0, fixed_lag_ms=0, inference_ms=1,
            scale_counts_per_unit=1, active_label=None))
        assert page.current_gesture.text() == "等待识别"
        page.stop_diagnostic()
        assert (diagnostic_directory / "signals.h5").is_file()
        assert (diagnostic_directory / "predictions.csv").is_file()
        assert (diagnostic_directory / "manual_annotations.csv").is_file()
        assert (diagnostic_directory / "analysis.json").is_file()
        assert "已保存" in page.diagnostic_status.text()
    finally:
        assert page.shutdown()
        page.close()
