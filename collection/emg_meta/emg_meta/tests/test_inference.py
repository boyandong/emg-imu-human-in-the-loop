from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from emgforce.inference.engine import (
    DetectedEvent, InferenceConfig, OnlineGestureStateMachine,
    RealtimeGestureEngine, detect_threshold_events,
)
from emgforce.inference.model_bundle import discover_model_bundles, load_model_bundle
from emgforce.inference.preprocessing import (
    estimate_realtime_scale, fixed_lag_model_window,
)
from emgforce.inference.worker import (
    RealtimeInferenceWorker, _evaluate_offline_predictions,
    _targets_from_prompts,
)
from emgforce.processing.training_preprocessing import preprocess_training_emg


LABELS = (
    "index_press", "index_release", "middle_press", "middle_release",
    "thumb_click", "thumb_down", "thumb_in", "thumb_out", "thumb_up",
)


def _write_bundle(root: Path) -> None:
    root.mkdir()
    artifact = root / "model.ckpt"
    artifact.write_bytes(b"test-artifact")
    (root / "labels.json").write_text(json.dumps({
        "model_outputs": LABELS,
        "display_names": {name: name for name in LABELS},
    }), encoding="utf-8")
    (root / "preprocessing.json").write_text("{}", encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps({
        "model_id": "unit-test-model",
        "artifact": {
            "filename": "model.ckpt",
            "format": "pytorch_lightning",
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        },
        "network": {
            "input_channels": 8, "output_channels": 9, "sample_rate_hz": 2000,
        },
        "labels_file": "labels.json",
        "preprocessing_file": "preprocessing.json",
    }), encoding="utf-8")


def test_model_bundle_manifest_and_hash_are_validated(tmp_path) -> None:
    root = tmp_path / "model"
    _write_bundle(root)
    bundle = load_model_bundle(root)
    assert bundle.model_id == "unit-test-model"
    assert bundle.algorithm_id == "meta_conv_lstm_v1"
    assert bundle.runtime_backend == "lightning_conv_lstm"
    assert bundle.labels == LABELS
    assert discover_model_bundles(tmp_path)[0].root == root.resolve()
    (root / "model.ckpt").write_bytes(b"changed")
    try:
        load_model_bundle(root)
    except ValueError as exc:
        assert "SHA-256" in str(exc)
    else:
        raise AssertionError("corrupted model artifact was accepted")


def test_realtime_scale_and_fixed_lag_match_training_preprocessing() -> None:
    rng = np.random.default_rng(7)
    raw = rng.normal(0, 1200, size=(9000, 8)).astype(np.int32)
    training = preprocess_training_emg(raw, 2000.0)
    assert np.isclose(estimate_realtime_scale(raw, 2000.0),
                      training.scale_counts_per_unit)
    live = fixed_lag_model_window(
        raw, 2000.0, training.scale_counts_per_unit,
        model_window_samples=4000, lag_samples=500,
    )
    np.testing.assert_allclose(live, training.signal[-4500:-500], atol=1e-6)


def test_threshold_crossings_are_debounced() -> None:
    probabilities = np.zeros((9, 4), dtype=np.float32)
    probabilities[4] = [0.1, 0.4, 0.2, 0.5]
    events, previous, last, last_name = detect_threshold_events(
        probabilities, np.array([0, 10, 20, 30]), LABELS,
        {"thumb_click": "拇指轻点"}, threshold=0.35,
        debounce_samples=50,
    )
    assert [(event.name, event.sample_index) for event in events] == [
        ("thumb_click", 10)]
    assert events[0].display_name == "拇指轻点"
    assert previous[4] == probabilities[4, -1]
    assert last == 10
    assert last_name == "thumb_click"


def test_meta_debounce_keeps_release_after_press_across_chunks() -> None:
    first = np.zeros((9, 1), dtype=np.float32)
    first[0, 0] = 0.8
    events, previous, last, last_name = detect_threshold_events(
        first, np.array([100]), LABELS, {}, 0.5, debounce_samples=100)
    assert [event.name for event in events] == ["index_press"]
    second = np.zeros((9, 1), dtype=np.float32)
    second[1, 0] = 0.9
    events, _, last, last_name = detect_threshold_events(
        second, np.array([120]), LABELS, {}, 0.5, previous,
        debounce_samples=100, last_event_index=last,
        last_event_name=last_name)
    assert [event.name for event in events] == ["index_release"]


def test_meta_debounce_suppresses_nonrelease_after_release() -> None:
    first = np.zeros((9, 1), dtype=np.float32)
    first[1, 0] = 0.8
    events, previous, last, last_name = detect_threshold_events(
        first, np.array([100]), LABELS, {}, 0.5, debounce_samples=100)
    assert [event.name for event in events] == ["index_release"]

    second = np.zeros((9, 1), dtype=np.float32)
    second[4, 0] = 0.9
    events, _, _, _ = detect_threshold_events(
        second, np.array([120]), LABELS, {}, 0.5, previous,
        debounce_samples=100, last_event_index=last,
        last_event_name=last_name)
    assert events == []


def test_same_frame_crossings_choose_highest_probability() -> None:
    probabilities = np.zeros((9, 1), dtype=np.float32)
    probabilities[0, 0] = 0.60
    probabilities[2, 0] = 0.85
    events, _, _, _ = detect_threshold_events(
        probabilities, np.array([500]), LABELS, {}, 0.5,
        debounce_samples=100)
    assert [(event.name, event.probability) for event in events] == [
        ("middle_press", probabilities[2, 0])]


def test_same_frame_probability_tie_has_stable_label_order() -> None:
    probabilities = np.zeros((9, 1), dtype=np.float32)
    probabilities[0, 0] = probabilities[2, 0] = 0.8
    events, _, _, _ = detect_threshold_events(
        probabilities, np.array([500]), LABELS, {}, 0.5,
        debounce_samples=100)
    assert [event.name for event in events] == ["index_press"]


def test_official_cler_debounce_suppresses_nonrelease_after_release() -> None:
    from generic_neuromotor_interface.cler import debounce_events

    assert debounce_events([
        ("index_release", 1.000),
        ("thumb_click", 1.020),
    ], debounce=0.05) == [("index_release", 1.000)]


def test_online_state_machine_filters_release_and_scores_holds() -> None:
    machine = OnlineGestureStateMachine(sample_rate=2000)
    def event(name: str, sample: int) -> DetectedEvent:
        return DetectedEvent(name, name, sample, 0.8)

    assert machine.filter([event("index_release", 10)]) == []
    assert [item.name for item in machine.filter([event("index_press", 100)])] == [
        "index_press"]
    completed = machine.filter([event("index_release", 1200)])
    assert completed[0].hold_valid is True
    assert np.isclose(completed[0].hold_duration_seconds, 0.55)

    machine.filter([event("middle_press", 2000)])
    interrupted = machine.filter([event("thumb_click", 2200)])
    assert [item.name for item in interrupted] == [
        "middle_release", "thumb_click"]
    assert interrupted[0].synthetic is True


class _DummyModel:
    def __init__(self) -> None:
        self.bundle = SimpleNamespace(
            labels=LABELS,
            display_names={name: name for name in LABELS},
            input_channels=8,
            sample_rate=2000,
        )

    def predict(self, emg: np.ndarray) -> np.ndarray:
        length = (len(emg) - 21) // 10 + 1
        output = np.zeros((9, length), dtype=np.float32)
        output[4, -1] = 0.8
        return output


def test_realtime_engine_produces_prediction_after_buffer_is_ready() -> None:
    config = InferenceConfig(
        buffer_seconds=0.20,
        model_window_seconds=0.05,
        fixed_lag_seconds=0.01,
    )
    engine = RealtimeGestureEngine(_DummyModel(), config)
    engine.set_scale(100.0)
    rng = np.random.default_rng(8)
    raw = rng.normal(0, 1000, size=(400, 8)).astype(np.int32)
    indices = np.arange(400, dtype=np.int64)
    assert engine.ingest(raw, indices) is False
    assert engine.ready_for_prediction
    frame = engine.predict()
    assert frame.probabilities.shape == (9,)
    assert np.isclose(frame.probabilities[4], 0.8)
    assert frame.events[0].name == "thumb_click"
    assert frame.output_sample_index <= indices[-1]
    expected_age = (indices[-1] - frame.output_sample_index) * 1000 / config.sample_rate
    assert np.isclose(frame.output_age_ms, expected_age)


def test_engine_resets_stream_on_sample_gap() -> None:
    engine = RealtimeGestureEngine(_DummyModel())
    engine.set_scale(1.0)
    first = np.zeros((20, 8), dtype=np.int32)
    assert engine.ingest(first, np.arange(20)) is False
    assert engine.ingest(first, np.arange(25, 45)) is True
    assert not engine.ready_for_prediction


def test_realtime_worker_never_drops_emg_when_inference_is_backlogged(tmp_path) -> None:
    worker = RealtimeInferenceWorker(tmp_path)
    for start in range(300):
        raw = np.full((1, 8), start, dtype=np.int32)
        worker.submit_emg(raw, np.array([start], dtype=np.int64))

    queued = [worker._commands.get_nowait() for _ in range(300)]
    assert worker._commands.empty()
    assert [int(item[2][0]) for item in queued] == list(range(300))


def test_offline_targets_use_nearest_output_frame() -> None:
    import pandas as pd

    times = np.array([10.00, 10.02, 10.04, 10.06])
    prompts = pd.DataFrame({
        "name": ["thumb_click", "index_press", "unknown"],
        "time": [10.019, 10.059, 10.04],
    })
    targets = _targets_from_prompts(times, LABELS, prompts)
    assert targets[1, LABELS.index("thumb_click")] == 1.0
    assert targets[3, LABELS.index("index_press")] == 1.0
    assert targets.sum() == 2.0


def test_offline_threshold_scan_reports_global_and_per_class_recommendations() -> None:
    import pandas as pd

    times = np.arange(200, dtype=np.float64) / 50.0
    probabilities = np.zeros((9, len(times)), dtype=np.float32)
    probabilities[LABELS.index("thumb_click"), 50] = 0.30
    probabilities[LABELS.index("thumb_click"), 10] = 0.20  # false crossing
    probabilities[LABELS.index("thumb_down"), 100] = 0.45
    prompts = pd.DataFrame({
        "name": ["thumb_click", "thumb_down"],
        "time": [times[50], times[100]],
    })

    result = _evaluate_offline_predictions(
        probabilities, times, LABELS, prompts,
        sample_rate=2000, stride=40, threshold=0.35,
        include_cler=False,
    )

    assert np.isclose(result["mean_fnr"], 0.5)
    assert len(result["threshold_sweep"]) == 9
    assert np.isclose(result["recommended_threshold"], 0.30)
    assert set(result["recommended_class_thresholds"]) == set(LABELS)
