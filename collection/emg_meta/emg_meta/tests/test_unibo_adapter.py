from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from emgforce.inference.unibo_adapter import (
    UniBoAdapterRuntime, discover_unibo_models,
)


REPOSITORY = Path(__file__).resolve().parents[4]


def test_discovers_tracked_unibo_models_with_e5_as_default() -> None:
    models = discover_unibo_models(REPOSITORY)

    assert [item.experiment for item in models] == ["E5", "E0", "E6b"]
    assert all(item.artifact.is_file() for item in models)
    assert models[0].validation_accuracy == pytest.approx(0.8127046813540127)
    assert models[2].posture_required is True


def test_rejects_duplicate_ring_channel_mapping() -> None:
    e5 = discover_unibo_models(REPOSITORY)[0]

    with pytest.raises(ValueError, match="四个不同"):
        UniBoAdapterRuntime(e5.artifact, (0, 0, 2, 3))


def test_real_e5_artifact_calibrates_and_predicts_four_probabilities() -> None:
    e5 = discover_unibo_models(REPOSITORY)[0]
    runtime = UniBoAdapterRuntime(e5.artifact, (0, 2, 4, 6))
    random = np.random.default_rng(20260913)
    neutral = random.normal(0.0, 500.0, size=(8 * 250, 8)).astype(np.int32)

    gains = runtime.calibrate_neutral(neutral)
    probabilities = runtime.predict(
        random.normal(0.0, 1500.0, size=(runtime.context_samples, 8)).astype(np.int32))

    assert gains.shape == (4,)
    assert np.all(np.isfinite(gains))
    assert np.all(gains > 0)
    assert probabilities.shape == (4,)
    assert np.all(probabilities >= 0)
    assert float(probabilities.sum()) == pytest.approx(1.0, abs=1e-6)
    assert runtime.make_bundle().preprocessing["online_event_threshold"] == 0.50


def test_bipolar_device_signal_is_converted_to_smooth_positive_envelope() -> None:
    e5 = discover_unibo_models(REPOSITORY)[0]
    runtime = UniBoAdapterRuntime(e5.artifact, (0, 2, 4, 6))
    random = np.random.default_rng(7)
    raw = random.normal(0.0, 1000.0, size=(8 * 250, 8)).astype(np.int32)
    runtime.channel_center = np.median(raw[:, runtime.channel_map], axis=0)

    envelope = runtime._envelope(raw)[-runtime.raw_window_samples:]
    rms = np.sqrt(np.mean(envelope * envelope, axis=0))

    assert np.all(envelope >= 0)
    assert np.all(np.mean(envelope, axis=0) / rms > 0.95)
    assert np.all(np.std(envelope, axis=0) / rms < 0.32)
