from __future__ import annotations

import numpy as np

import torch

from emgforce.processing.template_alignment import (
    _estimate_templates, _global_recenter_offsets, align_session_templates,
    mpf_features,
)
from emgforce.processing.training_preprocessing import filter_training_emg
from mpf_tds.features import MPFConfig, MultiBandMatrixPowerFeatures


def test_alignment_mpf_reuses_official_equivalent_feature_extractor() -> None:
    rng = np.random.default_rng(31)
    raw = rng.normal(size=(4000, 8)).astype(np.float32)
    actual, frame_samples = mpf_features(raw, 2000.0)
    filtered, _ = filter_training_emg(raw, 2000.0)
    config = MPFConfig()
    with torch.inference_mode():
        expected = MultiBandMatrixPowerFeatures(config)(
            torch.from_numpy(filtered.astype(np.float32).T[None]))[0].numpy()
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    np.testing.assert_array_equal(
        frame_samples,
        config.left_context_samples
        + np.arange(len(expected)) * config.output_stride_samples)


def test_rerp_estimator_disentangles_overlapping_templates() -> None:
    offsets = np.arange(-2, 3)
    first = np.asarray([0.0, 1.0, 3.0, 1.0, 0.0])[:, None]
    second = np.asarray([0.0, -2.0, 1.0, 2.0, 0.0])[:, None]
    names = ["first", "second"] * 3
    # Vary the inter-event interval so the two response bases are identifiable.
    centers = np.asarray([20, 22, 45, 48, 70, 71])
    features = np.full((100, 1), 7.0)
    for name, center in zip(names, centers):
        features[center + offsets] += first if name == "first" else second

    estimated = _estimate_templates(features, names, centers, offsets)

    np.testing.assert_allclose(estimated["first"], first, atol=3e-3)
    np.testing.assert_allclose(estimated["second"], second, atol=3e-3)


def test_global_recentering_correlates_only_along_time_axis() -> None:
    reference = np.zeros((11, 3), dtype=np.float32)
    reference[4:7] = np.asarray([[1, 2, 1], [3, 1, -2], [1, -1, 2]])
    session = np.zeros_like(reference)
    session[6:9] = reference[4:7]

    offsets = _global_recenter_offsets({"thumb_up": session}, {"thumb_up": reference})

    assert offsets == {"thumb_up": 2}


def test_session_template_alignment_recovers_variable_reaction_delays() -> None:
    sample_rate = 2000.0
    duration = 24.0
    rng = np.random.default_rng(4)
    raw = rng.normal(0.0, 20.0, (int(sample_rate * duration), 8))
    cue_dtype = np.dtype([
        ("name", "S32"), ("trial_id", "i8"), ("stage_id", "i8"),
        ("sample_index", "i8"),
    ])
    trial_dtype = np.dtype([("trial_id", "i8"), ("valid", "?")])
    cue_rows = []
    trial_rows = []
    true_event_times = []
    delays = (0.28, 0.42, 0.61, 0.35, 0.55)
    for index, cue_time in enumerate(np.arange(1.0, 23.0, 2.0)):
        name = "thumb_up" if index % 2 == 0 else "thumb_down"
        event_time = cue_time + delays[index % len(delays)]
        true_event_times.append(event_time)
        relative = np.arange(-0.18, 0.181, 1.0 / sample_rate)
        frequency = 211.0 if name == "thumb_up" else 287.0
        waveform = np.sin(2 * np.pi * frequency * relative) \
            * np.exp(-(relative / 0.065) ** 2) * 600.0
        spatial = np.asarray([1, .7, .3, -.2, -.4, .2, .6, .9]) \
            if name == "thumb_up" else np.asarray([-.2, .4, .9, .5, .1, -.5, -.7, .3])
        start = int((event_time - 0.18) * sample_rate)
        raw[start:start + len(waveform)] += waveform[:, None] * spatial
        cue_rows.append((name.encode(), index + 1, 1, int(cue_time * sample_rate)))
        trial_rows.append((index + 1, True))

    result = align_session_templates(
        raw, np.asarray(cue_rows, dtype=cue_dtype),
        np.asarray(trial_rows, dtype=trial_dtype), sample_rate)

    errors_ms = np.asarray([
        event.aligned_sample_index / sample_rate - true_time
        for event, true_time in zip(result.events, true_event_times)
    ]) * 1000.0
    assert result.converged
    # Session-only forced alignment has one unresolved common time origin until
    # a project/global template is available.  Relative event timing should be
    # recovered even when all events share that harmless common offset.
    for name in ("thumb_up", "thumb_down"):
        class_errors = errors_ms[np.asarray([
            event.name == name for event in result.events])]
        assert np.median(np.abs(class_errors - np.median(class_errors))) <= 20.1
    assert all(event.included for event in result.events)
    assert result.feature_timeseries.shape[1] == 192
    for event_index, (event, scores, cue_row) in enumerate(
            zip(result.events, result.match_scores, cue_rows)):
        peak_offset = result.match_offsets_ms[
            event_index, int(np.nanargmax(scores))]
        actual_offset = (event.aligned_sample_index - cue_row[3]) / sample_rate * 1000.0
        assert abs(actual_offset - peak_offset) <= 0.01
