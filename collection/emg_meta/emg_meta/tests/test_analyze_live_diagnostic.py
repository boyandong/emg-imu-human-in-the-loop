import csv
import hashlib
import json

import h5py
import numpy as np
import pytest

from emgforce.inference.analyze_live_diagnostic import analyze
from emgforce.inference.engine import PredictionFrame
from emgforce.inference.live_diagnostic import LiveDiagnosticRecorder


def test_manual_interval_analysis_counts_neutral_bias_without_claiming_truth(tmp_path):
    run = LiveDiagnosticRecorder(tmp_path, model_id="m", model_sha256="b" * 64,
                                 labels=("neutral", "open_hand"), sample_rate_hz=250,
                                 threshold=0.5, hand="right")
    run.record_emg(np.zeros((101, 8), dtype=np.int32), np.arange(101))
    run.record_imu(np.zeros((2, 3)), np.ones((2, 3)), np.array([10, 20]),
                   np.array([15, 30]))
    run.record_annotation("open_hand", "start", 10)
    for index, probs, active in ((20, (0.8, 0.2), "neutral"),
                                 (40, (0.4, 0.6), "open_hand"),
                                 (80, (0.7, 0.3), "neutral")):
        run.record_prediction(PredictionFrame(
            probabilities=np.array(probs), labels=("neutral", "open_hand"),
            events=(), output_sample_index=index, output_age_ms=0,
            fixed_lag_ms=0, inference_ms=2, scale_counts_per_unit=1,
            active_label=active), 0.5, host_receive_to_ui_ms=8.5)
    run.record_annotation("open_hand", "end", 90)
    directory = run.close()
    summary = analyze(directory)
    assert summary["peak_label_counts"] == {"neutral": 2, "open_hand": 1}
    assert summary["manual_intervals"] == 1
    assert summary["file_hashes_verified"] is True
    assert summary["raw_emg_samples_verified"] == 101
    assert summary["imu_samples_with_emg_boundary"] == 2
    assert summary["imu_boundaries_outside_captured_emg"] == 0
    assert summary["host_receive_to_ui_callback_ms"]["frames"] == 3
    assert summary["host_receive_to_ui_callback_ms"]["median"] == pytest.approx(8.5)
    assert summary["raw_signal_profile"]["zero_fraction_per_channel"] == [1.0] * 8
    assert summary["reported_lost_packets"] == 0
    assert summary["prediction_frames_outside_captured_raw"] == 0
    assert summary["annotated_action_summary"]["open_hand"]["mean_peak_agreement"] == pytest.approx(1 / 3)
    assert summary["annotated_action_summary"]["open_hand"]["mean_display_agreement"] == pytest.approx(1 / 3)
    assert summary["event_intervals_with_predictions"] == 1
    assert summary["event_intervals_ever_display_match"] == 1
    assert summary["event_intervals_endpoint_display_match"] == 0
    assert summary["intervals"][0]["first_display_match_after_start_seconds"] == pytest.approx(0.12)
    assert (directory / "analysis.json").is_file()
    assert "not formal recognition accuracy" in summary["scope"]
    with (directory / "predictions.csv").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        analyze(directory)


def test_event_summary_exposes_missed_action_and_pre_start_active_state(tmp_path):
    run = LiveDiagnosticRecorder(tmp_path, model_id="m", model_sha256="d" * 64,
                                 labels=("neutral", "open_hand"), sample_rate_hz=250,
                                 threshold=0.5, hand="right")
    run.record_emg(np.zeros((300, 8), dtype=np.int32), np.arange(300))

    def prediction(index, active):
        probability = np.array((0.8, 0.2) if active == "neutral" else (0.2, 0.8))
        run.record_prediction(PredictionFrame(
            probabilities=probability, labels=("neutral", "open_hand"), events=(),
            output_sample_index=index, output_age_ms=0, fixed_lag_ms=0,
            inference_ms=2, scale_counts_per_unit=1, active_label=active), 0.5)

    prediction(100, "open_hand")
    run.record_annotation("open_hand", "start", 150)
    prediction(160, "neutral")
    prediction(200, "neutral")
    run.record_annotation("open_hand", "end", 250)
    summary = analyze(run.close())
    assert summary["event_intervals_with_predictions"] == 1
    assert summary["event_intervals_ever_display_match"] == 0
    assert summary["event_intervals_endpoint_display_match"] == 0
    assert summary["pre_start_intervals_with_frames"] == 1
    assert summary["pre_start_intervals_any_active"] == 1
    assert summary["intervals"][0]["first_display_match_after_start_seconds"] is None


def test_signal_profile_flags_flat_and_near_limit_channels_descriptively(tmp_path):
    run = LiveDiagnosticRecorder(tmp_path, model_id="m", model_sha256="c" * 64,
                                 labels=("neutral", "open_hand"), sample_rate_hz=250,
                                 threshold=0.5, hand="right")
    raw = np.tile(np.arange(500, dtype=np.int32)[:, None], (1, 8))
    raw[:, 0] = 0
    raw[10:20, 2] = 8_300_000
    run.record_emg(raw, np.arange(len(raw)), np.arange(len(raw), dtype=np.int64) * 4_000_000)
    run.record_packet_loss(3)
    result = analyze(run.close())
    profile = result["raw_signal_profile"]
    assert profile["complete_one_second_windows"] == 2
    assert profile["flat_one_second_windows_per_channel"][0] == 2
    assert profile["flat_one_second_windows_per_channel"][1] == 0
    assert profile["near_adc_limit_fraction_per_channel"][2] == pytest.approx(10 / 500)
    assert profile["received_rate_hz_approx"] == pytest.approx(250.0)
    assert result["reported_lost_packets"] == 3
    assert result["sample_index_gap_edges"] == 0


def test_existing_v1_diagnostic_capture_remains_readable(tmp_path):
    run = LiveDiagnosticRecorder(tmp_path, model_id="old", model_sha256="e" * 64,
                                 labels=("neutral", "open_hand"), sample_rate_hz=250,
                                 threshold=.5, hand="right")
    run.record_emg(np.ones((50, 8), dtype=np.int32), np.arange(50))
    run.record_imu(np.ones((1, 3)), np.ones((1, 3)), np.array([100]))
    run.record_prediction(PredictionFrame(
        probabilities=np.array([.8, .2]), labels=("neutral", "open_hand"),
        events=(), output_sample_index=49, output_age_ms=0, fixed_lag_ms=0,
        inference_ms=1, scale_counts_per_unit=1, active_label="neutral"), .5)
    directory = run.close()
    with h5py.File(directory / "signals.h5", "r+") as handle:
        del handle["imu/emg_sample_index"]
    path = directory / "predictions.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    fields = [key for key in rows[0] if key != "host_receive_to_ui_ms"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in fields} for row in rows])
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema"] = "emgforce_live_diagnostic_v1"
    for name in ("signals.h5", "predictions.csv"):
        manifest["file_sha256"][name] = hashlib.sha256((directory / name).read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = analyze(directory)
    assert result["prediction_frames"] == 1
    assert result["imu_samples_with_emg_boundary"] == 0
    assert result["host_receive_to_ui_callback_ms"]["frames"] == 0


def test_joint28_still_neutral_is_rest_for_prestart_activity(tmp_path):
    run = LiveDiagnosticRecorder(tmp_path, model_id="joint28", model_sha256="f" * 64,
                                 labels=("still_neutral", "up_neutral"), sample_rate_hz=250,
                                 threshold=.5, hand="right")
    run.record_emg(np.ones((100, 8), dtype=np.int32), np.arange(100))
    run.record_prediction(PredictionFrame(
        probabilities=np.array([.9, .1]), labels=("still_neutral", "up_neutral"),
        events=(), output_sample_index=10, output_age_ms=0, fixed_lag_ms=0,
        inference_ms=1, scale_counts_per_unit=1, active_label="still_neutral"), .5)
    run.record_annotation("up_neutral", "start", 20)
    run.record_prediction(PredictionFrame(
        probabilities=np.array([.1, .9]), labels=("still_neutral", "up_neutral"),
        events=(), output_sample_index=50, output_age_ms=0, fixed_lag_ms=0,
        inference_ms=1, scale_counts_per_unit=1, active_label="up_neutral"), .5)
    run.record_annotation("up_neutral", "end", 80)
    result = analyze(run.close())
    assert result["pre_start_intervals_with_frames"] == 1
    assert result["pre_start_intervals_any_active"] == 0


def test_joint28_diagnostic_separates_hand_from_arm_errors(tmp_path):
    labels = tuple(f"{arm}_{hand}" for arm in
                   ("still", "up", "down", "left", "right", "forward", "backward")
                   for hand in ("neutral", "index_pinch", "fist", "open_hand"))
    run = LiveDiagnosticRecorder(tmp_path, model_id="joint28", model_sha256="a" * 64,
                                 labels=labels, sample_rate_hz=250,
                                 threshold=.15, hand="right")
    run.record_emg(np.ones((100, 8), dtype=np.int32), np.arange(100))
    run.record_annotation("left_open_hand", "start", 10)

    def prediction(index: int, name: str):
        values = np.full(len(labels), .001)
        values[labels.index(name)] = 1 - (len(labels) - 1) * .001
        run.record_prediction(PredictionFrame(
            probabilities=values, labels=labels, events=(), output_sample_index=index,
            output_age_ms=0, fixed_lag_ms=0, inference_ms=1,
            scale_counts_per_unit=1, active_label="right_open_hand"), .15)

    prediction(25, "right_open_hand")
    prediction(50, "left_open_hand")
    run.record_annotation("left_open_hand", "end", 75)
    result = analyze(run.close())
    interval = result["intervals"][0]
    assert interval["peak_agreement_fraction"] == .5
    assert interval["hand_peak_agreement_fraction"] == 1
    assert interval["arm_peak_agreement_fraction"] == .5
    assert interval["hand_display_agreement_fraction"] == 1
    assert interval["arm_display_agreement_fraction"] == 0
    assert result["factorized_28_state_summary"]["by_hand"]["open_hand"][
        "mean_peak_agreement"] == 1
    assert result["factorized_28_state_summary"]["by_arm"]["left"][
        "mean_peak_agreement"] == .5
