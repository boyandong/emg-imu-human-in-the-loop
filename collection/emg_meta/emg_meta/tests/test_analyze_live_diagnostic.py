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
    run.record_annotation("open_hand", "start", 10)
    for index, probs, active in ((20, (0.8, 0.2), "neutral"),
                                 (40, (0.4, 0.6), "open_hand"),
                                 (80, (0.7, 0.3), "neutral")):
        run.record_prediction(PredictionFrame(
            probabilities=np.array(probs), labels=("neutral", "open_hand"),
            events=(), output_sample_index=index, output_age_ms=0,
            fixed_lag_ms=0, inference_ms=2, scale_counts_per_unit=1,
            active_label=active), 0.5)
    run.record_annotation("open_hand", "end", 90)
    directory = run.close()
    summary = analyze(directory)
    assert summary["peak_label_counts"] == {"neutral": 2, "open_hand": 1}
    assert summary["manual_intervals"] == 1
    assert summary["file_hashes_verified"] is True
    assert summary["raw_emg_samples_verified"] == 101
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
