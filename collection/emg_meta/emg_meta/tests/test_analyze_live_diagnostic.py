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
    assert summary["annotated_action_summary"]["open_hand"]["mean_peak_agreement"] == pytest.approx(1 / 3)
    assert summary["annotated_action_summary"]["open_hand"]["mean_display_agreement"] == pytest.approx(1 / 3)
    assert (directory / "analysis.json").is_file()
    assert "not formal recognition accuracy" in summary["scope"]
