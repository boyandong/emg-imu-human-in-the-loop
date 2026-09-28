"""Exercise fault boundaries and committed synthetic-quality replay."""
import json

import numpy as np
import pytest

from benchmarks.new_bank_v2.envelope_quality_analysis import build as verify_envelope
from benchmarks.new_bank_v2.roam_quality_paired import build as verify_pairs
from benchmarks.new_bank_v2.roam_quality_run import ROOT, fault


def test_synthetic_faults_are_test_only_copies_with_fixed_channel_semantics():
    windows = np.ones((2, 40, 8), dtype=np.float32)
    original = windows.copy()
    q95 = np.full(8, 0.75, dtype=np.float32)
    rms = np.full(8, 2.0, dtype=np.float32)
    np.testing.assert_array_equal(fault(windows, "clean", q95, rms), original)
    dropped = fault(windows, "dropout_ch3", q95, rms)
    assert not dropped[:, :, 3].any()
    np.testing.assert_array_equal(dropped[:, :, [0, 1, 2, 4, 5, 6, 7]],
                                  original[:, :, [0, 1, 2, 4, 5, 6, 7]])
    np.testing.assert_array_equal(fault(windows, "gain_half_all", q95, rms), original * 0.5)
    assert np.max(fault(windows, "clip_q95_all", q95, rms)) == pytest.approx(0.75)
    assert not np.array_equal(fault(windows, "line50_half_rms_all", q95, rms), original)
    np.testing.assert_array_equal(windows, original)
    with pytest.raises(ValueError, match="unknown"):
        fault(windows, "real_device_fault", q95, rms)


def test_roam_quality_saved_evidence_replays_without_external_archive():
    result = json.loads((ROOT / "ROAM_QUALITY_RESULTS.json").read_text(encoding="utf-8"))
    protocol = result["protocol"]
    assert len(protocol["conditions"]) == 14
    assert result["source_rest_windows"] == 1828
    assert len(result["validation_trial_ids"]) == len(result["final_trial_ids"]) == 45
    assert result["clean_source_model_replay"].startswith("all 360")
    verify_pairs(verify=True)
    verify_envelope(verify=True)
