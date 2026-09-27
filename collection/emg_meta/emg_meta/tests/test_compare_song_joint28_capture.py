from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from emgforce.inference.compare_song_joint28_capture import run as compare, verify
from emgforce.inference.live_diagnostic import LiveDiagnosticRecorder
from emgforce.inference.song_joint28_local import SongJoint28WindowRuntime
from emgforce.inference.compare_song_joint28_capture import BASELINE


def _capture(tmp_path, *, indexed_imu: bool = True):
    model = SongJoint28WindowRuntime(BASELINE)
    run = LiveDiagnosticRecorder(tmp_path, model_id="original", model_sha256=model.sha256,
                                 labels=model.joint_classes, sample_rate_hz=250,
                                 threshold=.15, hand="right")
    rng = np.random.default_rng(20260928)
    raw = rng.integers(-200, 200, size=(1100, 8), dtype=np.int32)
    run.record_emg(raw, np.arange(len(raw)), np.arange(len(raw), dtype=np.int64) * 4_000_000)
    imu_index = np.rint(np.arange(500) * 250 / 112).astype(np.int64)
    if not indexed_imu:
        imu_index[:] = -1
    run.record_imu(rng.normal(size=(500, 3)), rng.normal(size=(500, 3)),
                   np.arange(500, dtype=np.int64) * 9_000_000, imu_index)
    run.record_annotation("left_open_hand", "start", 200)
    run.record_annotation("left_open_hand", "end", 950)
    return run.close()


def test_paired_replay_uses_one_capture_and_conservative_manual_core(tmp_path):
    directory = _capture(tmp_path)
    result = compare(directory)
    assert result["emitted_common_frames"] > 20
    assert result["scored_manual_stable_core_intervals"] == 1
    assert result["frame_scores"]["baseline"]["joint"]["n"] > 0
    assert result["frame_scores"]["baseline"]["joint"]["n"] == result["frame_scores"]["signed"]["joint"]["n"]
    assert result["trial_scores"]["baseline"]["n"] == result["trial_scores"]["signed"]["n"] == 1
    assert result["intervals"][0]["stable_core_margin_seconds_each_end"] == .4
    assert result["model_sha256"]["baseline"] != result["model_sha256"]["signed"]
    assert "not formal device accuracy" in result["scope"]
    with (directory / "paired_song28_analysis.json").open(encoding="utf-8") as stream:
        saved = json.load(stream)
    assert saved["emitted_common_frames"] == result["emitted_common_frames"]
    assert verify(directory)["status"] == "verified"
    with (directory / "paired_song28_frames.csv").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="frame hash mismatch"):
        verify(directory)


def test_paired_replay_rejects_missing_alignment_and_tampered_source(tmp_path):
    unaligned = _capture(tmp_path / "unaligned", indexed_imu=False)
    with pytest.raises(ValueError, match="indexed EMG/IMU"):
        compare(unaligned)
    valid = _capture(tmp_path / "tampered")
    with h5py.File(valid / "signals.h5", "r+") as handle:
        handle["emg/raw"][0, 0] += 1
    with pytest.raises(ValueError, match="hash mismatch"):
        compare(valid)
