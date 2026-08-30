from __future__ import annotations

import h5py
import numpy as np
import pandas as pd

from emgforce.experiment.events import EventType
from emgforce.experiment.models import CueEvent, ExperimentEvent, TrialInfo
from emgforce.processing.meta_alignment import export_meta_aligned
from emgforce.processing.meta_alignment import update_alignment_review
from emgforce.processing.alignment_review import AlignmentReviewStore
from emgforce.storage.hdf5_reader import Hdf5SessionReader
from emgforce.storage.hdf5_recorder import Hdf5Recorder


def test_hdf5_raw_emg_imu_events_trials_round_trip(tmp_path) -> None:
    path = tmp_path / "session.h5"
    recorder = Hdf5Recorder(batch_samples=3)
    recorder.start(path, {"participant_id": "P001", "session_id": "S01",
        "sampling_rate": 2000, "num_emg_channels": 8, "protocol_name": "tiny",
        "protocol_json": "{}"})
    raw = np.arange(40, dtype=np.int32).reshape(5, 8) - 20
    recorder.enqueue_emg(raw[:2], np.arange(2), np.array([10, 11], np.uint8))
    recorder.enqueue_emg(raw[2:], np.arange(2, 5), np.array([12, 13, 14], np.uint8))
    recorder.enqueue_imu(np.array([[1, 2, 3]], np.float32), np.array([[4, 5, 6]], np.float32),
        np.array([123], np.int64), np.array([15], np.uint8), np.array([4], np.int64))
    recorder.enqueue_event(ExperimentEvent(0, 2, .001, 123, EventType.PROMPT_START,
                                           "click", 1, 1, 1, "提示"))
    recorder.enqueue_trial(TrialInfo(1, "click", 1, 1, 0, 0, 2, 4, 5, True))
    recorder.stop()
    with h5py.File(path, "r") as h5:
        np.testing.assert_array_equal(h5["streams/emg/raw"][:], raw)
        assert h5["streams/emg/raw"].dtype == np.dtype("int32")
        assert h5["streams/imu/gyro"].shape == (1, 3)
        assert h5["events"][0]["note"].decode() == "提示"
        assert h5["trials"].shape == (1,)
        assert h5["meta"].attrs["schema_version"] == "2.1"
    reader = Hdf5SessionReader(path)
    assert reader.validate() == []
    assert reader.summary()["emg_samples"] == 5
    segment, indices = reader.trial_emg(1, padding_sec=0)
    np.testing.assert_array_equal(segment, raw)
    np.testing.assert_array_equal(indices, np.arange(5))


def test_raw_cues_are_preserved_and_meta_export_uses_aligned_times(tmp_path) -> None:
    path = tmp_path / "session.h5"
    target = tmp_path / "session_meta_aligned.hdf5"
    recorder = Hdf5Recorder(batch_samples=200)
    recorder.start(path, {
        "participant_id": "P001", "session_id": "S01", "sampling_rate": 2000,
        "num_emg_channels": 8, "protocol_name": "meta", "start_unix_time": 1000.0,
    })
    raw = np.zeros((8000, 8), np.int32)
    carrier = np.where(np.arange(4000) % 2, 1200, -1200).astype(np.int32)
    raw[2300:6300] = carrier[:, None]
    recorder.enqueue_emg(raw, np.arange(len(raw)),
                         np.arange(len(raw), dtype=np.uint8))
    recorder.enqueue_event(ExperimentEvent(
        0, 0, 0.0, 1, EventType.STAGE_START, note="default"))
    recorder.enqueue_cue_event(CueEvent("index_press", 2000, 1.0, 1, 1))
    # Defensive fixture: even if a legacy collector wrote a timed null cue,
    # Meta export must not align it or turn it into a tenth prompt class.
    recorder.enqueue_cue_event(CueEvent("null_finger_snap", 4000, 2.0, 2, 2))
    recorder.enqueue_cue_event(CueEvent("index_release", 6000, 3.0, 1, 1))
    recorder.enqueue_trial(TrialInfo(
        1, "index_hold", 1, 1, 0, 0, 2000, 6800, 7000, True))
    recorder.enqueue_event(ExperimentEvent(
        1, 8000, 4.0, 2, EventType.STAGE_END))
    recorder.stop()
    with h5py.File(path, "r") as h5:
        assert "prompts" not in h5 and "data" not in h5
        assert h5["meta"].attrs["label_alignment_status"] == "raw_cues_only"
        assert [value.decode() for value in h5["cue_events"]["name"]] == [
            "index_press", "null_finger_snap", "index_release"]
        assert h5["cue_events"].dtype.names[-2:] == (
            "scheduled_monotonic_ns", "emitted_monotonic_ns")
    export_meta_aligned(path, target)
    prompts = pd.read_hdf(target, "prompts")
    stages = pd.read_hdf(target, "stages")
    assert prompts["name"].tolist() == ["index_press", "index_release"]
    offsets = (prompts["time"].to_numpy() - 1000.0) * 2000 - np.array([2000, 6000])
    np.testing.assert_allclose(offsets, [300, 300], atol=80)
    assert stages.to_dict("records") == [{
        "start": 1000.0, "end": 1004.0, "name": "default"}]
    with h5py.File(target, "r") as h5:
        assert h5["data"].dtype.names == ("emg", "time")
        assert h5["data"].shape == (8000,)
        assert h5["meta"].attrs["training_label_shift_applied"] == np.False_
        assert h5["alignment_events"]["included"].tolist() == [True, True]
        assert [value.rstrip(b"\x00") for value in h5["alignment_reviews"]["status"]] == [
            b"auto_accepted", b"auto_accepted"]

    review = AlignmentReviewStore(path, target)
    update_alignment_review(target, 1, status="rejected", note="check pair integrity")
    review.refresh()
    assert pd.read_hdf(target, "prompts").empty
    review.accept(1, "release accepted")
    assert pd.read_hdf(target, "prompts")["name"].tolist() == [
        "index_press", "index_release"]
    original = review.record(0)["reviewed_sample_index"]
    review.adjust_ms(0, 10, "visual onset correction")
    adjusted = review.record(0)
    assert adjusted["reviewed_sample_index"] == original + 20
    assert adjusted["status"] == "manually_adjusted"
    assert adjusted["note"] == "visual onset correction"
    prompts = pd.read_hdf(target, "prompts")
    np.testing.assert_allclose(
        prompts.iloc[0]["time"], 1000 + (original + 20) / 2000)
    assert review.reject_trial(0, "bad hold") == 2
    assert [review.record(i)["status"] for i in range(2)] == ["rejected", "rejected"]
    assert pd.read_hdf(target, "prompts").empty
