from __future__ import annotations

import h5py
import numpy as np

from emgforce.experiment.events import EventType
from emgforce.experiment.models import ExperimentEvent
from emgforce.storage.hdf5_recorder import Hdf5Recorder


def test_event_order_is_preserved_by_writer_queue(tmp_path) -> None:
    path = tmp_path / "session.h5"
    recorder = Hdf5Recorder(batch_samples=2)
    recorder.start(path, {"participant_id": "P1", "session_id": "S1",
                          "sampling_rate": 2000, "num_emg_channels": 8,
                          "protocol_name": "test"})
    recorder.enqueue_emg(np.zeros((2, 8), np.int32), np.arange(2), np.array([1, 2], np.uint8))
    for event_id, kind in enumerate((EventType.SESSION_START, EventType.PROMPT_START,
                                     EventType.PROMPT_END, EventType.SESSION_END)):
        recorder.enqueue_event(ExperimentEvent(event_id, min(event_id, 2),
            min(event_id, 2) / 2000, 100 + event_id, kind))
    recorder.stop()
    with h5py.File(path, "r") as h5:
        assert h5["events"]["event_id"].tolist() == [0, 1, 2, 3]
        assert [x.decode() for x in h5["events"]["event_type"]] == [
            "SESSION_START", "PROMPT_START", "PROMPT_END", "SESSION_END"]

