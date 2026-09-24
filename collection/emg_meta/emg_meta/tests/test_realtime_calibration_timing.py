from __future__ import annotations

import json
import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from emgforce.ui.realtime_inference_page import RealtimeInferencePage


class FakeWorker:
    def __init__(self) -> None:
        self.requests: list[int] = []
        self.paused = False

    def begin_calibration(self, seconds: int) -> None:
        self.requests.append(seconds)

    def pause_recognition(self) -> None:
        self.paused = True


def test_calibration_records_click_to_completion_separately_from_requested_signal(tmp_path, monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    page = RealtimeInferencePage(tmp_path / "models")
    worker = FakeWorker()
    page.worker = worker
    page.bundle = SimpleNamespace(model_id="example", sha256="abc", sample_rate=250, metadata={})
    page._connected = True
    page.calibration_seconds.setValue(5)
    ticks = iter((100.0, 107.5))
    monkeypatch.setattr("emgforce.ui.realtime_inference_page.time.monotonic", lambda: next(ticks))

    page.start_calibration()
    page.start_calibration()
    page._calibration_finished(1.25)

    assert worker.requests == [5]
    records = [json.loads(line) for line in (tmp_path / "models" / "calibration_timings.jsonl").read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["requested_signal_seconds"] == 5
    assert records[0]["click_to_outcome_seconds"] == 7.5
    assert records[0]["outcome"] == "completed"
    assert records[0]["scale"] == 1.25
    assert "7.5 秒" in page.calibration_detail.text()
    page.close()
    app.processEvents()


def test_disconnect_records_interrupted_calibration(tmp_path, monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    page = RealtimeInferencePage(tmp_path / "models")
    worker = FakeWorker()
    page.worker = worker
    page.bundle = SimpleNamespace(model_id="example", sha256="abc", sample_rate=250, metadata={})
    page._connected = True
    ticks = iter((10.0, 12.0))
    monkeypatch.setattr("emgforce.ui.realtime_inference_page.time.monotonic", lambda: next(ticks))

    page.start_calibration()
    page.set_connected(False)
    page._calibration_finished(1.0)

    lines = (tmp_path / "models" / "calibration_timings.jsonl").read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["outcome"] == "device_disconnected"
    assert record["click_to_outcome_seconds"] == 2.0
    assert worker.paused
    page.close()
    app.processEvents()
