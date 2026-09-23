"""Qt worker for the experimental causal Song 8-channel local model."""
from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QThread, Signal

from .engine import DetectedEvent, PredictionFrame
from .song_local import DISPLAY, LABELS, SongLocalRuntime, SongOnlineDecision


class SongRealtimeWorker(QThread):
    model_loaded = Signal(object)
    status_changed = Signal(str)
    failed = Signal(str)
    calibration_progress = Signal(int, int)
    calibration_finished = Signal(float)
    prediction_ready = Signal(object)

    def __init__(self, directory: Path, parent=None):
        super().__init__(parent)
        self.directory = Path(directory)
        self._commands: queue.Queue[tuple[str, object, object]] = queue.Queue()
        self._stopping = threading.Event()
        self._threshold = 0.5

    def submit_emg(self, raw: np.ndarray, indices: np.ndarray):
        self._commands.put_nowait(("emg", np.asarray(raw, dtype=np.int32).copy(),
                                   np.asarray(indices, dtype=np.int64).copy()))

    def notify_packet_loss(self, lost: int):
        if lost > 0:
            self._commands.put_nowait(("gap", int(lost), None))

    def begin_calibration(self, seconds: float):
        self._commands.put_nowait(("calibrate", float(seconds), None))

    def begin_recognition(self):
        self._commands.put_nowait(("recognize", None, None))

    def pause_recognition(self):
        self._commands.put_nowait(("pause", None, None))

    def set_threshold(self, value: float):
        self._commands.put_nowait(("threshold", float(value), None))

    def stop(self):
        self._stopping.set()
        self._commands.put_nowait(("stop", None, None))

    def run(self):
        try:
            runtime = SongLocalRuntime(self.directory)
            self.model_loaded.emit(runtime.make_bundle())
            self.status_changed.emit("Song 8 通道实验模型已校验；连接设备后可直接开始识别")
            mode = "idle"
            pending: tuple[str, object, object] | None = None
            decision = SongOnlineDecision()
            while not self._stopping.is_set():
                if pending is None:
                    try:
                        command, value, extra = self._commands.get(timeout=0.05)
                    except queue.Empty:
                        continue
                else:
                    command, value, extra = pending
                    pending = None
                if command == "stop":
                    break
                if command == "threshold":
                    self._threshold = float(value)
                    continue
                if command == "gap":
                    runtime.reset()
                    decision.reset()
                    self.status_changed.emit(f"检测到 {value} 帧丢失；Song 滤波状态已重置")
                    continue
                if command == "calibrate":
                    self.status_changed.emit("该模型使用源数据训练，当前无需静息校准；可直接开始识别")
                    continue
                if command == "recognize":
                    runtime.reset()
                    decision.reset()
                    mode = "recognizing"
                    self.status_changed.emit("Song 四分类因果推理运行中；连续动作/延迟尚未验证")
                    continue
                if command == "pause":
                    mode = "idle"
                    runtime.reset()
                    decision.reset()
                    self.status_changed.emit("Song 实时识别已暂停")
                    continue
                if command != "emg" or mode != "recognizing":
                    continue
                chunks = [(value, extra)]
                while True:
                    try:
                        next_item = self._commands.get_nowait()
                    except queue.Empty:
                        break
                    if next_item[0] != "emg":
                        pending = next_item
                        break
                    chunks.append((next_item[1], next_item[2]))
                started = time.perf_counter()
                events = []
                latest = None
                had_gap = False
                for raw, indices in chunks:
                    gap, frames = runtime.ingest(raw, indices)
                    had_gap |= gap
                    if gap:
                        decision.reset()
                        # A gap may occur inside this chunk; omit any pre-gap
                        # predictions rather than mixing two stream epochs.
                        frames = []
                    for sample_index, probabilities in frames:
                        latest = (sample_index, probabilities)
                        name, changed = decision.step(probabilities, self._threshold)
                        if changed and name is not None:
                            events.append(DetectedEvent(
                                name=name, display_name=DISPLAY[name], sample_index=sample_index,
                                probability=float(probabilities[LABELS.index(name)])))
                if had_gap:
                    self.status_changed.emit("检测到采样不连续；Song 滤波状态和 200 ms 窗口已重置")
                if latest is not None:
                    sample_index, probabilities = latest
                    self.prediction_ready.emit(PredictionFrame(
                        probabilities=probabilities, labels=LABELS, events=tuple(events),
                        output_sample_index=sample_index, output_age_ms=0.0,
                        fixed_lag_ms=0.0, inference_ms=(time.perf_counter() - started) * 1000.0,
                        scale_counts_per_unit=1.0, active_label=decision.active_label))
        except Exception as exc:
            self.failed.emit(str(exc))
