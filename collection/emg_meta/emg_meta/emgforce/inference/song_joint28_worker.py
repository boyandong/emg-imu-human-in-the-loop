"""Opt-in Qt worker for the exploratory Song 8ch EMG × 6-axis IMU model."""
from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QThread, Signal

from .engine import DetectedEvent, PredictionFrame
from .song_joint28_local import SongJoint28Stream, SongJoint28WindowRuntime


class SongJoint28RealtimeWorker(QThread):
    model_loaded = Signal(object)
    status_changed = Signal(str)
    failed = Signal(str)
    calibration_progress = Signal(int, int)
    calibration_finished = Signal(float)
    prediction_ready = Signal(object)

    def __init__(self, directory: Path, parent=None):
        super().__init__(parent)
        self.directory = Path(directory)
        self._commands: queue.Queue[tuple[str, object, object, object]] = queue.Queue()
        self._stopping = threading.Event()
        self._threshold = .15

    def submit_emg(self, raw: np.ndarray, indices: np.ndarray):
        self._commands.put_nowait(("emg", np.asarray(raw, dtype=np.int32).copy(),
                                   np.asarray(indices, dtype=np.int64).copy(), None))

    def submit_imu(self, accel: np.ndarray, gyro: np.ndarray, emg_indices: np.ndarray):
        self._commands.put_nowait(("imu", np.asarray(accel, dtype=np.float32).copy(),
                                   np.asarray(gyro, dtype=np.float32).copy(),
                                   np.asarray(emg_indices, dtype=np.int64).copy()))

    def notify_packet_loss(self, lost: int):
        if lost > 0:
            self._commands.put_nowait(("gap", int(lost), None, None))

    def begin_calibration(self, seconds: float):
        self._commands.put_nowait(("calibrate", float(seconds), None, None))

    def begin_recognition(self):
        self._commands.put_nowait(("recognize", None, None, None))

    def pause_recognition(self):
        self._commands.put_nowait(("pause", None, None, None))

    def set_threshold(self, value: float):
        self._commands.put_nowait(("threshold", float(value), None, None))

    def stop(self):
        self._stopping.set()
        self._commands.put_nowait(("stop", None, None, None))

    def run(self):
        try:
            model = SongJoint28WindowRuntime(self.directory)
            stream = SongJoint28Stream(model)
            bundle = model.make_bundle()
            self.model_loaded.emit(bundle)
            self.status_changed.emit("Song 28 类实验模型已校验；需要同时接收 EMG 与 IMU")
            mode = "idle"
            candidate = None
            count = 0
            active = None
            while not self._stopping.is_set():
                try:
                    command, first, second, third = self._commands.get(timeout=.05)
                except queue.Empty:
                    continue
                if command == "stop":
                    break
                if command == "threshold":
                    self._threshold = float(first)
                    continue
                if command in {"gap", "recognize", "pause"}:
                    stream.reset()
                    candidate = active = None
                    count = 0
                    if command == "gap":
                        self.status_changed.emit(f"检测到 {first} 帧丢失；双传感器窗口已重置")
                    else:
                        mode = "recognizing" if command == "recognize" else "idle"
                        self.status_changed.emit("Song 28 类实时实验推理中；准确率和端到端延迟未验证"
                                                 if mode == "recognizing" else "Song 28 类推理已暂停")
                    continue
                if command == "calibrate":
                    self.status_changed.emit("该源模型未启用现场校准，可直接开始识别")
                    continue
                if mode != "recognizing":
                    continue
                started = time.perf_counter()
                if command == "emg":
                    gap, frames = stream.ingest_emg(first, second)
                    if gap:
                        candidate = active = None
                        count = 0
                        self.status_changed.emit("EMG 索引中断；EMG/IMU 历史和滤波状态已重置")
                elif command == "imu":
                    frames = stream.ingest_imu(first, second, third)
                else:
                    continue
                events = []
                latest = None
                for sample_index, probabilities in frames:
                    latest = (sample_index, probabilities)
                    peak = int(np.argmax(probabilities))
                    name = model.joint_classes[peak] if probabilities[peak] >= self._threshold else None
                    if name == candidate:
                        count += 1
                    else:
                        candidate, count = name, 1
                    if count >= 3 and name != active:
                        active = name
                        if name is not None:
                            events.append(DetectedEvent(
                                name=name, display_name=bundle.display_names[name],
                                sample_index=sample_index, probability=float(probabilities[peak])))
                if latest is not None:
                    sample_index, probabilities = latest
                    self.prediction_ready.emit(PredictionFrame(
                        probabilities=probabilities, labels=model.joint_classes,
                        events=tuple(events), output_sample_index=sample_index,
                        output_age_ms=0.0, fixed_lag_ms=0.0,
                        inference_ms=(time.perf_counter() - started) * 1000.0,
                        scale_counts_per_unit=1.0, active_label=active))
        except Exception as exc:
            self.failed.emit(str(exc))
