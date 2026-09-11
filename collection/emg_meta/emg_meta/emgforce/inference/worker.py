from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PySide6.QtCore import QThread, Signal

from emgforce.algorithms import META_CONV_LSTM
from mpf_tds.metrics import evaluate_events, evaluate_paper_fnr, search_thresholds

from .engine import (
    InferenceConfig, OnlineGestureStateMachine, RealtimeGestureEngine,
    detect_threshold_events,
)
from .model_bundle import create_gesture_model, load_model_bundle


class RealtimeInferenceWorker(QThread):
    model_loaded = Signal(object)
    status_changed = Signal(str)
    failed = Signal(str)
    calibration_progress = Signal(int, int)
    calibration_finished = Signal(float)
    prediction_ready = Signal(object)

    def __init__(self, bundle_root: Path, parent=None) -> None:
        super().__init__(parent)
        self.bundle_root = Path(bundle_root)
        # EMG must remain lossless here.  A bounded command queue used to drop
        # the oldest EMG chunk whenever inference briefly fell behind, which
        # created a sample-index gap and forced the realtime engine to clear
        # the 2.25 s model context.  The run loop drains consecutive EMG
        # commands as one batch, so a transient backlog is caught up without
        # running the model once per queued chunk.
        self._commands: queue.Queue[tuple[str, object, object]] = queue.Queue()
        self._stopping = threading.Event()
        self._mode = "idle"
        self._threshold: float | np.ndarray = 0.50

    def submit_emg(self, raw: np.ndarray, indices: np.ndarray) -> None:
        item = ("emg", np.asarray(raw, dtype=np.int32).copy(),
                np.asarray(indices, dtype=np.int64).copy())
        self._commands.put_nowait(item)

    def begin_calibration(self, seconds: float) -> None:
        self._put_control("calibrate", float(seconds))

    def begin_recognition(self) -> None:
        self._put_control("recognize", None)

    def pause_recognition(self) -> None:
        self._put_control("pause", None)

    def set_threshold(self, value: float) -> None:
        self._put_control("threshold", float(value))

    def stop(self) -> None:
        self._stopping.set()
        self._put_control("stop", None)

    def _put_control(self, command: str, value: object) -> None:
        self._commands.put_nowait((command, value, None))

    def run(self) -> None:
        try:
            bundle = load_model_bundle(self.bundle_root, verify_hash=True)
            model = create_gesture_model(bundle, verify_hash=False)
            class_thresholds = bundle.preprocessing.get("class_thresholds", {})
            if isinstance(class_thresholds, dict) and all(label in class_thresholds for label in bundle.labels):
                self._threshold = np.asarray(
                    [class_thresholds[label] for label in bundle.labels], dtype=np.float32)
            else:
                self._threshold = float(bundle.preprocessing.get("online_event_threshold", 0.50))
            config = InferenceConfig(
                sample_rate=bundle.sample_rate,
                channels=bundle.input_channels,
                buffer_seconds=float(bundle.preprocessing.get(
                    "realtime_buffer_seconds", 4.0)),
                model_window_seconds=float(bundle.preprocessing.get(
                    "realtime_model_window_seconds", 2.0)),
                fixed_lag_seconds=float(bundle.preprocessing.get(
                    "realtime_fixed_lag_seconds", 0.25)),
                threshold=float(bundle.preprocessing.get(
                    "online_event_threshold", 0.50)),
                debounce_seconds=float(bundle.preprocessing.get(
                    "debounce_seconds", 0.05)),
                minimum_hold_seconds=float(bundle.preprocessing.get(
                    "minimum_hold_seconds", 0.50)),
                left_context_samples=int(bundle.metadata["network"].get(
                    "left_context_samples", 20)),
                stride_samples=int(bundle.metadata["network"].get("stride", 10)),
            )
            engine = RealtimeGestureEngine(model, config)
            self.model_loaded.emit(bundle)
            self.status_changed.emit("模型已加载；请连接设备并执行推理校准")
            next_inference = 0.0
            deferred_command: tuple[str, object, object] | None = None
            while not self._stopping.is_set():
                if deferred_command is None:
                    try:
                        command, value, extra = self._commands.get(timeout=0.05)
                    except queue.Empty:
                        continue
                else:
                    command, value, extra = deferred_command
                    deferred_command = None
                if command == "stop":
                    break
                if command == "threshold":
                    self._threshold = float(value)
                    continue
                if command == "calibrate":
                    engine.begin_calibration(float(value))
                    self._mode = "calibrating"
                    self.status_changed.emit(
                        "正在校准：请保持自然姿势，并按页面提示做几次目标手势")
                    continue
                if command == "recognize":
                    if engine.scale_counts_per_unit is None:
                        self.status_changed.emit("请先完成本次佩戴的推理校准")
                    else:
                        engine.reset_stream()
                        self._mode = "recognizing"
                        next_inference = 0.0
                        self.status_changed.emit("实时识别运行中")
                    continue
                if command == "pause":
                    self._mode = "idle"
                    engine.reset_stream()
                    self.status_changed.emit("实时识别已暂停")
                    continue
                if command != "emg":
                    continue

                # While model inference was running, many small acquisition
                # chunks may have accumulated.  Ingest every consecutive EMG
                # command now, then perform at most one prediction using the
                # newest complete rolling window.  A following control command
                # keeps its original ordering and is handled next.
                emg_commands = [(value, extra)]
                while True:
                    try:
                        queued = self._commands.get_nowait()
                    except queue.Empty:
                        break
                    if queued[0] != "emg":
                        deferred_command = queued
                        break
                    emg_commands.append((queued[1], queued[2]))

                discontinuity = False
                for raw, indices in emg_commands:
                    discontinuity = engine.ingest(raw, indices) or discontinuity
                if discontinuity:
                    self.status_changed.emit("检测到采样不连续，已重置推理缓冲")
                if self._mode == "calibrating":
                    current, total = engine.calibration_progress
                    self.calibration_progress.emit(current, total)
                    if engine.calibration_complete:
                        scale = engine.finish_calibration()
                        self._mode = "idle"
                        self.calibration_finished.emit(scale)
                        self.status_changed.emit(
                            f"校准完成，缩放系数 {scale:.3f}；可以开始实时识别")
                    continue
                if self._mode != "recognizing" or not engine.ready_for_prediction:
                    continue
                now = time.monotonic()
                if now < next_inference:
                    continue
                frame = engine.predict(self._threshold)
                # Schedule from completion time.  Scheduling from the time
                # before a slow prediction made the next deadline already
                # overdue and caused an immediate inference loop on backlog.
                next_inference = time.monotonic() + config.inference_interval_seconds
                self.prediction_ready.emit(frame)
        except Exception as exc:
            self.failed.emit(str(exc))


class OfflineReplayWorker(QThread):
    progress = Signal(int, int)
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, bundle_root: Path, hdf5_path: Path,
                 threshold: float = 0.35, parent=None) -> None:
        super().__init__(parent)
        self.bundle_root = Path(bundle_root)
        self.hdf5_path = Path(hdf5_path)
        self.threshold = float(threshold)
        self._stopping = threading.Event()

    def stop(self) -> None:
        self._stopping.set()

    def run(self) -> None:
        try:
            bundle = load_model_bundle(self.bundle_root, verify_hash=True)
            model = create_gesture_model(bundle, verify_hash=False)
            labels = bundle.labels
            chunk_samples = 16000
            network = bundle.metadata.get("network", {})
            left_context = int(network.get("left_context_samples", 20))
            stride = int(network.get("stride", 10))
            # Each replay block is preceded by the same amount of history used
            # by realtime inference.  Outputs from the overlap are discarded,
            # avoiding convolution gaps and reducing LSTM cold-start effects at
            # every 8-second block boundary.
            warmup_samples = round(float(bundle.preprocessing.get(
                "realtime_model_window_seconds", 2.0)) * bundle.sample_rate)
            peak = np.zeros(len(labels), dtype=np.float32)
            event_counts = {name: 0 for name in labels}
            probability_chunks: list[np.ndarray] = []
            time_chunks: list[np.ndarray] = []
            previous: np.ndarray | None = None
            last_event_index: int | None = None
            last_event_name: str | None = None
            state_machine = OnlineGestureStateMachine(bundle.sample_rate)
            with h5py.File(self.hdf5_path, "r") as handle:
                if "data" not in handle:
                    raise ValueError("请选择 session_meta_aligned.hdf5，而不是原始 session.h5")
                data = handle["data"]
                if "emg" not in data.dtype.fields:
                    raise ValueError("HDF5 /data 中缺少 emg 字段")
                if "time" not in data.dtype.fields:
                    raise ValueError("HDF5 /data 中缺少 time 字段")
                channels = int(data.dtype.fields["emg"][0].shape[0])
                rate = float(data.attrs.get("sample_rate", 0.0))
                version = str(data.attrs.get("preprocessing_version", ""))
                if channels != bundle.input_channels or not np.isclose(rate, bundle.sample_rate):
                    raise ValueError("回放文件的通道数或采样率与模型不一致")
                if version != "meta_8ch_v1":
                    raise ValueError("回放文件不是 meta_8ch_v1 训练输入")
                total = len(data)
                processed = 0
                for start in range(0, total, chunk_samples):
                    if self._stopping.is_set():
                        raise InterruptedError("离线回放已取消")
                    end = min(total, start + chunk_samples)
                    context_start = max(0, start - warmup_samples)
                    block = data[context_start:end]
                    emg = np.asarray(block["emg"], dtype=np.float32)
                    block_times = np.asarray(block["time"], dtype=np.float64)
                    minimum_samples = left_context + 1
                    if len(emg) < minimum_samples:
                        break
                    probabilities = model.predict(emg)
                    output_indices = context_start + left_context + np.arange(
                        probabilities.shape[1], dtype=np.int64) * stride
                    keep = (output_indices >= start) & (output_indices < end)
                    probabilities = probabilities[:, keep]
                    output_indices = output_indices[keep]
                    if not len(output_indices):
                        processed = end
                        self.progress.emit(processed, total)
                        continue
                    output_times = block_times[output_indices - context_start]
                    probability_chunks.append(probabilities)
                    time_chunks.append(output_times)
                    peak = np.maximum(peak, probabilities.max(axis=1))
                    (events, previous, last_event_index,
                     last_event_name) = detect_threshold_events(
                        probabilities, output_indices, labels, bundle.display_names,
                        self.threshold, previous, round(0.05 * bundle.sample_rate),
                        last_event_index, last_event_name,
                    )
                    events = state_machine.filter(events)
                    for event in events:
                        event_counts[event.name] += 1
                    processed = end
                    self.progress.emit(processed, total)

            if not probability_chunks:
                raise ValueError("回放文件没有产生可评测的模型输出")
            all_probabilities = np.concatenate(probability_chunks, axis=1)
            all_times = np.concatenate(time_chunks)
            if len(all_times) > 1 and np.any(np.diff(all_times) <= 0):
                raise ValueError("回放输出时间不连续递增")

            try:
                prompts = pd.read_hdf(self.hdf5_path, key="prompts")
            except (KeyError, OSError, ValueError):
                prompts = pd.DataFrame(columns=["name", "time"])
            evaluation: dict[str, object] = {}
            if not prompts.empty and {"name", "time"}.issubset(prompts.columns):
                evaluation = _evaluate_offline_predictions(
                    all_probabilities, all_times, labels, prompts,
                    sample_rate=bundle.sample_rate,
                    stride=stride,
                    threshold=self.threshold,
                    include_cler=bundle.algorithm_id == META_CONV_LSTM,
                )
            self.succeeded.emit({
                "path": str(self.hdf5_path),
                "samples": processed,
                "duration_seconds": processed / bundle.sample_rate,
                "peak_probabilities": dict(zip(labels, map(float, peak))),
                "event_counts": event_counts,
                "threshold": self.threshold,
                "output_frames": len(all_times),
                "evaluation": evaluation,
            })
        except InterruptedError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            self.failed.emit(str(exc))


def _targets_from_prompts(
    output_times: np.ndarray,
    labels: tuple[str, ...],
    prompts: pd.DataFrame,
) -> np.ndarray:
    """Place each prompt on its nearest model output frame for FNR scoring."""
    times = np.asarray(output_times, dtype=np.float64)
    if times.ndim != 1 or not len(times):
        raise ValueError("模型输出时间不能为空")
    targets = np.zeros((len(times), len(labels)), dtype=np.float32)
    label_indices = {name: index for index, name in enumerate(labels)}
    for row in prompts.itertuples(index=False):
        name = str(getattr(row, "name"))
        if name not in label_indices:
            continue
        event_time = float(getattr(row, "time"))
        position = int(np.searchsorted(times, event_time))
        candidates = [index for index in (position - 1, position)
                      if 0 <= index < len(times)]
        if not candidates:
            continue
        nearest = min(candidates, key=lambda index: abs(times[index] - event_time))
        targets[nearest, label_indices[name]] = 1.0
    return targets


def _evaluate_offline_predictions(
    probabilities: np.ndarray,
    output_times: np.ndarray,
    labels: tuple[str, ...],
    prompts: pd.DataFrame,
    *,
    sample_rate: float,
    stride: int,
    threshold: float,
    include_cler: bool,
) -> dict[str, object]:
    """Evaluate one model pass over a fixed global and per-class threshold grid."""
    probs = np.asarray(probabilities, dtype=np.float32)
    times = np.asarray(output_times, dtype=np.float64)
    if probs.shape != (len(labels), len(times)):
        raise ValueError("离线概率、标签和输出时间形状不一致")
    frame_rate = float(sample_rate) / int(stride)
    targets = _targets_from_prompts(times, labels, prompts)
    frame_probabilities = probs.T

    current_fnr = evaluate_paper_fnr(
        frame_probabilities, targets, labels,
        threshold=threshold, frame_rate=frame_rate,
    )
    evaluation: dict[str, object] = {
        "mean_fnr": current_fnr.mean_fnr,
        "per_class_fnr": current_fnr.per_class_fnr,
        "ground_truth_counts": current_fnr.total,
        "correct_counts": current_fnr.correct,
        "predicted_events": current_fnr.predicted_events,
    }

    compute_cler = None
    if include_cler:
        from generic_neuromotor_interface.cler import compute_cler as cler_function
        compute_cler = cler_function
        evaluation["cler"] = float(compute_cler(
            probs, times, prompts, threshold=threshold))

    candidates = tuple(float(value) for value in np.round(
        np.arange(0.10, 0.501, 0.05), 2))
    sweep: list[dict[str, float]] = []
    for candidate in candidates:
        paper = evaluate_paper_fnr(
            frame_probabilities, targets, labels,
            threshold=candidate, frame_rate=frame_rate,
        )
        event_metrics = evaluate_events(
            frame_probabilities, targets, labels,
            thresholds=(candidate,) * len(labels),
            frame_rate=frame_rate,
        )
        row = {
            "threshold": candidate,
            "mean_fnr": paper.mean_fnr,
            "macro_f1": event_metrics.macro_f1,
            "false_positives_per_minute": event_metrics.false_positives_per_minute,
        }
        if compute_cler is not None:
            row["cler"] = float(compute_cler(
                probs, times, prompts, threshold=candidate))
        sweep.append(row)

    # Macro F1 balances missed events and extra threshold crossings.  FNR and
    # false positives are deterministic tie-breakers; the final term avoids a
    # systematic preference for an unnecessarily low threshold.
    best = min(sweep, key=lambda row: (
        -row["macro_f1"], row["mean_fnr"],
        row["false_positives_per_minute"], -row["threshold"],
    ))
    class_thresholds = search_thresholds(
        frame_probabilities, targets, labels, candidates=candidates)
    evaluation.update({
        "threshold_sweep": sweep,
        "recommended_threshold": best["threshold"],
        "recommended_class_thresholds": dict(zip(labels, class_thresholds)),
    })
    return evaluation
