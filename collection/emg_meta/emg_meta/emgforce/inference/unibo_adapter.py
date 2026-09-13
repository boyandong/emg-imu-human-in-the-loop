from __future__ import annotations

import hashlib
import json
import pickle
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QThread, Signal
from scipy.signal import resample_poly

from emgforce.algorithms import UNIBO_4CH_ADAPTER

from .engine import DetectedEvent, PredictionFrame
from .model_bundle import ModelBundle


MUSCLE_NAMES = ("ECU", "EDC", "FCR", "FCU")
MODEL_LABELS = ("neutral", "pinch", "fist", "open")
DISPLAY_NAMES = {
    "neutral": "自然静息",
    "pinch": "捏合 (Pinch)",
    "fist": "握拳 (Fist)",
    "open": "张手 (Open)",
}
POSTURE_NAMES = {
    1: "近端 (proximal)",
    2: "远端 (distal)",
    3: "远端掌心向下",
    4: "远端手臂上抬 45°",
}
DEFAULT_CHANNEL_MAP = (0, 2, 4, 6)
DEVICE_SAMPLE_RATE = 250
UNIBO_SAMPLE_RATE = 200
WINDOW_SECONDS = 0.2


@dataclass(frozen=True, slots=True)
class UniBoModelInfo:
    artifact: Path
    experiment: str
    fold: str
    validation_accuracy: float | None
    validation_macro_f1: float | None
    validation_active_f1: float | None
    posture_required: bool

    @property
    def combo_key(self) -> str:
        return f"unibo::{self.artifact}"


def _find_repository_root(start: Path) -> Path | None:
    path = Path(start).resolve()
    for candidate in (path, *path.parents):
        if (candidate / "emgimu_classifier" / "src" / "emgimu").is_dir():
            return candidate
    return None


def _metric_value(metrics: dict[str, Any], key: str) -> float | None:
    try:
        return float(metrics["validation"]["window_raw"][key])
    except (KeyError, TypeError, ValueError):
        return None


def discover_unibo_models(start: Path) -> list[UniBoModelInfo]:
    """Find the trained UniBo physiology artifacts shipped with this repository."""
    repository = _find_repository_root(start)
    if repository is None:
        return []
    base = repository / "emgimu_classifier" / "baseline" / "unibo"
    models_root = base / "models"
    found: list[UniBoModelInfo] = []
    for artifact in models_root.glob("*/*/*/model.pkl"):
        relative = artifact.relative_to(models_root)
        run_name, experiment, fold = relative.parts[:3]
        metrics_path = base / "results" / run_name / experiment / fold / "metrics.json"
        metrics: dict[str, Any] = {}
        try:
            with metrics_path.open("r", encoding="utf-8") as stream:
                loaded = json.load(stream)
            if isinstance(loaded, dict):
                metrics = loaded
        except (OSError, ValueError, TypeError):
            pass
        found.append(UniBoModelInfo(
            artifact=artifact.resolve(),
            experiment=experiment,
            fold=fold,
            validation_accuracy=_metric_value(metrics, "accuracy"),
            validation_macro_f1=_metric_value(metrics, "macro_f1"),
            validation_active_f1=_metric_value(metrics, "active_gesture_macro_f1"),
            posture_required=experiment.lower().startswith("e6"),
        ))
    priority = {"E5": 0, "E0": 1, "E6b": 2}
    return sorted(found, key=lambda item: (priority.get(item.experiment, 9), item.experiment, item.fold))


def unibo_key_path(value: object) -> Path | None:
    text = str(value or "")
    if not text.startswith("unibo::"):
        return None
    return Path(text[len("unibo::"):])


class _UniBoArtifactUnpickler(pickle.Unpickler):
    """Load artifacts written while unibo_physiology ran as ``__main__``."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "__main__" and name == "PhysiologyFeatureTransformer":
            from emgimu.datasets.unibo_physiology import PhysiologyFeatureTransformer
            return PhysiologyFeatureTransformer
        return super().find_class(module, name)


def _load_artifact(path: Path) -> dict[str, Any]:
    artifact_path = Path(path).resolve()
    repository = _find_repository_root(artifact_path)
    if repository is None:
        raise RuntimeError("无法定位 emgimu_classifier/src，UniBo 模型不能反序列化")
    source_root = str(repository / "emgimu_classifier" / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    try:
        with artifact_path.open("rb") as stream:
            artifact = _UniBoArtifactUnpickler(stream).load()
    except ModuleNotFoundError as exc:
        if exc.name == "sklearn":
            raise RuntimeError("缺少 scikit-learn；请更新 emgforce 环境后再加载 UniBo 模型") from exc
        raise
    if not isinstance(artifact, dict):
        raise ValueError("UniBo model.pkl 顶层不是模型字典")
    if artifact.get("model_kind") != "unibo_four_channel_physiology_rbf_svm":
        raise ValueError(f"不支持的 UniBo 模型类型：{artifact.get('model_kind')}")
    required = {"model", "feature_transformer", "classes", "class_names"}
    missing = sorted(required - set(artifact))
    if missing:
        raise ValueError(f"UniBo 模型缺少字段：{', '.join(missing)}")
    return artifact


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64) / max(float(temperature), 1e-6)
    values -= np.max(values, axis=1, keepdims=True)
    exponential = np.exp(values)
    return exponential / np.maximum(exponential.sum(axis=1, keepdims=True), 1e-12)


class UniBoAdapterRuntime:
    """Adapt an 8-channel 250 Hz ring stream to a trained 4-channel 200 Hz UniBo SVM.

    The channel selection is explicit because ring electrodes are not anatomically
    equivalent to the UniBo ECU/EDC/FCR/FCU placement.  The adapter is intended for
    exploratory live checks and preserves that domain-mismatch marker in its bundle.
    """

    def __init__(self, artifact_path: Path, channel_map: tuple[int, int, int, int],
                 posture: int = 1) -> None:
        if len(channel_map) != 4 or len(set(channel_map)) != 4:
            raise ValueError("ECU/EDC/FCR/FCU 必须映射到四个不同的设备通道")
        if any(channel < 0 or channel >= 8 for channel in channel_map):
            raise ValueError("UniBo 通道映射必须位于 CH1–CH8")
        if int(posture) not in POSTURE_NAMES:
            raise ValueError("UniBo 姿态编号必须为 1–4")
        self.artifact_path = Path(artifact_path).resolve()
        self.channel_map = tuple(int(value) for value in channel_map)
        self.posture = int(posture)
        self.artifact = _load_artifact(self.artifact_path)
        self.model = self.artifact["model"]
        self.transformer = self.artifact["feature_transformer"]
        self.temperature = float(self.artifact.get("temperature", 1.0))
        self.threshold = float(self.artifact.get("threshold", 0.5))
        self.experiment = str(self.artifact.get("experiment", self.artifact_path.parent.parent.name))
        self.fold = str(self.artifact.get("fold", self.artifact_path.parent.name))
        self.channel_gains = np.ones(4, dtype=np.float64)

    @property
    def raw_window_samples(self) -> int:
        return round(DEVICE_SAMPLE_RATE * WINDOW_SECONDS)

    @property
    def model_window_samples(self) -> int:
        return round(UNIBO_SAMPLE_RATE * WINDOW_SECONDS)

    def _select_and_resample(self, raw: np.ndarray) -> np.ndarray:
        values = np.asarray(raw, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 8:
            raise ValueError(f"UniBo 适配器要求输入 [samples,8]，实际为 {values.shape}")
        selected = values[:, self.channel_map]
        converted = resample_poly(selected, up=4, down=5, axis=0)
        return converted.astype(np.float64, copy=False)

    def calibrate_neutral(self, raw: np.ndarray) -> np.ndarray:
        converted = self._select_and_resample(raw)
        if len(converted) < self.model_window_samples:
            raise ValueError("静息校准数据不足 200 ms")
        window_count = 1 + (len(converted) - self.model_window_samples) // 10
        rms = np.empty((window_count, 4), dtype=np.float64)
        for index in range(window_count):
            start = index * 10
            window = converted[start:start + self.model_window_samples]
            rms[index] = np.sqrt(np.mean(window * window, axis=0))
        observed = np.maximum(np.percentile(rms, 95, axis=0), 1e-12)
        target = getattr(self.transformer, "neutral_thresholds_", None)
        if target is None:
            try:
                target = np.asarray(self.model.named_steps["scaler"].mean_[:4], dtype=np.float64)
            except (AttributeError, KeyError, TypeError):
                target = observed
        target = np.maximum(np.asarray(target, dtype=np.float64).reshape(4), 1e-12)
        self.channel_gains = np.clip(target / observed, 1e-8, 1e8)
        return self.channel_gains.copy()

    def predict(self, raw_window: np.ndarray) -> np.ndarray:
        if len(raw_window) != self.raw_window_samples:
            raise ValueError(
                f"UniBo 实时窗口必须为 {self.raw_window_samples} 个 250 Hz 采样点")
        converted = self._select_and_resample(raw_window)
        if len(converted) != self.model_window_samples:
            raise RuntimeError(f"250→200 Hz 重采样得到异常长度：{len(converted)}")
        converted *= self.channel_gains.reshape(1, 4)
        posture = np.asarray([self.posture], dtype=np.int16)
        features = self.transformer.transform(converted[None, :, :], posture)
        logits = np.asarray(self.model.decision_function(features), dtype=np.float64)
        if logits.ndim == 1:
            logits = np.column_stack([-logits, logits])
        probabilities = _softmax(logits, self.temperature)[0]
        if probabilities.shape != (4,) or not np.isfinite(probabilities).all():
            raise RuntimeError(f"UniBo 模型输出异常：{probabilities.shape}")
        return probabilities.astype(np.float32)

    def make_bundle(self) -> ModelBundle:
        digest = hashlib.sha256(self.artifact_path.read_bytes()).hexdigest()
        mapping = {muscle: f"CH{channel + 1}" for muscle, channel in zip(MUSCLE_NAMES, self.channel_map)}
        metrics = _read_neighbor_metrics(self.artifact_path)
        return ModelBundle(
            root=self.artifact_path.parent,
            model_id=f"UniBo {self.experiment} · 8ch/250Hz 实验适配",
            artifact=self.artifact_path,
            sha256=digest,
            labels=MODEL_LABELS,
            display_names=DISPLAY_NAMES,
            sample_rate=DEVICE_SAMPLE_RATE,
            input_channels=8,
            output_channels=4,
            algorithm_id=UNIBO_4CH_ADAPTER,
            runtime_backend="unibo_svm_adapter",
            preprocessing={
                "online_event_threshold": self.threshold,
                "source_sample_rate_hz": UNIBO_SAMPLE_RATE,
                "window_samples": self.model_window_samples,
            },
            metadata={
                "experimental_adapter": True,
                "domain_mismatch": "8-channel ring electrodes mapped to UniBo ECU/EDC/FCR/FCU",
                "channel_mapping": mapping,
                "posture": self.posture,
                "posture_name": POSTURE_NAMES[self.posture],
                "training": metrics,
                "source_checkpoint": {"path": str(self.artifact_path), "sha256": digest},
            },
        )


def _read_neighbor_metrics(artifact_path: Path) -> dict[str, Any]:
    parts = artifact_path.parts
    try:
        index = parts.index("models")
    except ValueError:
        return {}
    results_path = Path(*parts[:index], "results", *parts[index + 1:-1], "metrics.json")
    try:
        metrics = json.loads(results_path.read_text(encoding="utf-8"))
        raw = metrics["validation"]["window_raw"]
        return {
            "validation_accuracy": float(raw["accuracy"]),
            "validation_macro_f1": float(raw["macro_f1"]),
            "validation_active_f1": float(raw["active_gesture_macro_f1"]),
            "trained_at": "UniBo Day 1–5 → Day 6",
        }
    except (OSError, ValueError, TypeError, KeyError):
        return {"trained_at": "UniBo Day 1–5 → Day 6"}


class UniBoRealtimeWorker(QThread):
    model_loaded = Signal(object)
    status_changed = Signal(str)
    failed = Signal(str)
    calibration_progress = Signal(int, int)
    calibration_finished = Signal(float)
    prediction_ready = Signal(object)

    def __init__(self, artifact_path: Path, channel_map: tuple[int, int, int, int],
                 posture: int = 1, parent=None) -> None:
        super().__init__(parent)
        self.artifact_path = Path(artifact_path)
        self.channel_map = channel_map
        self.posture = int(posture)
        self._commands: queue.Queue[tuple[str, object, object]] = queue.Queue()
        self._stopping = threading.Event()
        self._mode = "idle"
        self._threshold = 0.5

    def submit_emg(self, raw: np.ndarray, indices: np.ndarray) -> None:
        self._commands.put_nowait((
            "emg", np.asarray(raw, dtype=np.int32).copy(),
            np.asarray(indices, dtype=np.int64).copy()))

    def begin_calibration(self, seconds: float) -> None:
        self._commands.put_nowait(("calibrate", float(seconds), None))

    def begin_recognition(self) -> None:
        self._commands.put_nowait(("recognize", None, None))

    def pause_recognition(self) -> None:
        self._commands.put_nowait(("pause", None, None))

    def set_threshold(self, value: float) -> None:
        self._commands.put_nowait(("threshold", float(value), None))

    def stop(self) -> None:
        self._stopping.set()
        self._commands.put_nowait(("stop", None, None))

    def run(self) -> None:
        try:
            runtime = UniBoAdapterRuntime(self.artifact_path, self.channel_map, self.posture)
            bundle = runtime.make_bundle()
            self._threshold = runtime.threshold
            self.model_loaded.emit(bundle)
            self.status_changed.emit("UniBo 适配器已加载；请保持静息完成通道幅值校准")
            raw_buffer = np.empty((0, 8), dtype=np.int32)
            calibration_chunks: list[np.ndarray] = []
            calibration_total = 0
            calibration_count = 0
            calibrated = False
            last_sample_index: int | None = None
            last_class: str | None = None
            next_inference = 0.0
            while not self._stopping.is_set():
                try:
                    command, value, extra = self._commands.get(timeout=0.05)
                except queue.Empty:
                    continue
                if command == "stop":
                    break
                if command == "threshold":
                    self._threshold = float(value)
                    continue
                if command == "calibrate":
                    calibrated = False
                    calibration_chunks.clear()
                    calibration_count = 0
                    calibration_total = max(1, round(float(value) * DEVICE_SAMPLE_RATE))
                    self._mode = "calibrating"
                    self.status_changed.emit("正在进行 UniBo 静息校准：保持手掌自然放松")
                    continue
                if command == "recognize":
                    if not calibrated:
                        self.status_changed.emit("请先完成 UniBo 静息校准")
                    else:
                        raw_buffer = np.empty((0, 8), dtype=np.int32)
                        last_class = None
                        self._mode = "recognizing"
                        self.status_changed.emit("UniBo 四分类实时识别运行中（实验适配）")
                    continue
                if command == "pause":
                    self._mode = "idle"
                    raw_buffer = np.empty((0, 8), dtype=np.int32)
                    self.status_changed.emit("实时识别已暂停")
                    continue
                if command != "emg":
                    continue
                raw = np.asarray(value, dtype=np.int32)
                indices = np.asarray(extra, dtype=np.int64)
                if raw.ndim != 2 or raw.shape[1] != 8 or len(raw) != len(indices):
                    continue
                if len(indices) and last_sample_index is not None and indices[0] != last_sample_index + 1:
                    raw_buffer = np.empty((0, 8), dtype=np.int32)
                    self.status_changed.emit("检测到采样不连续，已重置 UniBo 推理窗口")
                if len(indices):
                    last_sample_index = int(indices[-1])
                if self._mode == "calibrating":
                    needed = max(0, calibration_total - calibration_count)
                    accepted = raw[:needed]
                    if len(accepted):
                        calibration_chunks.append(accepted)
                        calibration_count += len(accepted)
                    self.calibration_progress.emit(calibration_count, calibration_total)
                    if calibration_count >= calibration_total:
                        gains = runtime.calibrate_neutral(np.concatenate(calibration_chunks, axis=0))
                        calibrated = True
                        self._mode = "idle"
                        scale = float(np.exp(np.mean(np.log(np.maximum(gains, 1e-12)))))
                        self.calibration_finished.emit(scale)
                        mapping = ", ".join(
                            f"{name}=CH{channel + 1}" for name, channel in zip(MUSCLE_NAMES, self.channel_map))
                        self.status_changed.emit(f"UniBo 静息校准完成；{mapping}")
                    continue
                if self._mode != "recognizing":
                    continue
                raw_buffer = np.concatenate((raw_buffer, raw), axis=0)[-500:]
                if len(raw_buffer) < runtime.raw_window_samples or time.monotonic() < next_inference:
                    continue
                started = time.perf_counter()
                probabilities = runtime.predict(raw_buffer[-runtime.raw_window_samples:])
                inference_ms = (time.perf_counter() - started) * 1000.0
                peak_index = int(np.argmax(probabilities))
                peak_name = MODEL_LABELS[peak_index]
                peak_probability = float(probabilities[peak_index])
                events: tuple[DetectedEvent, ...] = ()
                confident_class = peak_name if peak_probability >= self._threshold else None
                if confident_class is not None and confident_class != last_class:
                    events = (DetectedEvent(
                        name=confident_class,
                        display_name=DISPLAY_NAMES[confident_class],
                        sample_index=int(last_sample_index or 0),
                        probability=peak_probability,
                    ),)
                last_class = confident_class
                scale = float(np.exp(np.mean(np.log(np.maximum(runtime.channel_gains, 1e-12)))))
                self.prediction_ready.emit(PredictionFrame(
                    probabilities=probabilities,
                    labels=MODEL_LABELS,
                    events=events,
                    output_sample_index=int(last_sample_index or 0),
                    output_age_ms=0.0,
                    fixed_lag_ms=WINDOW_SECONDS * 1000.0,
                    inference_ms=inference_ms,
                    scale_counts_per_unit=scale,
                ))
                next_inference = time.monotonic() + 0.10
        except Exception as exc:
            self.failed.emit(str(exc))
