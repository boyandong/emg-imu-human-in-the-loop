from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from time import time
from typing import Any, Callable, Sequence

import numpy as np


GESTURE_ORDER = (
    "index_press",
    "index_release",
    "middle_press",
    "middle_release",
)
PREPROCESSING_VERSION_BY_SAMPLE_RATE = {
    200: "emg_8ch_200hz_v1",
    2000: "meta_8ch_v1",
}


@dataclass(frozen=True)
class ActivityDetectorConfig:
    """Settings for user-paced EMG activity onset detection."""

    sample_rate_hz: int = 200
    channels: int = 8
    baseline_seconds: float = 3.0
    envelope_window_ms: float = 40.0
    onset_hold_ms: float = 20.0
    quiet_hold_ms: float = 150.0
    refractory_ms: float = 100.0
    onset_sigma: float = 8.0
    offset_sigma: float = 3.0
    minimum_scale: float = 1e-6

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0 or self.channels <= 0:
            raise ValueError("sample_rate_hz 和 channels 必须为正数")
        if self.baseline_seconds <= 0:
            raise ValueError("baseline_seconds 必须为正数")
        if min(self.envelope_window_ms, self.onset_hold_ms,
               self.quiet_hold_ms, self.refractory_ms) <= 0:
            raise ValueError("时间参数必须为正数")
        if not self.onset_sigma > self.offset_sigma > 0:
            raise ValueError("onset_sigma 必须大于 offset_sigma，且两者为正数")


@dataclass(frozen=True)
class ActivityEvent:
    onset_time: float
    offset_time: float
    peak_envelope: float


@dataclass(frozen=True)
class RecordedGestureEvent:
    name: str
    onset_time: float
    offset_time: float
    peak_envelope: float


class AdaptiveActivityDetector:
    """Detect activity bursts without imposing an action deadline.

    The detector first estimates a robust resting baseline, then uses a high
    threshold to start an event and a lower threshold to finish it.  It detects
    only *when* activity occurs; the recorder's fixed sequence supplies the label.
    """

    def __init__(self, config: ActivityDetectorConfig | None = None) -> None:
        self.config = config or ActivityDetectorConfig()
        cfg = self.config
        self._envelope_samples = max(1, round(cfg.envelope_window_ms * cfg.sample_rate_hz / 1000))
        self._baseline_samples = max(1, round(cfg.baseline_seconds * cfg.sample_rate_hz))
        self._onset_samples = max(1, round(cfg.onset_hold_ms * cfg.sample_rate_hz / 1000))
        self._quiet_samples = max(1, round(cfg.quiet_hold_ms * cfg.sample_rate_hz / 1000))
        self._refractory_samples = max(1, round(cfg.refractory_ms * cfg.sample_rate_hz / 1000))
        self._energy_window: deque[float] = deque()
        self._energy_sum = 0.0
        self._baseline_values: list[float] = []
        self._onset_threshold: float | None = None
        self._offset_threshold: float | None = None
        self._state = "CALIBRATING"
        self._candidate_count = 0
        self._candidate_time = 0.0
        self._quiet_count = 0
        self._quiet_start_time = 0.0
        self._refractory_count = 0
        self._active_start_time = 0.0
        self._active_peak = 0.0

    @property
    def calibrated(self) -> bool:
        return self._onset_threshold is not None

    @property
    def state(self) -> str:
        return self._state

    @property
    def thresholds(self) -> tuple[float, float] | None:
        if self._onset_threshold is None or self._offset_threshold is None:
            return None
        return self._onset_threshold, self._offset_threshold

    @property
    def calibration_progress(self) -> float:
        return min(1.0, len(self._baseline_values) / self._baseline_samples)

    def _envelope(self, sample: np.ndarray) -> float:
        energy = float(np.mean(np.square(sample, dtype=np.float64)))
        self._energy_window.append(energy)
        self._energy_sum += energy
        if len(self._energy_window) > self._envelope_samples:
            self._energy_sum -= self._energy_window.popleft()
        return float(np.sqrt(max(0.0, self._energy_sum / len(self._energy_window))))

    def _finish_calibration(self) -> None:
        values = np.asarray(self._baseline_values, dtype=np.float64)
        baseline = float(np.median(values))
        mad_scale = 1.4826 * float(np.median(np.abs(values - baseline)))
        std_scale = float(values.std())
        scale = max(mad_scale, std_scale * 0.25, self.config.minimum_scale)
        self._onset_threshold = baseline + self.config.onset_sigma * scale
        self._offset_threshold = baseline + self.config.offset_sigma * scale
        self._state = "WAITING"

    def process(self, samples: np.ndarray, timestamps: np.ndarray) -> list[ActivityEvent]:
        signal = np.asarray(samples, dtype=np.float32)
        times = np.asarray(timestamps, dtype=np.float64)
        if signal.ndim != 2 or signal.shape[1] != self.config.channels:
            raise ValueError(f"肌电数据应为 [samples,{self.config.channels}]")
        if times.ndim != 1 or len(times) != len(signal):
            raise ValueError("timestamps 必须与肌电采样点一一对应")
        if len(times) > 1 and np.any(np.diff(times) <= 0):
            raise ValueError("timestamps 必须严格递增")

        events: list[ActivityEvent] = []
        for sample, timestamp in zip(signal, times):
            envelope = self._envelope(sample)
            current_time = float(timestamp)
            if self._state == "CALIBRATING":
                self._baseline_values.append(envelope)
                if len(self._baseline_values) >= self._baseline_samples:
                    self._finish_calibration()
                continue

            assert self._onset_threshold is not None
            assert self._offset_threshold is not None
            if self._state == "REFRACTORY":
                self._refractory_count -= 1
                if self._refractory_count <= 0:
                    self._state = "WAITING"
                continue

            if self._state == "WAITING":
                if envelope >= self._onset_threshold:
                    if self._candidate_count == 0:
                        self._candidate_time = current_time
                    self._candidate_count += 1
                    if self._candidate_count >= self._onset_samples:
                        self._state = "ACTIVE"
                        self._active_start_time = self._candidate_time
                        self._active_peak = envelope
                        self._candidate_count = 0
                else:
                    self._candidate_count = 0
                continue

            self._active_peak = max(self._active_peak, envelope)
            if envelope <= self._offset_threshold:
                if self._quiet_count == 0:
                    self._quiet_start_time = current_time
                self._quiet_count += 1
                if self._quiet_count >= self._quiet_samples:
                    events.append(ActivityEvent(
                        onset_time=self._active_start_time,
                        offset_time=self._quiet_start_time,
                        peak_envelope=self._active_peak,
                    ))
                    self._quiet_count = 0
                    self._refractory_count = self._refractory_samples
                    self._state = "REFRACTORY"
            else:
                self._quiet_count = 0
        return events


class ContinuousGestureRecorder:
    """Continuously persist EMG and mark user-paced events in a fixed order.

    Acquisition software should call :meth:`append` for every device data block.
    The HDF5 ``/data`` dataset is extended immediately, so automatic event
    detection never replaces or crops the original continuous signal.
    """

    def __init__(
        self,
        output_path: str | Path,
        detector_config: ActivityDetectorConfig | None = None,
        gesture_order: Sequence[str] = GESTURE_ORDER,
        preprocessing_version: str | None = None,
        on_event: Callable[[RecordedGestureEvent], None] | None = None,
    ) -> None:
        self.output_path = Path(output_path)
        self.detector = AdaptiveActivityDetector(detector_config)
        self.gesture_order = tuple(gesture_order)
        if not self.gesture_order:
            raise ValueError("gesture_order 不能为空")
        if preprocessing_version is None:
            preprocessing_version = PREPROCESSING_VERSION_BY_SAMPLE_RATE.get(
                self.detector.config.sample_rate_hz)
        if preprocessing_version is None:
            raise ValueError("未知采样率需要显式指定 preprocessing_version")
        self.preprocessing_version = preprocessing_version
        self.on_event = on_event
        self.events: list[RecordedGestureEvent] = []
        self._handle: Any = None
        self._dataset: Any = None
        self._last_timestamp: float | None = None

    @property
    def started(self) -> bool:
        return self._handle is not None

    @property
    def expected_label(self) -> str:
        return self.gesture_order[len(self.events) % len(self.gesture_order)]

    def start(self) -> None:
        if self.started:
            raise RuntimeError("采集已经开始")
        import h5py

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        cfg = self.detector.config
        dtype = np.dtype([("time", np.float64), ("emg", np.float32, (cfg.channels,))])
        self._handle = h5py.File(self.output_path, "x")
        self._dataset = self._handle.create_dataset(
            "data", shape=(0,), maxshape=(None,), chunks=True, dtype=dtype)
        self._dataset.attrs["task"] = "discrete_gestures"
        self._dataset.attrs["sample_rate"] = cfg.sample_rate_hz
        self._dataset.attrs["preprocessing_version"] = self.preprocessing_version
        self._dataset.attrs["prompt_time_reference"] = "emg_onset"
        self._dataset.attrs["gesture_order"] = ",".join(self.gesture_order)
        self._handle.flush()

    def _timestamps_for(self, sample_count: int, timestamps: np.ndarray | None) -> np.ndarray:
        if timestamps is not None:
            result = np.asarray(timestamps, dtype=np.float64)
            if result.shape != (sample_count,):
                raise ValueError("timestamps 必须与肌电采样点一一对应")
        else:
            step = 1.0 / self.detector.config.sample_rate_hz
            start = time() if self._last_timestamp is None else self._last_timestamp + step
            result = start + np.arange(sample_count, dtype=np.float64) * step
        if len(result) > 1 and np.any(np.diff(result) <= 0):
            raise ValueError("timestamps 必须严格递增")
        if len(result) and self._last_timestamp is not None and result[0] <= self._last_timestamp:
            raise ValueError("新数据的时间戳必须晚于上一数据块")
        return result

    def append(
        self, samples: np.ndarray, timestamps: np.ndarray | None = None,
    ) -> list[RecordedGestureEvent]:
        if not self.started:
            raise RuntimeError("请先调用 start()")
        signal = np.asarray(samples, dtype=np.float32)
        if signal.ndim != 2 or signal.shape[1] != self.detector.config.channels:
            raise ValueError(f"肌电数据应为 [samples,{self.detector.config.channels}]")
        if not len(signal):
            return []
        times = self._timestamps_for(len(signal), timestamps)

        rows = np.empty(len(signal), dtype=self._dataset.dtype)
        rows["time"] = times
        rows["emg"] = signal
        start = len(self._dataset)
        self._dataset.resize((start + len(rows),))
        self._dataset[start:] = rows
        self._handle.flush()
        self._last_timestamp = float(times[-1])

        recorded: list[RecordedGestureEvent] = []
        for activity in self.detector.process(signal, times):
            event = RecordedGestureEvent(
                name=self.expected_label,
                onset_time=activity.onset_time,
                offset_time=activity.offset_time,
                peak_envelope=activity.peak_envelope,
            )
            self.events.append(event)
            recorded.append(event)
            if self.on_event is not None:
                self.on_event(event)
        return recorded

    def undo_last_event(self) -> RecordedGestureEvent | None:
        return self.events.pop() if self.events else None

    def stop(self) -> Path:
        if not self.started:
            raise RuntimeError("采集尚未开始")
        self._handle.attrs["detector_config_json"] = __import__("json").dumps(
            asdict(self.detector.config), ensure_ascii=False)
        if self.detector.thresholds is not None:
            self._handle.attrs["detector_onset_threshold"] = self.detector.thresholds[0]
            self._handle.attrs["detector_offset_threshold"] = self.detector.thresholds[1]
        event_dtype = np.dtype([
            ("name", "S32"), ("onset_time", np.float64),
            ("offset_time", np.float64), ("peak_envelope", np.float64),
        ])
        event_rows = np.empty(len(self.events), dtype=event_dtype)
        for index, event in enumerate(self.events):
            event_rows[index] = (
                event.name.encode("utf-8"), event.onset_time,
                event.offset_time, event.peak_envelope)
        self._handle.create_dataset("collection_events", data=event_rows)
        self._handle.close()
        self._handle = None
        self._dataset = None

        import pandas as pd

        prompts = pd.DataFrame(
            [{"name": event.name, "time": event.onset_time} for event in self.events],
            columns=["name", "time"],
        )
        prompts.to_hdf(self.output_path, "prompts", mode="a", format="table")
        return self.output_path

    def __enter__(self) -> "ContinuousGestureRecorder":
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.started:
            if exc_type is None:
                self.stop()
            else:
                self._handle.close()
                self._handle = None
                self._dataset = None
