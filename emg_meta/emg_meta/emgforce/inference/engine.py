from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .preprocessing import estimate_realtime_scale, fixed_lag_model_window


class GestureModel(Protocol):
    bundle: object

    def predict(self, emg: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    sample_rate: int = 2000
    channels: int = 8
    buffer_seconds: float = 4.0
    model_window_seconds: float = 2.0
    fixed_lag_seconds: float = 0.25
    inference_interval_seconds: float = 0.10
    threshold: float = 0.50
    debounce_seconds: float = 0.05
    minimum_hold_seconds: float = 0.50
    left_context_samples: int = 20
    stride_samples: int = 10

    @property
    def buffer_samples(self) -> int:
        return round(self.buffer_seconds * self.sample_rate)

    @property
    def model_window_samples(self) -> int:
        return round(self.model_window_seconds * self.sample_rate)

    @property
    def lag_samples(self) -> int:
        return round(self.fixed_lag_seconds * self.sample_rate)


@dataclass(frozen=True, slots=True)
class DetectedEvent:
    name: str
    display_name: str
    sample_index: int
    probability: float
    synthetic: bool = False
    hold_duration_seconds: float | None = None
    hold_valid: bool | None = None


@dataclass(frozen=True, slots=True)
class PredictionFrame:
    probabilities: np.ndarray
    labels: tuple[str, ...]
    events: tuple[DetectedEvent, ...]
    output_sample_index: int
    output_age_ms: float
    fixed_lag_ms: float
    inference_ms: float
    scale_counts_per_unit: float


def detect_threshold_events(
    probabilities: np.ndarray,
    sample_indices: np.ndarray,
    labels: tuple[str, ...],
    display_names: dict[str, str],
    threshold: float | np.ndarray | tuple[float, ...],
    previous_probabilities: np.ndarray | None = None,
    debounce_samples: int = 100,
    last_event_index: int | None = None,
    last_event_name: str | None = None,
) -> tuple[list[DetectedEvent], np.ndarray, int | None, str | None]:
    """Detect rising threshold crossings and apply Meta-compatible debouncing."""
    probs = np.asarray(probabilities, dtype=np.float32)
    indices = np.asarray(sample_indices, dtype=np.int64)
    if probs.ndim != 2 or probs.shape != (len(labels), len(indices)):
        raise ValueError("概率、标签和采样索引的形状不一致")
    thresholds = np.asarray(threshold, dtype=np.float32)
    if thresholds.ndim == 0:
        thresholds = np.full(len(labels), float(thresholds), dtype=np.float32)
    if thresholds.shape != (len(labels),):
        raise ValueError("阈值必须是标量或与标签数相同的一维数组")
    if previous_probabilities is None:
        previous = np.zeros(len(labels), dtype=np.float32)
    else:
        previous = np.asarray(previous_probabilities, dtype=np.float32).copy()
    candidates: list[tuple[int, str, float]] = []
    for column, sample_index in enumerate(indices):
        current = probs[:, column]
        crossed = np.flatnonzero((current >= thresholds) & (previous < thresholds))
        for label_index in crossed:
            candidates.append((
                int(sample_index), labels[int(label_index)], float(current[label_index])))
        previous = current

    # The network is trained with independent BCE outputs, so more than one
    # class can cross its threshold on the same output frame.  Treat one frame
    # as one physical event and arbitrate explicitly by probability instead of
    # allowing the label order to decide which event survives debouncing.
    winners: list[tuple[int, str, float]] = []
    for candidate in sorted(candidates, key=lambda item: (item[0], -item[2])):
        if winners and candidate[0] == winners[-1][0]:
            continue
        winners.append(candidate)

    release_names = {"index_release", "middle_release"}
    events: list[DetectedEvent] = []
    for sample_index, name, probability in winners:
        if last_event_index is not None and sample_index - last_event_index < debounce_samples:
            current_is_release = name in release_names
            # Meta: suppress every nearby second event except a release that is
            # preceded by a different gesture (for rapid press/release taps).
            if not (current_is_release and name != last_event_name):
                continue
        events.append(DetectedEvent(
            name=name,
            display_name=display_names.get(name, name),
            sample_index=sample_index,
            probability=probability,
        ))
        last_event_index = sample_index
        last_event_name = name
    return events, previous, last_event_index, last_event_name


class OnlineGestureStateMachine:
    """Meta online press/release filtering and hold-duration tracking."""

    _PRESS_TO_RELEASE = {
        "index_press": "index_release",
        "middle_press": "middle_release",
    }

    def __init__(self, sample_rate: int, minimum_hold_seconds: float = 0.5) -> None:
        self.sample_rate = int(sample_rate)
        self.minimum_hold_samples = round(minimum_hold_seconds * sample_rate)
        self._pending_press: DetectedEvent | None = None

    def reset(self) -> None:
        self._pending_press = None

    def filter(self, events: list[DetectedEvent]) -> list[DetectedEvent]:
        output: list[DetectedEvent] = []
        for event in events:
            pending = self._pending_press
            if pending is not None:
                expected_release = self._PRESS_TO_RELEASE[pending.name]
                if event.name == expected_release:
                    duration_samples = max(0, event.sample_index - pending.sample_index)
                    output.append(DetectedEvent(
                        name=event.name,
                        display_name=event.display_name,
                        sample_index=event.sample_index,
                        probability=event.probability,
                        hold_duration_seconds=duration_samples / self.sample_rate,
                        hold_valid=duration_samples >= self.minimum_hold_samples,
                    ))
                    self._pending_press = None
                    continue

                # Meta online logic synthetically releases an active hold before
                # accepting any event other than its matching release.
                output.append(DetectedEvent(
                    name=expected_release,
                    display_name=("食指释放（自动补全）" if pending.name == "index_press"
                                  else "中指释放（自动补全）"),
                    sample_index=event.sample_index,
                    probability=0.0,
                    synthetic=True,
                    hold_duration_seconds=max(
                        0, event.sample_index - pending.sample_index) / self.sample_rate,
                    hold_valid=False,
                ))
                self._pending_press = None

            if event.name in self._PRESS_TO_RELEASE:
                self._pending_press = event
                output.append(event)
            elif event.name in self._PRESS_TO_RELEASE.values():
                # Release without the corresponding active press has no online effect.
                continue
            else:
                output.append(event)
        return output


class RealtimeGestureEngine:
    """State container for calibration, fixed-lag filtering and event decoding."""

    def __init__(self, model: GestureModel, config: InferenceConfig | None = None) -> None:
        self.model = model
        self.config = config or InferenceConfig()
        bundle = model.bundle
        self.labels = tuple(bundle.labels)
        self.display_names = dict(bundle.display_names)
        if bundle.input_channels != self.config.channels:
            raise ValueError("模型通道数与实时推理配置不一致")
        if bundle.sample_rate != self.config.sample_rate:
            raise ValueError("模型采样率与实时推理配置不一致")
        self.scale_counts_per_unit: float | None = None
        self._raw = np.empty((0, self.config.channels), dtype=np.int32)
        self._last_sample_index: int | None = None
        self._calibration = np.empty((0, self.config.channels), dtype=np.int32)
        self._calibration_target = 0
        self._last_scanned_output_index: int | None = None
        self._previous_probabilities: np.ndarray | None = None
        self._last_event_index: int | None = None
        self._last_event_name: str | None = None
        self._state_machine = OnlineGestureStateMachine(
            self.config.sample_rate, self.config.minimum_hold_seconds)

    @property
    def ready_for_prediction(self) -> bool:
        required = self.config.model_window_samples + self.config.lag_samples
        return self.scale_counts_per_unit is not None and len(self._raw) >= required

    @property
    def calibration_progress(self) -> tuple[int, int]:
        return min(len(self._calibration), self._calibration_target), self._calibration_target

    @property
    def calibration_complete(self) -> bool:
        return self._calibration_target > 0 and len(self._calibration) >= self._calibration_target

    def begin_calibration(self, seconds: float) -> None:
        target = round(float(seconds) * self.config.sample_rate)
        if target < self.config.sample_rate:
            raise ValueError("校准时长至少为 1 秒")
        self._calibration_target = target
        self._calibration = np.empty((0, self.config.channels), dtype=np.int32)
        self.scale_counts_per_unit = None
        self.reset_stream()

    def finish_calibration(self) -> float:
        if not self.calibration_complete:
            raise RuntimeError("校准数据尚未采集完成")
        block = self._calibration[:self._calibration_target]
        self.scale_counts_per_unit = estimate_realtime_scale(
            block, self.config.sample_rate)
        self._calibration_target = 0
        self._calibration = np.empty((0, self.config.channels), dtype=np.int32)
        return self.scale_counts_per_unit

    def set_scale(self, scale_counts_per_unit: float) -> None:
        value = float(scale_counts_per_unit)
        if not np.isfinite(value) or value <= 0:
            raise ValueError("缩放系数必须为正数")
        self.scale_counts_per_unit = value

    def reset_stream(self) -> None:
        self._raw = np.empty((0, self.config.channels), dtype=np.int32)
        self._last_sample_index = None
        self._last_scanned_output_index = None
        self._previous_probabilities = None
        self._last_event_index = None
        self._last_event_name = None
        self._state_machine.reset()

    def ingest(self, raw: np.ndarray, indices: np.ndarray | None = None) -> bool:
        block = np.asarray(raw, dtype=np.int32)
        if block.ndim != 2 or block.shape[1] != self.config.channels:
            raise ValueError(f"EMG 数据应为 [samples,{self.config.channels}]，实际 {block.shape}")
        if len(block) == 0:
            return False
        discontinuity = False
        if indices is None:
            first = 0 if self._last_sample_index is None else self._last_sample_index + 1
            sample_indices = np.arange(first, first + len(block), dtype=np.int64)
        else:
            sample_indices = np.asarray(indices, dtype=np.int64)
            if sample_indices.shape != (len(block),):
                raise ValueError("EMG 采样索引长度不匹配")
            if np.any(np.diff(sample_indices) != 1):
                discontinuity = True
            if (self._last_sample_index is not None
                    and int(sample_indices[0]) != self._last_sample_index + 1):
                discontinuity = True
        if discontinuity:
            self.reset_stream()
        self._last_sample_index = int(sample_indices[-1])
        self._raw = np.concatenate((self._raw, block), axis=0)
        if len(self._raw) > self.config.buffer_samples:
            self._raw = self._raw[-self.config.buffer_samples:]
        if self._calibration_target:
            self._calibration = np.concatenate((self._calibration, block), axis=0)
            if len(self._calibration) > self._calibration_target:
                self._calibration = self._calibration[:self._calibration_target]
        return discontinuity

    def predict(self, threshold: float | np.ndarray | tuple[float, ...] | None = None) -> PredictionFrame:
        if not self.ready_for_prediction or self._last_sample_index is None:
            raise RuntimeError("实时缓冲或校准尚未就绪")
        started = time.perf_counter()
        assert self.scale_counts_per_unit is not None
        model_input = fixed_lag_model_window(
            self._raw,
            self.config.sample_rate,
            self.scale_counts_per_unit,
            self.config.model_window_samples,
            self.config.lag_samples,
            expected_channels=self.config.channels,
        )
        probabilities = self.model.predict(model_input)
        input_end = self._last_sample_index + 1 - self.config.lag_samples
        input_start = input_end - self.config.model_window_samples
        output_indices = (
            input_start + self.config.left_context_samples
            + np.arange(probabilities.shape[1], dtype=np.int64) * self.config.stride_samples
        )
        if self._last_scanned_output_index is None:
            scan_start = max(0, len(output_indices) - 2)
        else:
            scan_start = int(np.searchsorted(
                output_indices, self._last_scanned_output_index, side="right"))
        scan_probabilities = probabilities[:, scan_start:]
        scan_indices = output_indices[scan_start:]
        events: list[DetectedEvent] = []
        if len(scan_indices):
            (events, self._previous_probabilities, self._last_event_index,
             self._last_event_name) = (
                detect_threshold_events(
                    scan_probabilities,
                    scan_indices,
                    self.labels,
                    self.display_names,
                    self.config.threshold if threshold is None else threshold,
                    self._previous_probabilities,
                    round(self.config.debounce_seconds * self.config.sample_rate),
                    self._last_event_index,
                    self._last_event_name,
                )
            )
            events = self._state_machine.filter(events)
            self._last_scanned_output_index = int(scan_indices[-1])
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return PredictionFrame(
            probabilities=probabilities[:, -1].copy(),
            labels=self.labels,
            events=tuple(events),
            output_sample_index=int(output_indices[-1]),
            output_age_ms=(self._last_sample_index - int(output_indices[-1]))
            * 1000.0 / self.config.sample_rate,
            fixed_lag_ms=self.config.fixed_lag_seconds * 1000.0,
            inference_ms=elapsed_ms,
            scale_counts_per_unit=self.scale_counts_per_unit,
        )
