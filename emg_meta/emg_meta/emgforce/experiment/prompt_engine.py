from __future__ import annotations

import random
import time
from enum import Enum

from PySide6.QtCore import QObject, QTimer, Signal

from .models import ProtocolConfig


class PromptState(str, Enum):
    IDLE = "IDLE"
    COUNTDOWN = "COUNTDOWN"
    REST = "REST"
    PROMPT = "PROMPT"
    HOLD = "HOLD"
    RELEASE = "RELEASE"
    PAUSED = "PAUSED"
    FINISHED = "FINISHED"


class PromptEngine(QObject):
    """Timer-only protocol state machine. It has no acquisition dependency."""

    state_changed = Signal(object, str, int, int)
    phase_scheduled = Signal(object, float)
    trial_started = Signal(int, str)
    rest_started = Signal(int, str)
    rest_ended = Signal(int, str)
    prompt_started = Signal(int, str)
    prompt_ended = Signal(int, str)
    trial_ended = Signal(int, str)
    trial_skipped = Signal(int, str)
    gesture_cued = Signal(int, str)
    finished = Signal()

    def __init__(self, parent: QObject | None = None, seed: int | None = None) -> None:
        super().__init__(parent)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._advance)
        self._rng = random.Random(seed)
        self.config: ProtocolConfig | None = None
        self.sequence: list[str] = []
        self.position = -1
        self.state = PromptState.IDLE
        self._deadline = 0.0
        self._remaining_ms = 0
        self._paused_state = PromptState.IDLE
        self._repeat_after_current = False

    @property
    def current_label(self) -> str:
        return self.sequence[self.position] if 0 <= self.position < len(self.sequence) else ""

    @property
    def scheduled_deadline_monotonic_ns(self) -> int:
        """Planned transition deadline used for timer-lateness diagnostics."""
        return round(self._deadline * 1_000_000_000)

    def prepare(self, config: ProtocolConfig) -> list[str]:
        config.validate()
        discrete_labels = {
            "thumb_tap", "thumb_swipe_left", "thumb_swipe_right",
            "thumb_swipe_up", "thumb_swipe_down", "index_hold", "middle_hold",
        }
        if set(config.labels) == discrete_labels:
            # A route is one balanced collection block: four navigation gestures
            # and three activation gestures. Each label still occurs exactly
            # trials_per_class times in the HDF5 trial table.
            clockwise = [
                "thumb_swipe_right", "thumb_swipe_down",
                "thumb_swipe_left", "thumb_swipe_up",
            ]
            counter_clockwise = [
                "thumb_swipe_right", "thumb_swipe_up",
                "thumb_swipe_left", "thumb_swipe_down",
            ]
            sequence: list[str] = []
            for route_index in range(config.trials_per_class):
                navigation = list(self._rng.choice((clockwise, counter_clockwise)))
                if config.randomize:
                    shift = self._rng.randrange(len(navigation))
                    navigation = navigation[shift:] + navigation[:shift]
                activation = ["thumb_tap", "index_hold", "middle_hold"]
                if config.randomize:
                    self._rng.shuffle(activation)
                for step in range(3):
                    sequence.extend((navigation[step], activation[step]))
                sequence.append(navigation[3])
        else:
            sequence = [
                label for label in config.labels for _ in range(config.trial_count(label))
            ]
            if config.randomize:
                self._rng.shuffle(sequence)
        timed_null = [
            label
            for label in config.timed_null_actions
            for _ in range(config.timed_null_repetitions)
        ]
        if config.randomize:
            self._rng.shuffle(timed_null)
        # Meta collected specifically timed null gestures and longer-form null
        # behaviours in dedicated stages. Keep those stages after target trials.
        sequence.extend(timed_null)
        sequence.extend(str(block["name"]) for block in config.continuous_null_blocks)
        self.config = config
        self.sequence = sequence
        self.position = -1
        self.state = PromptState.IDLE
        return list(sequence)

    def start(self) -> None:
        if self.config is None or not self.sequence:
            raise RuntimeError("请先加载实验协议")
        self.position = -1
        self._set_state(PromptState.COUNTDOWN)
        self._schedule(self.config.countdown_sec)

    def pause(self) -> None:
        if self.state in {PromptState.IDLE, PromptState.PAUSED, PromptState.FINISHED}:
            return
        self._remaining_ms = max(0, int((self._deadline - time.monotonic()) * 1000))
        self._timer.stop()
        self._paused_state = self.state
        self._set_state(PromptState.PAUSED)

    def resume(self) -> None:
        if self.state != PromptState.PAUSED:
            return
        self._set_state(self._paused_state)
        self._timer.start(self._remaining_ms)
        self._deadline = time.monotonic() + self._remaining_ms / 1000

    def skip(self) -> None:
        active_state = self._paused_state if self.state == PromptState.PAUSED else self.state
        if active_state in {PromptState.REST, PromptState.PROMPT,
                            PromptState.HOLD, PromptState.RELEASE}:
            self._timer.stop()
            if active_state in {PromptState.PROMPT, PromptState.HOLD, PromptState.RELEASE}:
                self.prompt_ended.emit(self.position + 1, self.current_label)
            else:
                self.rest_ended.emit(self.position + 1, self.current_label)
            self.trial_skipped.emit(self.position + 1, self.current_label)
            self.trial_ended.emit(self.position + 1, self.current_label)
            self._begin_next_trial()

    def repeat(self) -> None:
        if self.state in {PromptState.REST, PromptState.PROMPT, PromptState.HOLD,
                          PromptState.RELEASE, PromptState.PAUSED}:
            self._repeat_after_current = True

    def stop(self) -> None:
        self._timer.stop()
        if self.state == PromptState.FINISHED:
            return
        self.state = PromptState.IDLE
        self.state_changed.emit(self.state, "", self.position + 1, len(self.sequence))


    def _advance(self) -> None:
        assert self.config is not None
        if self.state == PromptState.COUNTDOWN:
            self._begin_next_trial()
        elif self.state == PromptState.REST:
            self.rest_ended.emit(self.position + 1, self.current_label)
            self.prompt_started.emit(self.position + 1, self.current_label)
            if self.current_label in {"index_hold", "middle_hold"}:
                self._set_state(PromptState.HOLD)
                self.gesture_cued.emit(
                    self.position + 1,
                    "index_press" if self.current_label == "index_hold" else "middle_press",
                )
                self._schedule(self._rng.uniform(
                    self.config.hold_min_sec, self.config.hold_max_sec))
            else:
                self._set_state(PromptState.PROMPT)
                if self.config.null_kind(self.current_label) is None:
                    self.gesture_cued.emit(self.position + 1, self.current_label)
                self._schedule(self.config.prompt_duration(self.current_label))
        elif self.state == PromptState.PROMPT:
            self.prompt_ended.emit(self.position + 1, self.current_label)
            self.trial_ended.emit(self.position + 1, self.current_label)
            self._begin_next_trial()
        elif self.state == PromptState.HOLD:
            self._set_state(PromptState.RELEASE)
            self.gesture_cued.emit(
                self.position + 1,
                "index_release" if self.current_label == "index_hold" else "middle_release",
            )
            self._schedule(self.config.release_display_sec)
        elif self.state == PromptState.RELEASE:
            self.prompt_ended.emit(self.position + 1, self.current_label)
            self.trial_ended.emit(self.position + 1, self.current_label)
            self._begin_next_trial()

    def _begin_next_trial(self) -> None:
        assert self.config is not None
        if self._repeat_after_current and self.current_label:
            self.sequence.insert(self.position + 1, self.current_label)
            self._repeat_after_current = False
        self.position += 1
        if self.position >= len(self.sequence):
            self._set_state(PromptState.FINISHED)
            self.finished.emit()
            return
        label = self.current_label
        self.trial_started.emit(self.position + 1, label)
        self._set_state(PromptState.REST)
        self.rest_started.emit(self.position + 1, label)
        self._schedule(self._rng.uniform(self.config.rest_min_sec, self.config.rest_max_sec))

    def _schedule(self, seconds: float) -> None:
        milliseconds = max(0, round(seconds * 1000))
        self._deadline = time.monotonic() + milliseconds / 1000
        self.phase_scheduled.emit(self.state, milliseconds / 1000.0)
        self._timer.start(milliseconds)

    def _set_state(self, state: PromptState) -> None:
        self.state = state
        self.state_changed.emit(state, self.current_label, self.position + 1, len(self.sequence))
