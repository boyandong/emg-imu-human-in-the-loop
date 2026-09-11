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
    BREAK = "BREAK"
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
    trial_repeat_requested = Signal(int, str)
    gesture_cued = Signal(int, str)
    block_break_started = Signal(int, int)
    block_break_ended = Signal(int, int)
    finished = Signal()

    def __init__(self, parent: QObject | None = None, seed: int | None = None) -> None:
        super().__init__(parent)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._advance)
        self._rng = random.Random(seed)
        self.config: ProtocolConfig | None = None
        self.sequence: list[str] = []
        self.onset_offsets: list[int] = []
        self.trial_kinds: list[str] = []
        self.block_indices: list[int] = []
        self.repeat_of_trial_ids: list[int] = []
        self.attempts: list[int] = []
        self.position = -1
        self.state = PromptState.IDLE
        self._deadline = 0.0
        self._remaining_ms = 0
        self._paused_state = PromptState.IDLE
        self._repeat_after_current = False
        self._after_block_break = False
        self._scheduled_cues: dict[tuple[int, str], int] = {}

    @property
    def current_label(self) -> str:
        return self.sequence[self.position] if 0 <= self.position < len(self.sequence) else ""

    @property
    def scheduled_deadline_monotonic_ns(self) -> int:
        """Planned transition deadline used for timer-lateness diagnostics."""
        return round(self._deadline * 1_000_000_000)

    @property
    def current_onset_offset_ms(self) -> int:
        return self.onset_offsets[self.position] if 0 <= self.position < len(self.onset_offsets) else 0

    @property
    def current_trial_kind(self) -> str:
        return self.trial_kinds[self.position] if 0 <= self.position < len(self.trial_kinds) else "formal"

    @property
    def current_block_index(self) -> int:
        return self.block_indices[self.position] if 0 <= self.position < len(self.block_indices) else 0

    @property
    def current_rerecord_of_trial_id(self) -> int:
        return self.repeat_of_trial_ids[self.position] if 0 <= self.position < len(self.repeat_of_trial_ids) else -1

    @property
    def current_attempt(self) -> int:
        return self.attempts[self.position] if 0 <= self.position < len(self.attempts) else 1

    def scheduled_cue_monotonic_ns(self, trial_id: int, name: str) -> int:
        return self._scheduled_cues.get((trial_id, name), -1)

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
            sequence = [label for label in config.labels
                        for _ in range(config.trial_count(label))]
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
        if config.formal_collection:
            calibration = [str(block["name"]) for block in config.calibration_blocks]
            formal_pairs = self._balanced_formal_pairs(config)
            sequence = calibration + [label for label, _offset, _block in formal_pairs]
            self.onset_offsets = [0] * len(calibration) + [offset for _, offset, _ in formal_pairs]
            self.trial_kinds = ["calibration"] * len(calibration) + ["formal"] * len(formal_pairs)
            self.block_indices = [0] * len(calibration) + [block for _, _, block in formal_pairs]
        else:
            self.onset_offsets = [self._rng.choice(config.onset_offsets_ms or [0])
                                  for _ in sequence]
            self.trial_kinds = [
                "null" if config.null_kind(label) else "formal" for label in sequence
            ]
            self.block_indices = [0] * len(sequence)
        self.sequence = sequence
        self.repeat_of_trial_ids = [-1] * len(sequence)
        self.attempts = [1] * len(sequence)
        self.position = -1
        self.state = PromptState.IDLE
        self._after_block_break = False
        self._scheduled_cues.clear()
        return list(sequence)

    def _balanced_formal_pairs(self, config: ProtocolConfig) -> list[tuple[str, int, int]]:
        """Build block-constrained trials with per-label balanced onset offsets."""
        offsets = config.onset_offsets_ms or [0]
        pairs: list[tuple[str, int]] = []
        for label in config.labels:
            count = config.trial_count(label)
            assigned = [offsets[index % len(offsets)] for index in range(count)]
            if config.randomize:
                self._rng.shuffle(assigned)
            pairs.extend((label, offset) for offset in assigned)

        block_size = config.block_size or len(pairs)
        stationary = [pair for pair in pairs if pair[0].startswith("still_")]
        moving = [pair for pair in pairs if not pair[0].startswith("still_")]
        if config.randomize:
            self._rng.shuffle(stationary)
            self._rng.shuffle(moving)

        result: list[tuple[str, int, int]] = []
        if stationary and len(stationary) == len(moving) and block_size % 2 == 0:
            half = block_size // 2
            block = 1
            while stationary or moving:
                chunk = stationary[:half] + moving[:half]
                del stationary[:half]; del moving[:half]
                if config.randomize:
                    self._rng.shuffle(chunk)
                result.extend((label, offset, block) for label, offset in chunk)
                block += 1
        else:
            if config.randomize:
                self._rng.shuffle(pairs)
            result = [(label, offset, index // block_size + 1)
                      for index, (label, offset) in enumerate(pairs)]
        return result

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
            if not self._repeat_after_current:
                self._repeat_after_current = True
                self.trial_repeat_requested.emit(self.position + 1, self.current_label)

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
                self._emit_gesture_cue(
                    self.position + 1,
                    "index_press" if self.current_label == "index_hold" else "middle_press",
                )
                self._schedule(self._rng.uniform(
                    self.config.hold_min_sec, self.config.hold_max_sec))
            elif self.current_label.endswith("_open_hand"):
                self._set_state(PromptState.HOLD)
                self._emit_combination_cues()
                self._schedule(self.config.prompt_duration(self.current_label))
            else:
                self._set_state(PromptState.PROMPT)
                if self.config.calibration_block(self.current_label) is not None:
                    self._emit_gesture_cue(self.position + 1, self.current_label)
                elif self.config.null_kind(self.current_label) is None:
                    self._emit_combination_cues()
                self._schedule(self.config.prompt_duration(self.current_label))
        elif self.state == PromptState.PROMPT:
            self.prompt_ended.emit(self.position + 1, self.current_label)
            self.trial_ended.emit(self.position + 1, self.current_label)
            self._begin_next_trial()
        elif self.state == PromptState.HOLD:
            self._set_state(PromptState.RELEASE)
            if self.current_label.endswith("_open_hand"):
                self._emit_gesture_cue(self.position + 1, "hand:release")
            else:
                self._emit_gesture_cue(
                    self.position + 1,
                    "index_release" if self.current_label == "index_hold" else "middle_release",
                )
            self._schedule(self.config.release_display_sec)
        elif self.state == PromptState.RELEASE:
            self.prompt_ended.emit(self.position + 1, self.current_label)
            self.trial_ended.emit(self.position + 1, self.current_label)
            self._begin_next_trial()
        elif self.state == PromptState.BREAK:
            block = self.current_block_index
            total = max(self.block_indices, default=0)
            self.block_break_ended.emit(block, total)
            self._after_block_break = True
            self._begin_next_trial()

    def _begin_next_trial(self) -> None:
        assert self.config is not None
        if self._repeat_after_current and self.current_label:
            self.sequence.insert(self.position + 1, self.current_label)
            self.onset_offsets.insert(self.position + 1, self.current_onset_offset_ms)
            self.trial_kinds.insert(self.position + 1, self.current_trial_kind)
            self.block_indices.insert(self.position + 1, self.current_block_index)
            root_id = (self.current_rerecord_of_trial_id
                       if self.current_rerecord_of_trial_id >= 0 else self.position + 1)
            self.repeat_of_trial_ids.insert(self.position + 1, root_id)
            self.attempts.insert(self.position + 1, self.current_attempt + 1)
            self._repeat_after_current = False
        next_position = self.position + 1
        if (not self._after_block_break and self.config.block_break_sec > 0
                and 0 <= self.position < len(self.sequence)
                and next_position < len(self.sequence)
                and self.current_trial_kind == "formal"
                and self.trial_kinds[next_position] == "formal"
                and self.current_block_index != self.block_indices[next_position]):
            block = self.current_block_index
            total = max(self.block_indices, default=0)
            self._set_state(PromptState.BREAK)
            self.block_break_started.emit(block, total)
            self._schedule(self.config.block_break_sec)
            return
        self._after_block_break = False
        self.position = next_position
        if self.position >= len(self.sequence):
            self._set_state(PromptState.FINISHED)
            self.finished.emit()
            return
        label = self.current_label
        self.trial_started.emit(self.position + 1, label)
        self._set_state(PromptState.REST)
        self.rest_started.emit(self.position + 1, label)
        rest_sec = (0.0 if self.current_trial_kind == "calibration" else
                    self._rng.uniform(self.config.rest_min_sec, self.config.rest_max_sec))
        self._schedule(rest_sec)

    def _schedule(self, seconds: float) -> None:
        milliseconds = max(0, round(seconds * 1000))
        self._deadline = time.monotonic() + milliseconds / 1000
        self.phase_scheduled.emit(self.state, milliseconds / 1000.0)
        self._timer.start(milliseconds)

    def _set_state(self, state: PromptState) -> None:
        self.state = state
        self.state_changed.emit(state, self.current_label, self.position + 1, len(self.sequence))

    def _emit_combination_cues(self) -> None:
        parts = self.current_label.split("_", 1)
        if len(parts) != 2 or parts[0] not in {
                "still", "up", "down", "left", "right", "forward", "backward"}:
            self._emit_gesture_cue(self.position + 1, self.current_label)
            return
        arm, hand = parts
        offset = self.current_onset_offset_ms
        first, second = ((f"hand:{hand}", f"arm:{arm}") if offset < 0
                         else (f"arm:{arm}", f"hand:{hand}"))
        if offset == 0:
            scheduled_ns = time.monotonic_ns()
            self._emit_gesture_cue(self.position + 1, f"arm:{arm}", scheduled_ns)
            self._emit_gesture_cue(self.position + 1, f"hand:{hand}", scheduled_ns)
        else:
            scheduled_ns = time.monotonic_ns()
            self._emit_gesture_cue(self.position + 1, first, scheduled_ns)
            trial_id = self.position + 1
            second_scheduled_ns = scheduled_ns + abs(offset) * 1_000_000
            QTimer.singleShot(abs(offset), lambda cue=second, tid=trial_id, planned=second_scheduled_ns:
                              self._emit_gesture_cue(tid, cue, planned))

    def _emit_gesture_cue(self, trial_id: int, name: str,
                          scheduled_ns: int | None = None) -> None:
        planned = time.monotonic_ns() if scheduled_ns is None else scheduled_ns
        self._scheduled_cues[(trial_id, name)] = planned
        self.gesture_cued.emit(trial_id, name)
