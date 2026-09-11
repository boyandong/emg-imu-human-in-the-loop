from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Iterable


class Gesture(IntEnum):
    NONE = 0
    FORWARD = 1
    BACKWARD = 2
    LEFT = 3
    RIGHT = 4
    UP = 5
    DOWN = 6
    PINCH_INDEX = 7
    PINCH_MIDDLE = 8


class Layer(IntEnum):
    DRUMS = 0
    TEXTURE = 1
    CHORDS = 2
    LEAD = 3
    ARP = 4
    BASS = 5


DIRECTION_TO_LAYER: dict[Gesture, Layer] = {
    Gesture.FORWARD: Layer.DRUMS,
    Gesture.BACKWARD: Layer.TEXTURE,
    Gesture.LEFT: Layer.CHORDS,
    Gesture.RIGHT: Layer.LEAD,
    Gesture.UP: Layer.ARP,
    Gesture.DOWN: Layer.BASS,
}


class EventKind(str, Enum):
    ENTER = "enter"
    HOLD = "hold"
    EXIT = "exit"


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True, slots=True)
class GestureFrame:
    timestamp_ms: int
    gesture: Gesture
    confidence: float
    motion_energy: float
    emg_activation: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_ms", int(self.timestamp_ms))
        object.__setattr__(self, "gesture", Gesture(self.gesture))
        object.__setattr__(self, "confidence", clamp01(self.confidence))
        object.__setattr__(self, "motion_energy", clamp01(self.motion_energy))
        object.__setattr__(self, "emg_activation", clamp01(self.emg_activation))

    @property
    def combined_energy(self) -> float:
        return clamp01(0.55 * self.motion_energy + 0.45 * self.emg_activation)


@dataclass(frozen=True, slots=True)
class GestureEvent:
    kind: EventKind
    gesture: Gesture
    timestamp_ms: int
    held_ms: int
    frame: GestureFrame


class GestureStateMachine:
    """Debounces a continuous gesture stream without requiring a NONE reset."""

    def __init__(self, stable_ms: int = 120, initial: Gesture = Gesture.NONE) -> None:
        if stable_ms < 0:
            raise ValueError("stable_ms must be non-negative")
        self.stable_ms = int(stable_ms)
        self.stable = Gesture(initial)
        self.candidate = Gesture(initial)
        self.candidate_since_ms = 0
        self.stable_since_ms = 0
        self.last_timestamp_ms = -1
        self.last_frame = GestureFrame(0, initial, 1.0, 0.0, 0.0)

    def update(self, frame: GestureFrame) -> list[GestureEvent]:
        if frame.timestamp_ms < self.last_timestamp_ms:
            return []

        self.last_timestamp_ms = frame.timestamp_ms
        self.last_frame = frame

        if frame.gesture != self.candidate:
            self.candidate = frame.gesture
            self.candidate_since_ms = frame.timestamp_ms

        if self.candidate != self.stable:
            candidate_age = frame.timestamp_ms - self.candidate_since_ms
            if candidate_age < self.stable_ms:
                return []

            held_ms = max(0, frame.timestamp_ms - self.stable_since_ms)
            old = self.stable
            self.stable = self.candidate
            self.stable_since_ms = frame.timestamp_ms
            return [
                GestureEvent(EventKind.EXIT, old, frame.timestamp_ms, held_ms, frame),
                GestureEvent(EventKind.ENTER, self.stable, frame.timestamp_ms, 0, frame),
            ]

        held_ms = max(0, frame.timestamp_ms - self.stable_since_ms)
        return [GestureEvent(EventKind.HOLD, self.stable, frame.timestamp_ms, held_ms, frame)]

    def force_none(self, timestamp_ms: int) -> list[GestureEvent]:
        if self.stable == Gesture.NONE and self.candidate == Gesture.NONE:
            return []
        frame = GestureFrame(timestamp_ms, Gesture.NONE, 1.0, 0.0, 0.0)
        self.candidate = Gesture.NONE
        self.candidate_since_ms = timestamp_ms - self.stable_ms
        return self.update(frame)


def gesture_names() -> Iterable[tuple[int, str]]:
    labels = {
        Gesture.NONE: "无动作 / none",
        Gesture.FORWARD: "前 / forward",
        Gesture.BACKWARD: "后 / backward",
        Gesture.LEFT: "左 / left",
        Gesture.RIGHT: "右 / right",
        Gesture.UP: "上 / up",
        Gesture.DOWN: "下 / down",
        Gesture.PINCH_INDEX: "拇指食指捏合 / index pinch",
        Gesture.PINCH_MIDDLE: "保留状态 / reserved",
    }
    for gesture in Gesture:
        yield int(gesture), labels[gesture]
