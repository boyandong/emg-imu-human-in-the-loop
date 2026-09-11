from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .music import HarmonyPlan, Phrase
from .osc import OscClient
from .protocol import Gesture, GestureFrame, Layer


class VisualPublisher(Protocol):
    def gesture(self, frame: GestureFrame) -> None: ...

    def transport(self, state: str, countdown: int, bpm: float) -> None: ...

    def beat(self, bar: int, beat: int, step: int) -> None: ...

    def layer(self, phrase: Phrase, active: bool, energy: float) -> None: ...

    def harmony(self, plan: HarmonyPlan) -> None: ...

    def personality(
        self, label: str, motion: float, muscle: float, legato: float, complexity: float
    ) -> None: ...

    def variation(self, kind: str) -> None: ...

    def reset(self) -> None: ...

    def error(self, message: str) -> None: ...

    def close(self) -> None: ...


class NullVisualPublisher:
    def gesture(self, frame: GestureFrame) -> None:
        pass

    def transport(self, state: str, countdown: int, bpm: float) -> None:
        pass

    def beat(self, bar: int, beat: int, step: int) -> None:
        pass

    def layer(self, phrase: Phrase, active: bool, energy: float) -> None:
        pass

    def harmony(self, plan: HarmonyPlan) -> None:
        pass

    def personality(
        self, label: str, motion: float, muscle: float, legato: float, complexity: float
    ) -> None:
        pass

    def variation(self, kind: str) -> None:
        pass

    def reset(self) -> None:
        pass

    def error(self, message: str) -> None:
        pass

    def close(self) -> None:
        pass


@dataclass(frozen=True, slots=True)
class VisualEvent:
    address: str
    values: tuple[object, ...]


class MemoryVisualPublisher(NullVisualPublisher):
    def __init__(self) -> None:
        self.events: list[VisualEvent] = []

    def _add(self, address: str, *values: object) -> None:
        self.events.append(VisualEvent(address, values))

    def gesture(self, frame: GestureFrame) -> None:
        self._add(
            "/music/gesture",
            int(frame.gesture),
            frame.confidence,
            frame.motion_energy,
            frame.emg_activation,
        )

    def transport(self, state: str, countdown: int, bpm: float) -> None:
        self._add("/music/transport", state, countdown, bpm)

    def beat(self, bar: int, beat: int, step: int) -> None:
        self._add("/music/beat", bar, beat, step)

    def layer(self, phrase: Phrase, active: bool, energy: float) -> None:
        self._add(
            "/music/layer",
            int(phrase.layer),
            int(active),
            phrase.generation,
            phrase.evolution_stage,
            energy,
            phrase.change_label,
        )

    def harmony(self, plan: HarmonyPlan) -> None:
        self._add("/music/harmony", plan.root_name, plan.mode_name)

    def personality(
        self, label: str, motion: float, muscle: float, legato: float, complexity: float
    ) -> None:
        self._add("/music/personality", label, motion, muscle, legato, complexity)

    def variation(self, kind: str) -> None:
        self._add("/music/variation", kind)

    def reset(self) -> None:
        self._add("/music/reset", 1)

    def error(self, message: str) -> None:
        self._add("/music/error", message)


class OscVisualPublisher(NullVisualPublisher):
    def __init__(self, host: str, port: int) -> None:
        self.client = OscClient(host, port)

    def gesture(self, frame: GestureFrame) -> None:
        self.client.send(
            "/music/gesture",
            int(frame.gesture),
            frame.confidence,
            frame.motion_energy,
            frame.emg_activation,
        )

    def transport(self, state: str, countdown: int, bpm: float) -> None:
        self.client.send("/music/transport", state, countdown, bpm)

    def beat(self, bar: int, beat: int, step: int) -> None:
        self.client.send("/music/beat", bar, beat, step)

    def layer(self, phrase: Phrase, active: bool, energy: float) -> None:
        self.client.send(
            "/music/layer",
            int(phrase.layer),
            int(active),
            phrase.generation,
            phrase.evolution_stage,
            energy,
            phrase.change_label,
        )

    def harmony(self, plan: HarmonyPlan) -> None:
        self.client.send("/music/harmony", plan.root_name, plan.mode_name)

    def personality(
        self, label: str, motion: float, muscle: float, legato: float, complexity: float
    ) -> None:
        self.client.send(
            "/music/personality", label, motion, muscle, legato, complexity
        )

    def variation(self, kind: str) -> None:
        self.client.send("/music/variation", kind)

    def reset(self) -> None:
        self.client.send("/music/reset", 1)

    def error(self, message: str) -> None:
        self.client.send("/music/error", message)

    def close(self) -> None:
        self.client.close()


LAYER_LABELS: dict[Layer, str] = {
    Layer.DRUMS: "鼓与节奏",
    Layer.TEXTURE: "环境纹理",
    Layer.CHORDS: "和弦铺底",
    Layer.LEAD: "主旋律",
    Layer.ARP: "高音琶音",
    Layer.BASS: "贝斯",
}


GESTURE_LABELS: dict[Gesture, str] = {
    Gesture.NONE: "无动作",
    Gesture.FORWARD: "前",
    Gesture.BACKWARD: "后",
    Gesture.LEFT: "左",
    Gesture.RIGHT: "右",
    Gesture.UP: "上",
    Gesture.DOWN: "下",
    Gesture.PINCH_INDEX: "食指捏合",
    Gesture.PINCH_MIDDLE: "保留状态",
}
