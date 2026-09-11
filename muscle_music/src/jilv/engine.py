from __future__ import annotations

from dataclasses import dataclass
import random
import threading
import time

from .music import HarmonyPlan, PerformanceProfile, Phrase, PhraseFactory, new_harmony
from .protocol import (
    DIRECTION_TO_LAYER,
    EventKind,
    Gesture,
    GestureEvent,
    GestureFrame,
    GestureStateMachine,
    Layer,
)
from .sequencer import Sequencer, TransportState
from .visual import VisualPublisher


@dataclass(frozen=True, slots=True)
class EngineSnapshot:
    transport: TransportState
    stable_gesture: Gesture
    harmony: HarmonyPlan
    generations: dict[Layer, int]
    active_layers: tuple[Layer, ...]
    pending_layers: tuple[Layer, ...]


class MusicEngine:
    def __init__(
        self,
        sequencer: Sequencer,
        visual: VisualPublisher,
        *,
        seed: int | None = None,
        stable_ms: int = 120,
        input_timeout_ms: int = 1000,
    ) -> None:
        self.sequencer = sequencer
        self.visual = visual
        self.session_seed = int(seed if seed is not None else time.time_ns() & 0x7FFFFFFF)
        self._rng = random.Random(self.session_seed)
        self.harmony = new_harmony(self.session_seed)
        self.factory = PhraseFactory()
        self.gestures = GestureStateMachine(stable_ms=stable_ms)
        self.input_timeout_ms = int(input_timeout_ms)
        self.profiles = {layer: PerformanceProfile() for layer in Layer}
        self.global_profile = PerformanceProfile()
        self.generations = {layer: -1 for layer in Layer}
        self.layer_seeds = {layer: self._rng.randrange(1, 2**31 - 1) for layer in Layer}
        self.last_frame_wall_time = time.monotonic()
        self.last_frame_timestamp_ms = 0
        self._session_finished = False
        self._lock = threading.RLock()
        self.visual.harmony(self.harmony)

    def process_frame(self, frame: GestureFrame) -> None:
        with self._lock:
            self.last_frame_wall_time = time.monotonic()
            self.last_frame_timestamp_ms = frame.timestamp_ms
            self.visual.gesture(frame)
            self._update_continuous_effort(frame)
            for event in self.gestures.update(frame):
                self._handle_event(event)

    def _update_continuous_effort(self, frame: GestureFrame) -> None:
        """EMG continuously shapes existing loops; it never toggles transport."""
        self.global_profile = self.global_profile.blended(
            motion=self.global_profile.motion,
            muscle=frame.emg_activation,
            hold_ms=None,
            weight=0.08,
        )
        sounding = set(self.sequencer.active) | set(self.sequencer.pending)
        for layer in sounding:
            profile = self.profiles[layer].blended(
                motion=self.profiles[layer].motion,
                muscle=frame.emg_activation,
                hold_ms=None,
                weight=0.10,
            )
            self.profiles[layer] = profile
            # IMU chooses/develops the layer; calibrated EMG alone controls
            # expression, rhythmic density and brightness while it loops.
            self.sequencer.update_control(
                layer, frame.emg_activation, frame.emg_activation
            )

    def check_input_timeout(self) -> bool:
        with self._lock:
            elapsed_ms = (time.monotonic() - self.last_frame_wall_time) * 1000.0
            if elapsed_ms < self.input_timeout_ms:
                return False
            timestamp = max(
                self.last_frame_timestamp_ms + self.input_timeout_ms,
                int(time.time() * 1000),
            )
            events = self.gestures.force_none(timestamp)
            for event in events:
                self._handle_event(event)
            return bool(events)

    def _handle_event(self, event: GestureEvent) -> None:
        gesture = event.gesture
        if event.kind == EventKind.EXIT:
            if gesture in DIRECTION_TO_LAYER:
                layer = DIRECTION_TO_LAYER[gesture]
                profile = self.profiles[layer]
                self.profiles[layer] = profile.blended(
                    motion=profile.motion,
                    muscle=profile.muscle,
                    hold_ms=event.held_ms,
                    weight=0.30,
                )
            return

        if gesture in DIRECTION_TO_LAYER:
            self._handle_direction(event, DIRECTION_TO_LAYER[gesture])
            return

        if gesture == Gesture.PINCH_INDEX and event.kind == EventKind.ENTER:
            self._toggle_session()

    def _handle_direction(self, event: GestureEvent, layer: Layer) -> None:
        if event.kind == EventKind.ENTER:
            profile = self.profiles[layer].blended(
                motion=event.frame.motion_energy,
                muscle=event.frame.emg_activation,
                hold_ms=None,
                weight=0.34,
            )
            self.profiles[layer] = profile
            self.global_profile = self.global_profile.blended(
                motion=event.frame.motion_energy,
                muscle=event.frame.emg_activation,
                hold_ms=None,
                weight=0.16,
            )
            self.sequencer.update_control(
                layer, event.frame.emg_activation, event.frame.emg_activation
            )
            self.generations[layer] += 1
            phrase = self._make_phrase(layer)
            self.sequencer.queue_phrase(phrase, boundary="bar")
            return

        if event.kind == EventKind.HOLD:
            profile = self.profiles[layer].blended(
                motion=event.frame.motion_energy,
                muscle=event.frame.emg_activation,
                hold_ms=event.held_ms,
                weight=0.06,
            )
            self.profiles[layer] = profile
            self.global_profile = self.global_profile.blended(
                motion=event.frame.motion_energy,
                muscle=event.frame.emg_activation,
                hold_ms=event.held_ms,
                weight=0.025,
            )
            self.sequencer.update_control(
                layer, event.frame.emg_activation, event.frame.emg_activation
            )
            self._publish_personality()

    def _make_phrase(self, layer: Layer) -> Phrase:
        effective_profile = self.profiles[layer].mixed(self.global_profile, 0.3)
        phrase = self.factory.generate(
            layer=layer,
            harmony=self.harmony,
            profile=effective_profile,
            seed=self.layer_seeds[layer],
            generation=self.generations[layer],
        )
        self._publish_personality()
        return phrase

    def _publish_personality(self) -> None:
        profile = self.global_profile
        movement = "舒展" if profile.motion < 0.38 else "跃动" if profile.motion > 0.68 else "流动"
        tone = "柔和" if profile.muscle < 0.38 else "明亮" if profile.muscle > 0.68 else "温润"
        developed = [max(0, value) for value in self.generations.values()]
        complexity = min(1.0, sum(developed) / 12.0)
        self.visual.personality(
            f"{tone}·{movement}",
            profile.motion,
            profile.muscle,
            profile.legato,
            complexity,
        )

    def _toggle_session(self) -> None:
        if self.sequencer.state in (TransportState.PLAYING, TransportState.COUNT_IN):
            self.sequencer.finish()
            self._session_finished = True
            self.visual.variation("session_complete")
            return
        if self._session_finished:
            self._new_session()
            self._session_finished = False
        self.sequencer.request_play()

    def _new_session(self) -> None:
        self.session_seed = self._rng.randrange(1, 2**31 - 1)
        self.harmony = new_harmony(self.session_seed)
        self.profiles = {layer: PerformanceProfile() for layer in Layer}
        self.global_profile = PerformanceProfile()
        self.generations = {layer: -1 for layer in Layer}
        self.layer_seeds = {layer: self._rng.randrange(1, 2**31 - 1) for layer in Layer}
        self.sequencer.clear()
        self.visual.harmony(self.harmony)
        self.visual.variation("new_session")

    def snapshot(self) -> EngineSnapshot:
        with self._lock:
            return EngineSnapshot(
                transport=self.sequencer.state,
                stable_gesture=self.gestures.stable,
                harmony=self.harmony,
                generations=dict(self.generations),
                active_layers=tuple(sorted(self.sequencer.active)),
                pending_layers=tuple(sorted(self.sequencer.pending)),
            )


def now_ms() -> int:
    return int(time.time() * 1000)
