from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import threading
import time

from .midi import MidiSink
from .music import PHRASE_STEPS, STEPS_PER_BAR, STEPS_PER_BEAT, Phrase
from .protocol import Layer, clamp01
from .visual import VisualPublisher


class TransportState(str, Enum):
    STOPPED = "stopped"
    COUNT_IN = "count_in"
    PLAYING = "playing"
    PAUSED = "paused"


@dataclass(slots=True)
class PendingPhrase:
    phrase: Phrase
    target_step: int


@dataclass(slots=True)
class LayerControl:
    current_energy: float = 0.5
    target_energy: float = 0.5
    current_muscle: float = 0.5
    target_muscle: float = 0.5
    last_expression: int = -1
    last_brightness: int = -1


class Sequencer:
    def __init__(
        self,
        midi: MidiSink,
        visual: VisualPublisher,
        *,
        bpm: float = 112.0,
        count_in_beats: int = 1,
    ) -> None:
        if bpm <= 0:
            raise ValueError("bpm must be positive")
        self.midi = midi
        self.visual = visual
        self.bpm = float(bpm)
        self.count_in_steps_default = max(0, int(count_in_beats) * STEPS_PER_BEAT)
        self.state = TransportState.STOPPED
        self.music_step = 0
        self.count_in_remaining = 0
        self.active: dict[Layer, Phrase] = {}
        self.pending: dict[Layer, PendingPhrase] = {}
        self.controls = {layer: LayerControl() for layer in Layer}
        self._off_queue: dict[int, list[tuple[int, int, int]]] = {}
        self._note_tokens: dict[tuple[int, int], int] = {}
        self._token = 0
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def step_seconds(self) -> float:
        return 60.0 / self.bpm / STEPS_PER_BEAT

    def start_thread(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, name="jilv-sequencer", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        next_tick = time.perf_counter()
        while not self._stop_event.is_set():
            now = time.perf_counter()
            if now < next_tick:
                self._stop_event.wait(min(next_tick - now, 0.02))
                continue
            self.step_once()
            next_tick += self.step_seconds
            if next_tick < now - self.step_seconds:
                next_tick = now + self.step_seconds

    def stop_thread(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        with self._lock:
            self._silence_all()
            self.state = TransportState.STOPPED
            self.visual.transport(self.state.value, 0, self.bpm)

    def toggle(self) -> TransportState:
        with self._lock:
            if self.state in (TransportState.PLAYING, TransportState.COUNT_IN):
                self.pause()
            else:
                self.request_play()
            return self.state

    def request_play(self) -> None:
        with self._lock:
            self.count_in_remaining = self.count_in_steps_default
            if self.count_in_remaining:
                self.state = TransportState.COUNT_IN
                self.visual.transport(self.state.value, self.count_in_remaining, self.bpm)
            else:
                self.state = TransportState.PLAYING
                self.visual.transport(self.state.value, 0, self.bpm)

    def pause(self) -> None:
        with self._lock:
            self._silence_all()
            self.count_in_remaining = 0
            self.state = TransportState.PAUSED
            self.visual.transport(self.state.value, 0, self.bpm)

    def finish(self) -> None:
        with self._lock:
            self._silence_all()
            self.count_in_remaining = 0
            self.state = TransportState.STOPPED
            self.visual.transport("finished", 0, self.bpm)

    def queue_phrase(self, phrase: Phrase, *, boundary: str = "bar") -> int:
        with self._lock:
            if boundary not in {"bar", "cycle"}:
                raise ValueError("boundary must be 'bar' or 'cycle'")
            quantum = STEPS_PER_BAR if boundary == "bar" else PHRASE_STEPS
            if self.state == TransportState.STOPPED and self.music_step == 0:
                target = 0
            else:
                target = ((self.music_step // quantum) + 1) * quantum
            self.pending[phrase.layer] = PendingPhrase(phrase, target)
            self.visual.layer(phrase, False, phrase.profile.energy)
            return target

    def update_control(self, layer: Layer, energy: float, muscle: float) -> None:
        with self._lock:
            control = self.controls[layer]
            control.target_energy = clamp01(energy)
            control.target_muscle = clamp01(muscle)

    def clear(self) -> None:
        with self._lock:
            self._silence_all()
            self.active.clear()
            self.pending.clear()
            self.music_step = 0
            self.visual.reset()

    def step_once(self) -> None:
        with self._lock:
            if self.state == TransportState.COUNT_IN:
                self.count_in_remaining -= 1
                self.visual.transport(
                    self.state.value, max(0, self.count_in_remaining), self.bpm
                )
                if self.count_in_remaining <= 0:
                    self.state = TransportState.PLAYING
                    self.visual.transport(self.state.value, 0, self.bpm)
                return
            if self.state != TransportState.PLAYING:
                return

            self._activate_pending()
            self._send_due_note_offs()
            self._smooth_controls()

            phrase_step = self.music_step % PHRASE_STEPS
            for layer, phrase in self.active.items():
                channel = int(layer)
                for note in phrase.notes:
                    if note.step == phrase_step and self._should_play(layer, note.step, note.pitch):
                        self._note_on(channel, note.pitch, note.velocity, note.duration)

            bar = self.music_step // STEPS_PER_BAR
            beat = (self.music_step % STEPS_PER_BAR) // STEPS_PER_BEAT
            subdivision = self.music_step % STEPS_PER_BEAT
            self.visual.beat(bar, beat, subdivision)
            self.music_step += 1

    def _should_play(self, layer: Layer, phrase_step: int, pitch: int) -> bool:
        """Uses gesture energy to thin ornaments while preserving musical anchors."""
        if phrase_step % STEPS_PER_BEAT == 0:
            return True
        if layer == Layer.DRUMS and pitch in (36, 38):
            return True
        energy = self.controls[layer].current_energy
        density_budget = max(0.62, 1.0 - max(0, len(self.active) - 3) * 0.09)
        threshold = (0.22 + 0.78 * energy) * density_budget
        stable_value = ((phrase_step * 37 + pitch * 17 + int(layer) * 29) % 101) / 100.0
        return stable_value <= threshold

    def _activate_pending(self) -> None:
        due = [layer for layer, request in self.pending.items() if request.target_step <= self.music_step]
        for layer in due:
            request = self.pending.pop(layer)
            self._silence_channel(int(layer))
            self.active[layer] = request.phrase
            self.visual.layer(request.phrase, True, self.controls[layer].current_energy)

    def _note_on(self, channel: int, pitch: int, velocity: int, duration: int) -> None:
        key = (channel, pitch)
        if key in self._note_tokens:
            self.midi.note_off(channel, pitch)
        self._token += 1
        token = self._token
        self._note_tokens[key] = token
        self.midi.note_on(channel, pitch, velocity)
        off_step = self.music_step + max(1, duration)
        self._off_queue.setdefault(off_step, []).append((channel, pitch, token))

    def _send_due_note_offs(self) -> None:
        for channel, pitch, token in self._off_queue.pop(self.music_step, []):
            key = (channel, pitch)
            if self._note_tokens.get(key) != token:
                continue
            self.midi.note_off(channel, pitch)
            self._note_tokens.pop(key, None)

    def _smooth_controls(self) -> None:
        for layer, control in self.controls.items():
            control.current_energy += 0.32 * (control.target_energy - control.current_energy)
            control.current_muscle += 0.32 * (control.target_muscle - control.current_muscle)
            mix_reduction = max(0, len(self.active) - 3) * 3
            expression = round(42 + 79 * control.current_energy - mix_reduction)
            brightness = round(24 + 99 * control.current_muscle)
            if abs(expression - control.last_expression) >= 2:
                self.midi.control_change(int(layer), 11, expression)
                control.last_expression = expression
            if abs(brightness - control.last_brightness) >= 2:
                self.midi.control_change(int(layer), 74, brightness)
                control.last_brightness = brightness

    def _silence_channel(self, channel: int) -> None:
        keys = [key for key in self._note_tokens if key[0] == channel]
        for _, pitch in keys:
            self.midi.note_off(channel, pitch)
            self._note_tokens.pop((channel, pitch), None)

    def _silence_all(self) -> None:
        for channel, pitch in list(self._note_tokens):
            self.midi.note_off(channel, pitch)
        self._note_tokens.clear()
        self._off_queue.clear()
        self.midi.panic()
