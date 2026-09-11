from __future__ import annotations

from dataclasses import dataclass, replace
import random
from typing import Iterable, Sequence

from .protocol import Layer, clamp01


STEPS_PER_BEAT = 4
BEATS_PER_BAR = 4
STEPS_PER_BAR = STEPS_PER_BEAT * BEATS_PER_BAR
PHRASE_BARS = 4
PHRASE_STEPS = STEPS_PER_BAR * PHRASE_BARS


@dataclass(frozen=True, slots=True)
class Note:
    step: int
    duration: int
    pitch: int
    velocity: int

    def __post_init__(self) -> None:
        if not 0 <= self.step < PHRASE_STEPS:
            raise ValueError(f"step out of range: {self.step}")
        if self.duration < 1:
            raise ValueError("duration must be positive")
        if not 0 <= self.pitch <= 127:
            raise ValueError(f"pitch out of range: {self.pitch}")
        if not 1 <= self.velocity <= 127:
            raise ValueError(f"velocity out of range: {self.velocity}")


@dataclass(frozen=True, slots=True)
class PerformanceProfile:
    """Session-local description of how one layer is physically performed."""

    motion: float = 0.5
    muscle: float = 0.5
    hold_ms: int = 450
    samples: int = 0

    @property
    def energy(self) -> float:
        return clamp01(0.55 * self.motion + 0.45 * self.muscle)

    @property
    def articulation(self) -> float:
        return clamp01(self.muscle)

    @property
    def density(self) -> float:
        return clamp01(0.7 * self.motion + 0.3 * self.muscle)

    @property
    def legato(self) -> float:
        return clamp01(self.hold_ms / 1400.0)

    @property
    def signature(self) -> int:
        motion = round(clamp01(self.motion) * 15)
        muscle = round(clamp01(self.muscle) * 15)
        duration = min(15, max(0, round(self.hold_ms / 100)))
        return (motion << 8) | (muscle << 4) | duration

    def blended(
        self,
        *,
        motion: float,
        muscle: float,
        hold_ms: int | None = None,
        weight: float = 0.28,
    ) -> PerformanceProfile:
        weight = clamp01(weight)
        if self.samples == 0:
            weight = 1.0
        next_hold = self.hold_ms if hold_ms is None else max(0, int(hold_ms))
        return PerformanceProfile(
            motion=(1.0 - weight) * self.motion + weight * clamp01(motion),
            muscle=(1.0 - weight) * self.muscle + weight * clamp01(muscle),
            hold_ms=round((1.0 - weight) * self.hold_ms + weight * next_hold),
            samples=self.samples + 1,
        )

    def mixed(self, other: PerformanceProfile, other_weight: float = 0.3) -> PerformanceProfile:
        other_weight = clamp01(other_weight)
        return PerformanceProfile(
            motion=(1.0 - other_weight) * self.motion + other_weight * other.motion,
            muscle=(1.0 - other_weight) * self.muscle + other_weight * other.muscle,
            hold_ms=round(
                (1.0 - other_weight) * self.hold_ms + other_weight * other.hold_ms
            ),
            samples=self.samples + other.samples,
        )


@dataclass(frozen=True, slots=True)
class HarmonyPlan:
    root_pc: int
    root_name: str
    mode_name: str
    scale_intervals: tuple[int, ...]
    progression_degrees: tuple[int, int, int, int]

    def scale_pitch_classes(self) -> set[int]:
        return {(self.root_pc + interval) % 12 for interval in self.scale_intervals}

    def scale_notes(self, low: int, high: int) -> list[int]:
        allowed = self.scale_pitch_classes()
        return [pitch for pitch in range(low, high + 1) if pitch % 12 in allowed]

    def degree_pitch(self, degree: int, octave: int) -> int:
        interval = self.scale_intervals[degree % len(self.scale_intervals)]
        octave_add = degree // len(self.scale_intervals)
        return 12 * (octave + 1 + octave_add) + self.root_pc + interval

    def chord(self, bar: int, octave: int = 3) -> tuple[int, int, int]:
        degree = self.progression_degrees[bar % len(self.progression_degrees)]
        if len(self.scale_intervals) < 7:
            return tuple(self.degree_pitch(degree + offset, octave) for offset in (0, 2, 4))
        return tuple(self.degree_pitch(degree + offset, octave) for offset in (0, 2, 4))


@dataclass(frozen=True, slots=True)
class Phrase:
    layer: Layer
    notes: tuple[Note, ...]
    seed: int
    generation: int
    evolution_stage: int
    profile: PerformanceProfile
    change_label: str


ROOTS: tuple[tuple[str, int], ...] = (
    ("C", 0),
    ("D", 2),
    ("E", 4),
    ("F", 5),
    ("G", 7),
    ("A", 9),
)

MODES: tuple[tuple[str, tuple[int, ...], tuple[int, int, int, int]], ...] = (
    ("natural minor", (0, 2, 3, 5, 7, 8, 10), (0, 5, 2, 6)),
    ("dorian", (0, 2, 3, 5, 7, 9, 10), (0, 3, 5, 4)),
    ("minor pentatonic", (0, 3, 5, 7, 10), (0, 3, 2, 4)),
)


def new_harmony(seed: int) -> HarmonyPlan:
    rng = random.Random(seed)
    root_name, root_pc = rng.choice(ROOTS)
    mode_name, intervals, progression = rng.choice(MODES)
    return HarmonyPlan(root_pc, root_name, mode_name, intervals, progression)


def _velocity(profile: PerformanceProfile, base: int, spread: int, rng: random.Random) -> int:
    value = base + round((profile.energy - 0.5) * 28) + rng.randint(-spread, spread)
    return max(25, min(122, value))


def _fit_pitch(pitch: int, allowed: Sequence[int]) -> int:
    return min(allowed, key=lambda candidate: (abs(candidate - pitch), candidate))


def _nearest_inversion(chord: tuple[int, ...], previous_top: int | None) -> tuple[int, ...]:
    candidates: list[tuple[int, ...]] = []
    for inversion in range(len(chord)):
        voiced = list(chord[inversion:]) + [note + 12 for note in chord[:inversion]]
        candidates.append(tuple(voiced))
    if previous_top is None:
        return candidates[0]
    return min(candidates, key=lambda notes: abs(notes[-1] - previous_top))


class PhraseFactory:
    """Creates related, clearly audible phrase generations for all six layers."""

    STAGE_LABELS = ("foundation", "rhythmic variation", "ornamented development", "peak variation")

    def generate(
        self,
        layer: Layer,
        harmony: HarmonyPlan,
        profile: PerformanceProfile,
        seed: int,
        generation: int = 0,
    ) -> Phrase:
        stage = min(3, max(0, int(generation)))
        epoch = max(0, int(generation) - 3)
        base_seed = int(seed) + int(layer) * 1009
        # The motif seed does not depend on live gesture values. This keeps the
        # original theme recognizable while the profile changes its expression.
        rng = random.Random(base_seed)
        builder = {
            Layer.DRUMS: self._drums,
            Layer.TEXTURE: self._texture,
            Layer.CHORDS: self._chords,
            Layer.LEAD: self._lead,
            Layer.ARP: self._arp,
            Layer.BASS: self._bass,
        }[layer]
        notes = builder(harmony, profile, stage, rng)
        if epoch:
            notes = self._late_mutation(notes, layer, harmony, base_seed, epoch)
        return Phrase(
            layer=layer,
            notes=tuple(sorted(notes, key=lambda note: (note.step, note.pitch))),
            seed=base_seed,
            generation=generation,
            evolution_stage=stage,
            profile=profile,
            change_label=self.STAGE_LABELS[stage],
        )

    def _late_mutation(
        self,
        notes: list[Note],
        layer: Layer,
        harmony: HarmonyPlan,
        base_seed: int,
        epoch: int,
    ) -> list[Note]:
        """Changes ornaments after peak complexity without replacing the motif."""
        rng = random.Random(base_seed + epoch * 7919)
        allowed = harmony.scale_notes(24, 108)
        result: list[Note] = []
        for index, note in enumerate(notes):
            anchor = note.step % STEPS_PER_BEAT == 0
            if layer == Layer.DRUMS and note.pitch in (36, 38):
                anchor = True
            if anchor or (index + epoch) % 3:
                result.append(note)
                continue
            local = note.step % STEPS_PER_BAR
            shift = rng.choice((-1, 1))
            shifted_local = max(1, min(STEPS_PER_BAR - 1, local + shift))
            step = note.step - local + shifted_local
            pitch = note.pitch
            if layer != Layer.DRUMS:
                candidates = [value for value in allowed if abs(value - pitch) <= 5 and value != pitch]
                if candidates:
                    pitch = rng.choice(candidates)
            velocity = max(1, min(127, note.velocity + rng.choice((-7, 5, 8))))
            result.append(replace(note, step=step, pitch=pitch, velocity=velocity))
        return result

    def _drums(
        self, harmony: HarmonyPlan, profile: PerformanceProfile, stage: int, rng: random.Random
    ) -> list[Note]:
        del harmony
        notes: list[Note] = []
        for bar in range(PHRASE_BARS):
            start = bar * STEPS_PER_BAR
            kick_steps = [0, 8]
            if stage >= 1:
                kick_steps.append(10 if bar % 2 else 6)
            if stage >= 3 and profile.density > 0.42:
                kick_steps.append(14)
            for local in kick_steps:
                notes.append(Note(start + local, 1, 36, _velocity(profile, 94, 5, rng)))
            for local in (4, 12):
                notes.append(Note(start + local, 1, 38, _velocity(profile, 88, 4, rng)))
            hat_stride = 4 if stage == 0 and profile.density < 0.58 else 2
            if stage >= 2 and profile.density > 0.66:
                hat_stride = 1
            for local in range(0, STEPS_PER_BAR, hat_stride):
                pitch = 46 if stage >= 2 and local in (6, 14) else 42
                accent = 9 if local % 4 == 0 else -7
                notes.append(Note(start + local, 1, pitch, _velocity(profile, 66 + accent, 5, rng)))
            if stage >= 2 and bar % 2 == 1:
                notes.append(Note(start + 15, 1, 39, _velocity(profile, 82, 5, rng)))
        return notes

    def _bass(
        self, harmony: HarmonyPlan, profile: PerformanceProfile, stage: int, rng: random.Random
    ) -> list[Note]:
        notes: list[Note] = []
        allowed = harmony.scale_notes(32, 55)
        for bar in range(PHRASE_BARS):
            root, _, fifth = harmony.chord(bar, 2)
            root = _fit_pitch(root, allowed)
            fifth = _fit_pitch(fifth, allowed)
            pattern = [(0, root), (8, fifth)]
            if stage >= 1:
                pattern += [(6, root + 12), (12, root)]
            if stage >= 2:
                passing = _fit_pitch(root + (2 if bar % 2 else 3), allowed)
                pattern += [(3, passing), (14, fifth)]
            if stage >= 3 and profile.density > 0.45:
                pattern += [(10, _fit_pitch(root + 7, allowed))]
            pattern.sort()
            for index, (local, pitch) in enumerate(pattern):
                next_local = pattern[index + 1][0] if index + 1 < len(pattern) else STEPS_PER_BAR
                duration = max(1, round((next_local - local) * (0.45 + 0.45 * profile.legato)))
                notes.append(
                    Note(bar * STEPS_PER_BAR + local, duration, pitch, _velocity(profile, 82, 5, rng))
                )
        return notes

    def _chords(
        self, harmony: HarmonyPlan, profile: PerformanceProfile, stage: int, rng: random.Random
    ) -> list[Note]:
        notes: list[Note] = []
        previous_top: int | None = None
        for bar in range(PHRASE_BARS):
            chord = _nearest_inversion(harmony.chord(bar, 3), previous_top)
            previous_top = chord[-1]
            attacks = [0]
            if stage >= 1:
                attacks.append(8)
            if stage >= 3 and profile.density > 0.55:
                attacks += [6, 14]
            attacks = sorted(set(attacks))
            for index, local in enumerate(attacks):
                next_local = attacks[index + 1] if index + 1 < len(attacks) else STEPS_PER_BAR
                duration = max(2, next_local - local - (1 if stage >= 2 else 0))
                for voice, pitch in enumerate(chord):
                    adjusted = pitch + (12 if stage >= 2 and voice == 2 and bar % 2 else 0)
                    notes.append(
                        Note(
                            bar * STEPS_PER_BAR + local,
                            duration,
                            adjusted,
                            _velocity(profile, 62 + voice * 3, 3, rng),
                        )
                    )
        return notes

    def _lead(
        self, harmony: HarmonyPlan, profile: PerformanceProfile, stage: int, rng: random.Random
    ) -> list[Note]:
        notes: list[Note] = []
        allowed = harmony.scale_notes(60, 84)
        base_rng = random.Random(rng.randint(0, 2**31 - 1) ^ 0x4C454144)
        motif_steps = [0, 4, 8, 12]
        motif: list[int] = []
        cursor = allowed[len(allowed) // 2]
        for _ in motif_steps:
            cursor = _fit_pitch(cursor + base_rng.choice((-3, -2, 2, 3, 5)), allowed)
            motif.append(cursor)
        for bar in range(PHRASE_BARS):
            bar_notes = list(zip(motif_steps, motif))
            if stage >= 1:
                bar_notes[1] = (3 if bar % 2 == 0 else 5, bar_notes[1][1])
                bar_notes.append((10, _fit_pitch(motif[2] + rng.choice((-2, 2, 3)), allowed)))
            if stage >= 2:
                bar_notes += [
                    (2, _fit_pitch(motif[0] + 2, allowed)),
                    (14, _fit_pitch(motif[-1] + (5 if bar % 2 else -3), allowed)),
                ]
            if stage >= 3 and profile.density > 0.38:
                bar_notes += [(7, _fit_pitch(motif[2] + 7, allowed)), (15, _fit_pitch(motif[0], allowed))]
            transpose = 0 if bar < 2 else (12 if profile.motion > 0.72 else 0)
            for local, pitch in sorted(set(bar_notes)):
                if local % 8 == 0:
                    chord_pcs = {value % 12 for value in harmony.chord(bar, 4)}
                    stable = [value for value in allowed if value % 12 in chord_pcs]
                    if stable:
                        pitch = _fit_pitch(pitch, stable)
                duration = 1 + round(profile.legato * 3)
                notes.append(
                    Note(
                        bar * STEPS_PER_BAR + local,
                        duration,
                        min(96, pitch + transpose),
                        _velocity(profile, 78, 8, rng),
                    )
                )
        return notes

    def _arp(
        self, harmony: HarmonyPlan, profile: PerformanceProfile, stage: int, rng: random.Random
    ) -> list[Note]:
        notes: list[Note] = []
        stride = 4 if stage == 0 else 2
        if stage >= 2 and profile.density > 0.62:
            stride = 1
        for bar in range(PHRASE_BARS):
            chord = list(harmony.chord(bar, 4))
            chord.append(chord[0] + 12)
            order = [0, 1, 2, 1]
            if stage >= 1:
                order = [0, 2, 1, 3, 2, 1, 0, 2]
            if stage >= 3 and bar % 2:
                order = list(reversed(order))
            for index, local in enumerate(range(0, STEPS_PER_BAR, stride)):
                pitch = chord[order[index % len(order)] % len(chord)]
                notes.append(
                    Note(
                        bar * STEPS_PER_BAR + local,
                        max(1, stride - 1 + round(profile.legato)),
                        pitch,
                        _velocity(profile, 67, 6, rng),
                    )
                )
        return notes

    def _texture(
        self, harmony: HarmonyPlan, profile: PerformanceProfile, stage: int, rng: random.Random
    ) -> list[Note]:
        notes: list[Note] = []
        allowed = harmony.scale_notes(48, 79)
        for bar in range(PHRASE_BARS):
            chord = harmony.chord(bar, 3)
            base_pitch = _fit_pitch(chord[stage % len(chord)], allowed)
            notes.append(
                Note(
                    bar * STEPS_PER_BAR,
                    STEPS_PER_BAR - 1,
                    base_pitch,
                    _velocity(profile, 47, 4, rng),
                )
            )
            if stage >= 1:
                shimmer = _fit_pitch(base_pitch + 12 + rng.choice((-2, 0, 2)), allowed)
                notes.append(
                    Note(
                        bar * STEPS_PER_BAR + 8,
                        7,
                        shimmer,
                        _velocity(profile, 38, 4, rng),
                    )
                )
            if stage >= 3 and profile.density > 0.45:
                grain = _fit_pitch(base_pitch + rng.choice((-5, 5, 7)), allowed)
                notes.append(Note(bar * STEPS_PER_BAR + 12, 3, grain, _velocity(profile, 50, 6, rng)))
        return notes


def all_pitched_notes_in_scale(phrase: Phrase, harmony: HarmonyPlan) -> bool:
    if phrase.layer == Layer.DRUMS:
        return True
    allowed = harmony.scale_pitch_classes()
    return all(note.pitch % 12 in allowed for note in phrase.notes)


def phrase_density(phrase: Phrase) -> float:
    return len(phrase.notes) / PHRASE_STEPS


def transpose_phrase(phrase: Phrase, semitones: int) -> Phrase:
    notes: Iterable[Note] = (
        replace(note, pitch=max(0, min(127, note.pitch + semitones))) for note in phrase.notes
    )
    return replace(phrase, notes=tuple(notes))
