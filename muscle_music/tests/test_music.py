from __future__ import annotations

import unittest

from jilv.music import (
    PHRASE_STEPS,
    PerformanceProfile,
    PhraseFactory,
    all_pitched_notes_in_scale,
    new_harmony,
)
from jilv.protocol import Layer


class MusicGenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.factory = PhraseFactory()
        self.profile = PerformanceProfile(0.63, 0.74, 610, 8)

    def test_generators_are_deterministic_and_valid(self) -> None:
        for session_seed in range(25):
            harmony = new_harmony(session_seed)
            for layer in Layer:
                first = self.factory.generate(layer, harmony, self.profile, 1234, 2)
                second = self.factory.generate(layer, harmony, self.profile, 1234, 2)
                self.assertEqual(first, second)
                self.assertTrue(first.notes)
                self.assertTrue(all_pitched_notes_in_scale(first, harmony))
                for note in first.notes:
                    self.assertGreaterEqual(note.step, 0)
                    self.assertLess(note.step, PHRASE_STEPS)
                    self.assertGreaterEqual(note.velocity, 1)
                    self.assertLessEqual(note.velocity, 127)

    def test_reentry_is_related_but_clearly_changed(self) -> None:
        harmony = new_harmony(99)
        for layer in Layer:
            base = self.factory.generate(layer, harmony, self.profile, 4321, 0)
            developed = self.factory.generate(layer, harmony, self.profile, 4321, 2)
            self.assertNotEqual(base.notes, developed.notes, layer.name)
            self.assertEqual(base.seed, developed.seed)
            self.assertEqual(developed.evolution_stage, 2)

    def test_performance_profile_personalizes_result(self) -> None:
        harmony = new_harmony(17)
        soft = PerformanceProfile(0.15, 0.18, 1100, 4)
        sharp = PerformanceProfile(0.92, 0.88, 180, 4)
        for layer in Layer:
            a = self.factory.generate(layer, harmony, soft, 88, 3)
            b = self.factory.generate(layer, harmony, sharp, 88, 3)
            self.assertNotEqual(a.notes, b.notes, layer.name)

    def test_complexity_is_bounded_after_peak(self) -> None:
        harmony = new_harmony(77)
        for layer in Layer:
            peak = self.factory.generate(layer, harmony, self.profile, 999, 3)
            late = self.factory.generate(layer, harmony, self.profile, 999, 12)
            self.assertEqual(late.evolution_stage, 3)
            self.assertLessEqual(len(late.notes), max(1, len(peak.notes) * 2))

    def test_late_mutation_preserves_peak_anchor_notes(self) -> None:
        harmony = new_harmony(71)
        for layer in Layer:
            peak = self.factory.generate(layer, harmony, self.profile, 302, 3)
            late = self.factory.generate(layer, harmony, self.profile, 302, 9)
            peak_anchors = {
                (note.step, note.pitch)
                for note in peak.notes
                if note.step % 4 == 0 or (layer == Layer.DRUMS and note.pitch in (36, 38))
            }
            late_anchors = {
                (note.step, note.pitch)
                for note in late.notes
                if note.step % 4 == 0 or (layer == Layer.DRUMS and note.pitch in (36, 38))
            }
            self.assertTrue(peak_anchors.issubset(late_anchors), layer.name)

    def test_lead_half_bar_anchors_are_chord_tones(self) -> None:
        harmony = new_harmony(63)
        lead = self.factory.generate(Layer.LEAD, harmony, self.profile, 112, 3)
        for note in lead.notes:
            local = note.step % 16
            if local % 8 != 0:
                continue
            bar = note.step // 16
            chord_pcs = {pitch % 12 for pitch in harmony.chord(bar, 4)}
            self.assertIn(note.pitch % 12, chord_pcs)


if __name__ == "__main__":
    unittest.main()
