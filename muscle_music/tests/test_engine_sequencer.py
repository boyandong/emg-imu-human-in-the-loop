from __future__ import annotations

import unittest

from jilv.engine import MusicEngine
from jilv.midi import MemoryMidiSink
from jilv.music import PerformanceProfile, PhraseFactory, new_harmony
from jilv.protocol import Gesture, GestureFrame, Layer
from jilv.sequencer import Sequencer, TransportState
from jilv.visual import MemoryVisualPublisher


class SequencerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.midi = MemoryMidiSink()
        self.visual = MemoryVisualPublisher()
        self.sequencer = Sequencer(self.midi, self.visual, count_in_beats=0)
        self.phrase = PhraseFactory().generate(
            Layer.DRUMS,
            new_harmony(1),
            PerformanceProfile(),
            seed=1,
            generation=0,
        )

    def test_phrase_activates_and_produces_midi(self) -> None:
        self.assertEqual(self.sequencer.queue_phrase(self.phrase), 0)
        self.sequencer.request_play()
        self.sequencer.step_once()
        self.assertIn(Layer.DRUMS, self.sequencer.active)
        self.assertTrue(any(event.kind == "note_on" for event in self.midi.events))

    def test_last_request_wins_until_bar_boundary(self) -> None:
        self.sequencer.request_play()
        self.sequencer.step_once()
        first = self.phrase
        second = PhraseFactory().generate(
            Layer.DRUMS, new_harmony(1), PerformanceProfile(), 1, 2
        )
        target1 = self.sequencer.queue_phrase(first)
        target2 = self.sequencer.queue_phrase(second)
        self.assertEqual(target1, target2)
        while self.sequencer.music_step <= target2:
            self.sequencer.step_once()
        self.assertEqual(self.sequencer.active[Layer.DRUMS], second)

    def test_pause_sends_panic(self) -> None:
        self.sequencer.queue_phrase(self.phrase)
        self.sequencer.request_play()
        self.sequencer.step_once()
        self.sequencer.pause()
        self.assertEqual(self.sequencer.state, TransportState.PAUSED)
        self.assertTrue(
            any(event.kind == "control_change" and event.data1 == 123 for event in self.midi.events)
        )


class EngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.midi = MemoryMidiSink()
        self.visual = MemoryVisualPublisher()
        self.sequencer = Sequencer(self.midi, self.visual, count_in_beats=0)
        self.engine = MusicEngine(self.sequencer, self.visual, seed=42, stable_ms=0)

    def frame(
        self,
        timestamp: int,
        gesture: Gesture,
        motion: float = 0.6,
        emg: float = 0.7,
    ) -> GestureFrame:
        return GestureFrame(timestamp, gesture, 0.99, motion, emg)

    def test_reentering_direction_advances_evolution(self) -> None:
        self.engine.process_frame(self.frame(0, Gesture.RIGHT, 0.3, 0.4))
        self.engine.process_frame(self.frame(100, Gesture.NONE))
        self.engine.process_frame(self.frame(200, Gesture.RIGHT, 0.9, 0.8))
        self.assertEqual(self.engine.generations[Layer.LEAD], 1)
        phrase = self.sequencer.pending[Layer.LEAD].phrase
        self.assertEqual(phrase.evolution_stage, 1)
        self.assertGreater(phrase.profile.motion, 0.3)

    def test_index_pinch_starts_and_finishes_session(self) -> None:
        self.engine.process_frame(self.frame(0, Gesture.PINCH_INDEX))
        self.assertEqual(self.sequencer.state, TransportState.PLAYING)
        self.engine.process_frame(self.frame(20, Gesture.NONE))
        self.engine.process_frame(self.frame(40, Gesture.PINCH_INDEX))
        self.assertEqual(self.sequencer.state, TransportState.STOPPED)

    def test_next_start_after_finish_creates_new_session(self) -> None:
        old_harmony = self.engine.harmony
        self.engine.process_frame(self.frame(0, Gesture.PINCH_INDEX))
        self.engine.process_frame(self.frame(20, Gesture.NONE))
        self.engine.process_frame(self.frame(40, Gesture.PINCH_INDEX))
        self.engine.process_frame(self.frame(60, Gesture.NONE))
        self.engine.process_frame(self.frame(80, Gesture.PINCH_INDEX))
        self.assertEqual(self.sequencer.state, TransportState.PLAYING)
        self.assertFalse(self.sequencer.pending)
        self.assertTrue(all(value == -1 for value in self.engine.generations.values()))
        self.assertNotEqual(self.engine.harmony, old_harmony)

    def test_middle_pinch_is_reserved_and_does_nothing(self) -> None:
        before = self.engine.snapshot()
        self.engine.process_frame(self.frame(0, Gesture.PINCH_MIDDLE))
        self.engine.process_frame(self.frame(1500, Gesture.PINCH_MIDDLE))
        after = self.engine.snapshot()
        self.assertEqual(before.transport, after.transport)
        self.assertEqual(before.harmony, after.harmony)
        self.assertEqual(before.generations, after.generations)

    def test_relaxed_emg_lowers_expression_without_stopping_layer(self) -> None:
        self.engine.process_frame(self.frame(0, Gesture.RIGHT, 0.8, 0.9))
        self.sequencer.request_play()
        self.sequencer.step_once()
        before_generation = self.engine.generations[Layer.LEAD]
        self.assertIn(Layer.LEAD, self.sequencer.active)

        self.engine.process_frame(self.frame(200, Gesture.NONE, 0.0, 0.0))

        self.assertEqual(self.sequencer.state, TransportState.PLAYING)
        self.assertIn(Layer.LEAD, self.sequencer.active)
        self.assertEqual(self.engine.generations[Layer.LEAD], before_generation)
        self.assertEqual(self.sequencer.controls[Layer.LEAD].target_energy, 0.0)
        self.assertEqual(self.sequencer.controls[Layer.LEAD].target_muscle, 0.0)


if __name__ == "__main__":
    unittest.main()
