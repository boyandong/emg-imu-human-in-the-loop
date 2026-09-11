from __future__ import annotations

import argparse
from pathlib import Path

from .music import PerformanceProfile, PhraseFactory, STEPS_PER_BEAT, new_harmony
from .protocol import Layer


PROGRAMS = {
    Layer.DRUMS: 0,
    Layer.TEXTURE: 89,
    Layer.CHORDS: 88,
    Layer.LEAD: 81,
    Layer.ARP: 10,
    Layer.BASS: 38,
}


def export_preview(
    output: Path,
    *,
    seed: int,
    bpm: float,
    motion: float,
    muscle: float,
    hold_ms: int,
    generation: int,
) -> None:
    try:
        import mido
    except ImportError as exc:
        raise RuntimeError("Install project dependencies before exporting MIDI.") from exc

    ticks_per_beat = 480
    ticks_per_step = ticks_per_beat // STEPS_PER_BEAT
    midi = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)
    conductor = mido.MidiTrack()
    conductor.append(mido.MetaMessage("track_name", name="JILV Conductor", time=0))
    conductor.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    midi.tracks.append(conductor)

    harmony = new_harmony(seed)
    profile = PerformanceProfile(motion, muscle, hold_ms, 8)
    factory = PhraseFactory()
    for layer in Layer:
        phrase = factory.generate(layer, harmony, profile, seed + int(layer) * 101, generation)
        track = mido.MidiTrack()
        track.append(mido.MetaMessage("track_name", name=layer.name.title(), time=0))
        track.append(
            mido.Message(
                "program_change", channel=int(layer), program=PROGRAMS[layer], time=0
            )
        )
        absolute_events: list[tuple[int, int, object]] = []
        for note in phrase.notes:
            start = note.step * ticks_per_step
            end = (note.step + note.duration) * ticks_per_step
            absolute_events.append(
                (
                    start,
                    1,
                    mido.Message(
                        "note_on",
                        channel=int(layer),
                        note=note.pitch,
                        velocity=note.velocity,
                        time=0,
                    ),
                )
            )
            absolute_events.append(
                (
                    end,
                    0,
                    mido.Message(
                        "note_off", channel=int(layer), note=note.pitch, velocity=0, time=0
                    ),
                )
            )
        last_tick = 0
        for absolute_tick, _, message in sorted(absolute_events, key=lambda item: (item[0], item[1])):
            message.time = absolute_tick - last_tick
            track.append(message)
            last_tick = absolute_tick
        midi.tracks.append(track)

    output.parent.mkdir(parents=True, exist_ok=True)
    midi.save(output)
    print(
        f"Saved {output} | {harmony.root_name} {harmony.mode_name} | "
        f"motion={motion:.2f} muscle={muscle:.2f} generation={generation}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export an offline JILV MIDI preview")
    parser.add_argument("--output", type=Path, default=Path("jilv_preview.mid"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bpm", type=float, default=112.0)
    parser.add_argument("--motion", type=float, default=0.58)
    parser.add_argument("--muscle", type=float, default=0.62)
    parser.add_argument("--hold-ms", type=int, default=600)
    parser.add_argument("--generation", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    export_preview(
        args.output,
        seed=args.seed,
        bpm=args.bpm,
        motion=max(0.0, min(1.0, args.motion)),
        muscle=max(0.0, min(1.0, args.muscle)),
        hold_ms=max(0, args.hold_ms),
        generation=max(0, args.generation),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
