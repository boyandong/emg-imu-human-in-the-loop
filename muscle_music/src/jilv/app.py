from __future__ import annotations

import argparse
import signal
import sys
import threading
import time

from .engine import MusicEngine
from .midi import MemoryMidiSink, MidoMidiSink
from .osc import OscStateServer
from .sequencer import Sequencer
from .visual import NullVisualPublisher, OscVisualPublisher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jilv-engine",
        description="JILV EMG-IMU continuous gesture to generative MIDI engine",
    )
    parser.add_argument("--input-host", default="127.0.0.1")
    parser.add_argument("--input-port", type=int, default=9000)
    parser.add_argument("--visual-host", default="127.0.0.1")
    parser.add_argument("--visual-port", type=int, default=9001)
    parser.add_argument("--midi-port", default=None)
    parser.add_argument("--bpm", type=float, default=112.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Do not open a MIDI device")
    parser.add_argument("--no-visual", action="store_true", help="Disable TouchDesigner OSC")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    visual = (
        NullVisualPublisher()
        if args.no_visual
        else OscVisualPublisher(args.visual_host, args.visual_port)
    )
    midi = MemoryMidiSink() if args.dry_run else MidoMidiSink(args.midi_port)
    sequencer = Sequencer(midi, visual, bpm=args.bpm)
    engine = MusicEngine(sequencer, visual, seed=args.seed)
    server = OscStateServer(args.input_host, args.input_port, engine.process_frame)
    stopping = threading.Event()

    def stop_handler(signum: int, frame: object) -> None:
        del signum, frame
        stopping.set()

    signal.signal(signal.SIGINT, stop_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, stop_handler)

    try:
        sequencer.start_thread()
        server.start()
        mode = "dry-run" if args.dry_run else f"MIDI: {midi.port_name}"
        print(f"JILV engine listening on OSC {args.input_host}:{args.input_port} ({mode})")
        print("Press Ctrl+C to stop.")
        while not stopping.wait(0.20):
            engine.check_input_timeout()
            if server.last_error is not None:
                print(f"OSC input warning: {server.last_error}", file=sys.stderr)
                visual.error(str(server.last_error))
                server.last_error = None
    finally:
        server.stop()
        sequencer.stop_thread()
        midi.close()
        visual.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
