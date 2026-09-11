from __future__ import annotations

import argparse
import os
import threading
import time

from .engine import now_ms
from .osc import OscClient
from .protocol import Gesture, gesture_names


KEY_TO_GESTURE = {
    "w": Gesture.FORWARD,
    "s": Gesture.BACKWARD,
    "a": Gesture.LEFT,
    "d": Gesture.RIGHT,
    "r": Gesture.UP,
    "f": Gesture.DOWN,
    " ": Gesture.NONE,
}


class GestureSimulator:
    def __init__(self, host: str, port: int, hz: float = 30.0) -> None:
        self.client = OscClient(host, port)
        self.period = 1.0 / hz
        self.gesture = Gesture.NONE
        self.motion = 0.55
        self.emg = 0.55
        self.confidence = 0.98
        self._lock = threading.RLock()
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._send_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stopping.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        self.client.close()

    def set_gesture(self, gesture: Gesture) -> None:
        with self._lock:
            self.gesture = gesture

    def pulse(self, gesture: Gesture, seconds: float = 0.25) -> None:
        self.set_gesture(gesture)
        time.sleep(seconds)
        self.set_gesture(Gesture.NONE)

    def adjust_motion(self, delta: float) -> None:
        with self._lock:
            self.motion = max(0.0, min(1.0, self.motion + delta))

    def adjust_emg(self, delta: float) -> None:
        with self._lock:
            self.emg = max(0.0, min(1.0, self.emg + delta))

    def _send_loop(self) -> None:
        while not self._stopping.is_set():
            with self._lock:
                values = (self.gesture, self.confidence, self.motion, self.emg)
            self.client.send(
                "/emgimu/state",
                now_ms(),
                int(values[0]),
                values[1],
                values[2],
                values[3],
            )
            self._stopping.wait(self.period)


def _print_help() -> None:
    print("\nControls")
    print("  W/S/A/D/R/F : forward/back/left/right/up/down")
    print("  Space       : no action")
    print("  J           : index pinch (start/finish session)")
    print("  +/-         : motion energy")
    print("  [/ ]        : EMG activation")
    print("  G           : scripted public-demo sequence")
    print("  H           : show this help")
    print("  Q           : quit\n")


def _scripted_demo(sim: GestureSimulator) -> None:
    sequence = [
        (Gesture.PINCH_INDEX, 0.25, 0.35),
        (Gesture.FORWARD, 0.35, 1.0),
        (Gesture.DOWN, 0.35, 1.0),
        (Gesture.LEFT, 0.35, 1.0),
        (Gesture.RIGHT, 0.35, 1.0),
        (Gesture.UP, 0.35, 1.0),
        (Gesture.BACKWARD, 0.35, 1.0),
        (Gesture.RIGHT, 0.35, 1.0),
        (Gesture.PINCH_INDEX, 0.25, 0.20),
    ]
    for gesture, hold, rest in sequence:
        sim.pulse(gesture, hold)
        time.sleep(rest)


def _read_key_windows() -> str:
    import msvcrt

    while True:
        key = msvcrt.getwch()
        if key in ("\x00", "\xe0"):
            msvcrt.getwch()
            continue
        return key.lower()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Keyboard simulator for the JILV OSC protocol")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--demo", action="store_true", help="Run one scripted gesture sequence")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sim = GestureSimulator(args.host, args.port)
    sim.start()
    try:
        if args.demo:
            _scripted_demo(sim)
            return 0
        if os.name != "nt":
            raise RuntimeError("Interactive simulator currently requires Windows; use --demo elsewhere.")
        _print_help()
        while True:
            key = _read_key_windows()
            if key == "q":
                break
            if key in KEY_TO_GESTURE:
                sim.pulse(KEY_TO_GESTURE[key])
            elif key == "j":
                sim.pulse(Gesture.PINCH_INDEX)
            elif key == "+" or key == "=":
                sim.adjust_motion(0.1)
            elif key == "-":
                sim.adjust_motion(-0.1)
            elif key == "]":
                sim.adjust_emg(0.1)
            elif key == "[":
                sim.adjust_emg(-0.1)
            elif key == "g":
                _scripted_demo(sim)
            elif key == "h":
                _print_help()
            print(
                f"state={sim.gesture.name:<12} motion={sim.motion:.2f} emg={sim.emg:.2f}",
                end="\r",
            )
    finally:
        sim.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
