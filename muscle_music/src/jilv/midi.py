from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Protocol


class MidiSink(Protocol):
    def note_on(self, channel: int, note: int, velocity: int) -> None: ...

    def note_off(self, channel: int, note: int) -> None: ...

    def control_change(self, channel: int, control: int, value: int) -> None: ...

    def panic(self) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class MidiEvent:
    kind: str
    channel: int
    data1: int
    data2: int
    timestamp: float


class MemoryMidiSink:
    """Dependency-free sink used by dry-run mode and automated tests."""

    def __init__(self) -> None:
        self.events: list[MidiEvent] = []
        self.closed = False

    def _append(self, kind: str, channel: int, data1: int, data2: int = 0) -> None:
        self.events.append(MidiEvent(kind, channel, data1, data2, time.perf_counter()))

    def note_on(self, channel: int, note: int, velocity: int) -> None:
        self._append("note_on", channel, note, velocity)

    def note_off(self, channel: int, note: int) -> None:
        self._append("note_off", channel, note)

    def control_change(self, channel: int, control: int, value: int) -> None:
        self._append("control_change", channel, control, value)

    def panic(self) -> None:
        for channel in range(16):
            self.control_change(channel, 123, 0)
            self.control_change(channel, 120, 0)

    def close(self) -> None:
        self.closed = True


class MidoMidiSink:
    def __init__(self, port_name: str | None = None) -> None:
        try:
            import mido
        except ImportError as exc:
            raise RuntimeError(
                "MIDI dependencies are missing. Run: pip install -r requirements.txt"
            ) from exc

        self._mido = mido
        names = mido.get_output_names()
        if port_name is None:
            if not names:
                raise RuntimeError(
                    "No MIDI output was found. Create a loopMIDI port, then pass --midi-port PORT_NAME."
                )
            port_name = names[0]
        if port_name not in names:
            available = ", ".join(names) if names else "(none)"
            raise RuntimeError(f"MIDI port '{port_name}' not found. Available ports: {available}")
        self.port_name = port_name
        self._port = mido.open_output(port_name)

    def note_on(self, channel: int, note: int, velocity: int) -> None:
        self._port.send(
            self._mido.Message("note_on", channel=channel, note=note, velocity=velocity)
        )

    def note_off(self, channel: int, note: int) -> None:
        self._port.send(self._mido.Message("note_off", channel=channel, note=note, velocity=0))

    def control_change(self, channel: int, control: int, value: int) -> None:
        self._port.send(
            self._mido.Message(
                "control_change", channel=channel, control=control, value=value
            )
        )

    def panic(self) -> None:
        for channel in range(16):
            self.control_change(channel, 123, 0)
            self.control_change(channel, 120, 0)

    def close(self) -> None:
        self.panic()
        self._port.close()


def list_output_ports() -> list[str]:
    try:
        import mido
    except ImportError as exc:
        raise RuntimeError(
            "MIDI dependencies are missing. Run: pip install -r requirements.txt"
        ) from exc
    return list(mido.get_output_names())


def list_ports_main() -> None:
    ports = list_output_ports()
    if not ports:
        print("No MIDI output ports found.")
        return
    for port in ports:
        print(port)
