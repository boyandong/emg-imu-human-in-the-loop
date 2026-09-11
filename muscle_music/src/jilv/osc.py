from __future__ import annotations

import socket
import struct
import threading
from collections.abc import Callable
from typing import Any

from .protocol import Gesture, GestureFrame


class OscError(ValueError):
    pass


def _pad4(length: int) -> int:
    return (4 - (length % 4)) % 4


def _encode_string(value: str) -> bytes:
    data = value.encode("utf-8") + b"\x00"
    return data + b"\x00" * _pad4(len(data))


def _decode_string(data: bytes, offset: int) -> tuple[str, int]:
    try:
        end = data.index(0, offset)
    except ValueError as exc:
        raise OscError("OSC string is not null-terminated") from exc
    value = data[offset:end].decode("utf-8")
    consumed = end - offset + 1
    return value, end + 1 + _pad4(consumed)


def encode_message(address: str, *args: Any) -> bytes:
    if not address.startswith("/"):
        raise OscError("OSC address must start with '/'")

    tags: list[str] = []
    payload = bytearray()
    for arg in args:
        if isinstance(arg, bool):
            tags.append("i")
            payload.extend(struct.pack(">i", int(arg)))
        elif isinstance(arg, int):
            if -(2**31) <= arg < 2**31:
                tags.append("i")
                payload.extend(struct.pack(">i", arg))
            else:
                tags.append("h")
                payload.extend(struct.pack(">q", arg))
        elif isinstance(arg, float):
            tags.append("f")
            payload.extend(struct.pack(">f", arg))
        elif isinstance(arg, str):
            tags.append("s")
            payload.extend(_encode_string(arg))
        else:
            raise OscError(f"unsupported OSC argument type: {type(arg)!r}")

    return _encode_string(address) + _encode_string("," + "".join(tags)) + bytes(payload)


def decode_message(data: bytes) -> tuple[str, list[Any]]:
    address, offset = _decode_string(data, 0)
    tags, offset = _decode_string(data, offset)
    if not tags.startswith(","):
        raise OscError("OSC type tag string must start with ','")

    values: list[Any] = []
    for tag in tags[1:]:
        if tag == "i":
            if offset + 4 > len(data):
                raise OscError("truncated OSC int32")
            values.append(struct.unpack_from(">i", data, offset)[0])
            offset += 4
        elif tag == "h":
            if offset + 8 > len(data):
                raise OscError("truncated OSC int64")
            values.append(struct.unpack_from(">q", data, offset)[0])
            offset += 8
        elif tag == "f":
            if offset + 4 > len(data):
                raise OscError("truncated OSC float32")
            values.append(struct.unpack_from(">f", data, offset)[0])
            offset += 4
        elif tag == "s":
            value, offset = _decode_string(data, offset)
            values.append(value)
        else:
            raise OscError(f"unsupported OSC type tag: {tag!r}")
    return address, values


class OscClient:
    def __init__(self, host: str, port: int) -> None:
        self.target = (host, int(port))
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._lock = threading.Lock()

    def send(self, address: str, *args: Any) -> None:
        packet = encode_message(address, *args)
        with self._lock:
            self.socket.sendto(packet, self.target)

    def close(self) -> None:
        self.socket.close()


class OscStateServer:
    def __init__(
        self,
        host: str,
        port: int,
        callback: Callable[[GestureFrame], None],
        address: str = "/emgimu/state",
    ) -> None:
        self.host = host
        self.port = int(port)
        self.callback = callback
        self.address = address
        self.socket: socket.socket | None = None
        self.thread: threading.Thread | None = None
        self.running = threading.Event()
        self.last_error: Exception | None = None

    def start(self) -> None:
        if self.running.is_set():
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        self.port = int(sock.getsockname()[1])
        sock.settimeout(0.25)
        self.socket = sock
        self.running.set()
        self.thread = threading.Thread(target=self._run, name="jilv-osc-input", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        assert self.socket is not None
        while self.running.is_set():
            try:
                packet, _ = self.socket.recvfrom(4096)
            except TimeoutError:
                continue
            except OSError:
                if self.running.is_set():
                    raise
                return

            try:
                address, args = decode_message(packet)
                if address != self.address:
                    continue
                if len(args) != 5:
                    raise OscError(f"{self.address} expects 5 arguments, got {len(args)}")
                frame = GestureFrame(
                    timestamp_ms=int(args[0]),
                    gesture=Gesture(int(args[1])),
                    confidence=float(args[2]),
                    motion_energy=float(args[3]),
                    emg_activation=float(args[4]),
                )
                self.callback(frame)
            except (OscError, ValueError, TypeError) as exc:
                self.last_error = exc

    def stop(self) -> None:
        self.running.clear()
        if self.socket is not None:
            self.socket.close()
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        self.socket = None
        self.thread = None
