"""Serve the browser demo and relay the local OSC control stream to the page."""

from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from jilv.osc import OscError, decode_message  # noqa: E402


OSC_ADDRESS = "/emgimu/state"
STALE_AFTER_SECONDS = 1.5


def _unit(value: object) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("控制参数必须是有限数值")
    return max(0.0, min(1.0, number))


def parse_music_state(packet: bytes) -> dict[str, int | float]:
    """Decode the five-field OSC packet kept by the acquisition backend."""
    address, args = decode_message(packet)
    if address != OSC_ADDRESS:
        raise OscError(f"忽略未支持的 OSC 地址：{address}")
    if len(args) != 5:
        raise OscError(f"{OSC_ADDRESS} 需要 5 个参数，实际收到 {len(args)} 个")

    gesture_id = int(args[1])
    if not 0 <= gesture_id <= 8:
        raise ValueError("gesture_id 必须位于 0 到 8")
    return {
        "timestamp_ms": int(args[0]),
        "gesture_id": gesture_id,
        "confidence": _unit(args[2]),
        "motion_energy": _unit(args[3]),
        # 第五个位置为兼容原 EMG-IMU 协议保留，纯 IMU 前端不使用。
        "reserved_expression": _unit(args[4]),
    }


class LiveInputState:
    def __init__(self, osc_port: int) -> None:
        self._condition = threading.Condition()
        self._revision = 0
        self._last_received: float | None = None
        self._error: str | None = None
        self._osc_port = int(osc_port)
        self._frame: dict[str, int | float] = {
            "timestamp_ms": 0,
            "gesture_id": 0,
            "confidence": 0.0,
            "motion_energy": 0.0,
            "reserved_expression": 0.0,
        }

    def update(self, frame: dict[str, int | float]) -> None:
        with self._condition:
            self._frame = dict(frame)
            self._last_received = time.monotonic()
            self._error = None
            self._revision += 1
            self._condition.notify_all()

    def set_error(self, message: str) -> None:
        with self._condition:
            self._error = str(message)
            self._revision += 1
            self._condition.notify_all()

    def wait_snapshot(self, previous_revision: int, timeout: float = 0.5) -> dict[str, object]:
        with self._condition:
            if self._revision == previous_revision:
                self._condition.wait(timeout)
            connected = (
                self._last_received is not None
                and time.monotonic() - self._last_received <= STALE_AFTER_SECONDS
            )
            return {
                **self._frame,
                "connected": connected,
                "osc_port": self._osc_port,
                "sample_rate_hz": 200,
                "error": self._error,
                "revision": self._revision,
            }


class OscInputRelay:
    def __init__(self, host: str, port: int, state: LiveInputState) -> None:
        self.host = host
        self.port = int(port)
        self.state = state
        self.socket: socket.socket | None = None
        self.thread: threading.Thread | None = None
        self.running = threading.Event()

    def start(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.bind((self.host, self.port))
        except OSError:
            sock.close()
            self.state.set_error(f"OSC {self.port} 端口被占用")
            return
        sock.settimeout(0.25)
        self.socket = sock
        self.running.set()
        self.thread = threading.Thread(target=self._run, name="jilv-browser-osc", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        assert self.socket is not None
        while self.running.is_set():
            try:
                packet, _peer = self.socket.recvfrom(4096)
            except TimeoutError:
                continue
            except OSError:
                return
            try:
                self.state.update(parse_music_state(packet))
            except (OscError, ValueError, TypeError) as exc:
                self.state.set_error(str(exc))

    def stop(self) -> None:
        self.running.clear()
        if self.socket is not None:
            self.socket.close()
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        self.socket = None
        self.thread = None


class DemoRequestHandler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def __init__(self, *args: object, directory: str, live_state: LiveInputState, **kwargs: object) -> None:
        self.live_state = live_state
        super().__init__(*args, directory=directory, **kwargs)

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/api/status":
            self._send_status()
            return
        if path == "/api/stream":
            self._send_stream()
            return
        super().do_GET()

    def end_headers(self) -> None:
        if urlsplit(self.path).path.startswith("/api/"):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _send_status(self) -> None:
        payload = json.dumps(
            self.live_state.wait_snapshot(-1, 0),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        revision = -1
        try:
            while True:
                snapshot = self.live_state.wait_snapshot(revision)
                revision = int(snapshot["revision"])
                payload = json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"))
                self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            return

    def log_message(self, format: str, *args: object) -> None:
        if not urlsplit(self.path).path.startswith("/api/stream"):
            super().log_message(format, *args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the JILV browser demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument("--osc-host", default="127.0.0.1")
    parser.add_argument("--osc-port", default=9000, type=int)
    args = parser.parse_args()

    live_state = LiveInputState(args.osc_port)
    relay = OscInputRelay(args.osc_host, args.osc_port, live_state)
    relay.start()

    demo_dir = PROJECT_DIR / "demo"
    handler = partial(DemoRequestHandler, directory=str(demo_dir), live_state=live_state)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    server.daemon_threads = True
    print(f"JILV demo: http://{args.host}:{args.port}")
    print(f"IMU OSC input: {args.osc_host}:{args.osc_port} {OSC_ADDRESS}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        relay.stop()


if __name__ == "__main__":
    main()
