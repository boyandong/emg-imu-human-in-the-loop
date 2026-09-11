from __future__ import annotations

import unittest
import threading

from jilv.osc import OscClient, OscStateServer, decode_message, encode_message
from jilv.protocol import EventKind, Gesture, GestureFrame, GestureStateMachine


class OscTests(unittest.TestCase):
    def test_round_trip_state_message(self) -> None:
        packet = encode_message("/emgimu/state", 2**40, 4, 0.98, 0.33, 0.71)
        address, values = decode_message(packet)
        self.assertEqual(address, "/emgimu/state")
        self.assertEqual(values[:2], [2**40, 4])
        self.assertAlmostEqual(values[2], 0.98, places=5)
        self.assertAlmostEqual(values[3], 0.33, places=5)
        self.assertAlmostEqual(values[4], 0.71, places=5)

    def test_udp_state_server_receives_real_packet(self) -> None:
        received: list[GestureFrame] = []
        ready = threading.Event()

        def callback(frame: GestureFrame) -> None:
            received.append(frame)
            ready.set()

        server = OscStateServer("127.0.0.1", 0, callback)
        server.start()
        client = OscClient("127.0.0.1", server.port)
        try:
            client.send("/emgimu/state", 123456, int(Gesture.LEFT), 0.97, 0.44, 0.66)
            self.assertTrue(ready.wait(1.0))
            self.assertEqual(received[0].gesture, Gesture.LEFT)
            self.assertAlmostEqual(received[0].emg_activation, 0.66, places=5)
        finally:
            client.close()
            server.stop()


class GestureStateMachineTests(unittest.TestCase):
    def frame(self, timestamp: int, gesture: Gesture) -> GestureFrame:
        return GestureFrame(timestamp, gesture, 0.99, 0.6, 0.7)

    def test_direct_transition_does_not_need_none(self) -> None:
        state = GestureStateMachine(stable_ms=100)
        self.assertEqual(state.update(self.frame(0, Gesture.UP)), [])
        events = state.update(self.frame(100, Gesture.UP))
        self.assertEqual([(event.kind, event.gesture) for event in events], [
            (EventKind.EXIT, Gesture.NONE),
            (EventKind.ENTER, Gesture.UP),
        ])
        self.assertEqual(state.update(self.frame(150, Gesture.LEFT))[0:0], [])
        events = state.update(self.frame(250, Gesture.LEFT))
        self.assertEqual(events[0].kind, EventKind.EXIT)
        self.assertEqual(events[0].gesture, Gesture.UP)
        self.assertEqual(events[1].kind, EventKind.ENTER)
        self.assertEqual(events[1].gesture, Gesture.LEFT)

    def test_same_state_only_emits_hold(self) -> None:
        state = GestureStateMachine(stable_ms=0)
        state.update(self.frame(0, Gesture.RIGHT))
        events = state.update(self.frame(30, Gesture.RIGHT))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].kind, EventKind.HOLD)

    def test_stale_frame_is_ignored(self) -> None:
        state = GestureStateMachine(stable_ms=0)
        state.update(self.frame(100, Gesture.RIGHT))
        self.assertEqual(state.update(self.frame(90, Gesture.LEFT)), [])
        self.assertEqual(state.stable, Gesture.RIGHT)


if __name__ == "__main__":
    unittest.main()
