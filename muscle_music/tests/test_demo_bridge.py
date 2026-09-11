from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from jilv.osc import OscError, encode_message
from serve_demo import LiveInputState, STALE_AFTER_SECONDS, parse_music_state


class DemoBridgeTests(unittest.TestCase):
    def test_parses_backend_music_state_packet(self) -> None:
        packet = encode_message("/emgimu/state", 2**40, 4, 0.93, 0.67, 0.0)
        frame = parse_music_state(packet)
        self.assertEqual(frame["timestamp_ms"], 2**40)
        self.assertEqual(frame["gesture_id"], 4)
        self.assertAlmostEqual(frame["confidence"], 0.93, places=5)
        self.assertAlmostEqual(frame["motion_energy"], 0.67, places=5)

    def test_rejects_other_osc_addresses(self) -> None:
        packet = encode_message("/other/state", 1, 0, 1.0, 0.0, 0.0)
        with self.assertRaises(OscError):
            parse_music_state(packet)

    def test_live_state_reports_200_hz_and_staleness(self) -> None:
        state = LiveInputState(9000)
        self.assertFalse(state.wait_snapshot(-1, 0)["connected"])
        state.update(parse_music_state(
            encode_message("/emgimu/state", 12, 1, 0.8, 0.4, 0.0)
        ))
        snapshot = state.wait_snapshot(-1, 0)
        self.assertTrue(snapshot["connected"])
        self.assertEqual(snapshot["sample_rate_hz"], 200)
        state._last_received = time.monotonic() - STALE_AFTER_SECONDS - 0.1
        self.assertFalse(state.wait_snapshot(snapshot["revision"], 0)["connected"])


if __name__ == "__main__":
    unittest.main()
