import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from emgimu.service import RawSample, RuntimeHealthMonitor, StateCsvLogger
from emgimu.state import (
    Confidence, Consistency, Direction, Gesture, HumanState, Phase, PhasePair,
    QualityFlag, SignalQuality,
)


class ServiceLoggingTests(unittest.TestCase):
    @staticmethod
    def _state(*, direction=Direction.UNKNOWN, gesture=Gesture.UNKNOWN, flags=QualityFlag.NONE):
        return HumanState(
            120, direction, gesture, 0.2,
            PhasePair(Phase.UNKNOWN, Phase.UNKNOWN), 0.1,
            Consistency.UNKNOWN, Confidence(0.1, 0.2, 0.0, 0.0),
            SignalQuality(0.8, 0.9, flags),
        )

    def test_csv_is_fit_consistency_compatible(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "observations.csv"
            with StateCsvLogger(path, run_id="run-1") as logger:
                logger.write(self._state(), -40.0)
            with path.open(encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["timestamp_ms"], "120")
            self.assertEqual(row["run_id"], "run-1")
            self.assertEqual(row["state_sequence"], "1")
            self.assertTrue(row["recorded_at_utc"].endswith("+00:00"))
            self.assertEqual(row["onset_lag_ms"], "-40.0")
            self.assertIn("activation", row)
            self.assertIn("direction_drift", row)

    def test_refuses_to_append_to_legacy_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "observations.csv"
            path.write_text("timestamp_ms,direction\n1,0\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema differs"):
                StateCsvLogger(path)

    def test_health_snapshot_is_atomic_and_tracks_rejection_reasons(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "health.json"
            monitor = RuntimeHealthMonitor(
                path, run_id="test-run", write_interval_seconds=3600,
            )
            monitor.observe_sample(RawSample(
                100, np.zeros(8), np.zeros(3), np.zeros(3),
            ))
            monitor.observe_state(self._state(
                flags=QualityFlag.MODEL_DRIFT | QualityFlag.LOW_CONFIDENCE,
            ), 4.0)
            monitor.observe_state(self._state(
                direction=Direction.RIGHT, gesture=Gesture.FIST,
                flags=QualityFlag.EMG_VALID | QualityFlag.IMU_VALID,
            ), 8.0)
            monitor.observe_error(ValueError("bad OSC"), parse=True)
            monitor.close()

            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "stopped")
            self.assertEqual(payload["run_id"], "test-run")
            self.assertEqual(payload["samples_received"], 1)
            self.assertEqual(payload["states_emitted"], 2)
            self.assertEqual(payload["direction_unknown_rate"], 0.5)
            self.assertEqual(payload["gesture_unknown_rate"], 0.5)
            self.assertEqual(payload["processing_latency_ms"]["p50"], 6.0)
            self.assertEqual(payload["quality_flag_counts"]["model_drift"], 1)
            self.assertEqual(payload["quality_flag_counts"]["low_confidence"], 1)
            self.assertEqual(payload["parse_errors"], 1)
            self.assertEqual(payload["processing_errors"], 0)
            self.assertEqual(payload["last_error"]["type"], "ValueError")
            self.assertFalse(list(Path(directory).glob("*.tmp")))

    def test_logger_rejects_invalid_durability_interval(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "positive"):
                StateCsvLogger(Path(directory) / "states.csv", fsync_every=0)


if __name__ == "__main__":
    unittest.main()
