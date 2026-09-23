"""Boundary checks for cue-relative timing, separate from movement onset."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "collection" / "emg_meta" / "emg_meta"))

from benchmarks.song_cue_response_audit import score_cue_response
from emgforce.inference.song_local import LABELS


def trial(stable_start: int = 125, stable_end: int = 225):
    return {"trial_kind": "formal", "valid": True, "completion_status": "completed",
            "label": "still_fist", "prompt_start_sample": 100,
            "prompt_end_sample": 250, "stable_start_sample": stable_start,
            "stable_end_sample": stable_end}


class CueResponseAuditTest(unittest.TestCase):
    def test_first_persistent_hit_uses_frame_end_and_keeps_decoder_history(self):
        indices = np.arange(49, 274, 25)
        probabilities = np.zeros((len(indices), len(LABELS)))
        probabilities[:, LABELS.index("neutral")] = 1
        probabilities[indices >= 124] = 0
        probabilities[indices >= 124, LABELS.index("fist")] = 1
        result = score_cue_response([trial()], indices, probabilities)
        row = result["trial_rows"][0]
        self.assertFalse(row["already_correct_before_cue"])
        self.assertEqual(row["raw_latency_ms"], 96)
        self.assertEqual(row["decoded_latency_ms"], 296)
        self.assertTrue(row["decoded_correct_in_stable_cue"])

    def test_short_stable_interval_is_counted_but_not_scored_as_stable(self):
        indices = np.arange(49, 274, 25)
        probabilities = np.zeros((len(indices), len(LABELS)))
        probabilities[:, LABELS.index("fist")] = 1
        result = score_cue_response([trial(126, 150)], indices, probabilities)
        row = result["trial_rows"][0]
        self.assertTrue(row["already_correct_before_cue"])
        self.assertFalse(row["stable_cue_evaluable"])
        self.assertEqual(result["all"]["stable_cue_evaluable"], 0)


if __name__ == "__main__":
    unittest.main()
