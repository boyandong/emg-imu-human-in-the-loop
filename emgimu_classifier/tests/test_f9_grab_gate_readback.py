"""Read back the frozen GRAB F0/F9 matched-trial rejection evidence."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1] / "benchmarks" / "new_bank_v2"


def test_grab_gate_rows_replay_frozen_parent_and_phase_summaries():
    result = json.loads((ROOT / "F9_GRAB_GATE_RESULTS.json").read_text(encoding="utf-8"))
    parent = json.loads((ROOT / "GRAB_USER_RESULTS.json").read_text(encoding="utf-8"))
    protocol = json.loads((ROOT / "F9_GRAB_GATE_PROTOCOL.json").read_text(encoding="utf-8"))
    rows_path = ROOT / "F9_GRAB_GATE_TRIALS.csv"
    assert hashlib.sha256(rows_path.read_bytes()).hexdigest() == result["trial_rows_sha256"]
    assert hashlib.sha256((ROOT / "GRAB_USER_PREDICTIONS.csv").read_bytes()).hexdigest() == result["parent_prediction_sha256"]
    assert hashlib.sha256((ROOT / "F9_GRAB_GATE_PROTOCOL.json").read_bytes()).hexdigest() == result["protocol_sha256"]
    assert protocol["source_quantile"] == .995
    assert result["source_trials"] == 112 and result["source_windows"] == 2240
    assert result["availability"] == {"adc_clipping": False, "line_noise": False,
                                      "low_frequency_pre_highpass": False}
    with (ROOT / "GRAB_USER_PREDICTIONS.csv").open(encoding="utf-8", newline="") as stream:
        frozen = {(row["phase"], row["trial_id"]): row for row in csv.DictReader(stream)
                  if row["arm"] == "F0v2"}
    with rows_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(frozen) == 112
    assert {(row["phase"], row["trial_id"]) for row in rows} == set(frozen)
    for phase in ("validation", "final"):
        phase_rows = [row for row in rows if row["phase"] == phase]
        assert {row["trial_id"] for row in phase_rows} == set(parent[f"{phase}_trial_ids"])
        correct_rejected = errors_rejected = errors = 0
        for row in phase_rows:
            parent_row = frozen[(phase, row["trial_id"])]
            assert row["subject"] == parent_row["subject"]
            assert row["gesture"] == parent_row["gesture"]
            probabilities = np.array([float(parent_row[f"p_{label}"]) for label in (4,15,16,17)])
            correct = (4,15,16,17)[int(np.argmax(probabilities))] == int(row["gesture"])
            rejected = float(row["min_quality"]) < .5
            assert 0. <= float(row["min_quality"]) <= 1.
            assert row["f0_correct"] == str(correct)
            assert row["rejected"] == str(rejected)
            errors += int(not correct)
            correct_rejected += int(rejected and correct)
            errors_rejected += int(rejected and not correct)
        summary = result["phases"][phase]["pooled"]
        assert summary["trials"] == len(phase_rows) == 56
        assert summary["baseline_errors"] == errors
        assert summary["correct_rejected"] == correct_rejected
        assert summary["errors_rejected"] == errors_rejected
        assert summary["rejected"] == correct_rejected + errors_rejected
        assert summary["accepted"] == 56 - summary["rejected"]
        assert summary["coverage"] == summary["accepted"] / 56
