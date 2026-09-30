"""Read back Song structural trial rows against immutable F0 predictions."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1] / "benchmarks" / "song_real8"
CLASSES = ("fist", "index_pinch", "neutral", "open_hand")


def test_song_structural_rows_and_parent_rule_provenance():
    result = json.loads((ROOT / "F9_STRUCTURAL_SONG_RESULTS.json").read_text(encoding="utf-8"))
    protocol = json.loads((ROOT / "F9_STRUCTURAL_SONG_PROTOCOL.json").read_text(encoding="utf-8"))
    parent = json.loads((ROOT / "QUALITY_MASK_V1.json").read_text(encoding="utf-8"))
    assert hashlib.sha256((ROOT / "F9_STRUCTURAL_SONG_PROTOCOL.json").read_bytes()).hexdigest() == result["protocol_sha256"]
    assert hashlib.sha256((ROOT / "QUALITY_MASK_V1.json").read_bytes()).hexdigest() == protocol["parent_quality_result_sha256"]
    assert result["parent_source_state_sha256"] == parent["source_state_sha256"]
    assert result["availability"] == parent["availability"]
    trial_file = ROOT / "F9_STRUCTURAL_SONG_TRIALS.csv"
    assert hashlib.sha256(trial_file.read_bytes()).hexdigest() == result["trial_rows_sha256"]
    with trial_file.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    with (ROOT / "F4_INCREMENT_TRIAL_PREDICTIONS.csv").open(encoding="utf-8", newline="") as stream:
        frozen = {(r["session"], r["trial_id"]): r for r in csv.DictReader(stream)
                  if r["arm"] == "F0" and r["session"] in ("S03", "S04")}
    assert len(rows) == len(frozen) == 284
    assert {(r["session"], r["trial_id"]) for r in rows} == set(frozen)
    for session, count in (("S03", 140), ("S04", 144)):
        subset = [r for r in rows if r["session"] == session]
        assert len(subset) == count
        errors = correct_rejected = error_rejected = synthetic = 0
        for row in subset:
            prior = frozen[(session, row["trial_id"])]
            assert row["truth"] == prior["truth"]
            probabilities = np.array([float(prior[f"p_{hand}"]) for hand in CLASSES])
            correct = CLASSES[int(probabilities.argmax())] == row["truth"]
            rejected = row["rejected"] == "True"
            assert row["f0_correct"] == str(correct)
            assert rejected == (int(row["structural_windows"]) > 0)
            assert 0 <= int(row["structural_windows"]) <= int(row["windows"])
            errors += not correct
            correct_rejected += rejected and correct
            error_rejected += rejected and not correct
            synthetic += row["synthetic_constant_ch1_detected"] == "True"
        summary = result["sessions"][session]
        assert summary["baseline_errors"] == errors
        assert summary["correct_rejected"] == correct_rejected
        assert summary["errors_rejected"] == error_rejected
        assert summary["rejected"] == correct_rejected + error_rejected
        assert summary["coverage"] == (count - summary["rejected"]) / count
        assert summary["synthetic_constant_channel_trials_detected"] == synthetic
