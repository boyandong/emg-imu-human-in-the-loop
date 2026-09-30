"""Independently read back the exploratory structural GRAB gate trial evidence."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1] / "benchmarks" / "new_bank_v2"
CLASSES = (4, 15, 16, 17)


def test_structural_gate_preserves_frozen_trial_and_prediction_identities():
    result = json.loads((ROOT / "F9_STRUCTURAL_GRAB_RESULTS.json").read_text(encoding="utf-8"))
    protocol = json.loads((ROOT / "F9_STRUCTURAL_GRAB_PROTOCOL.json").read_text(encoding="utf-8"))
    parent = json.loads((ROOT / "F9_GRAB_GATE_RESULTS.json").read_text(encoding="utf-8"))
    assert hashlib.sha256((ROOT / "F9_STRUCTURAL_GRAB_PROTOCOL.json").read_bytes()).hexdigest() == result["protocol_sha256"]
    assert hashlib.sha256((ROOT / "F9_GRAB_GATE_RESULTS.json").read_bytes()).hexdigest() == protocol["parent_results_sha256"]
    assert result["source_state_sha256"] == parent["source_state_sha256"]
    assert result["availability"] == parent["availability"]
    trial_file = ROOT / "F9_STRUCTURAL_GRAB_TRIALS.csv"
    assert hashlib.sha256(trial_file.read_bytes()).hexdigest() == result["trial_rows_sha256"]
    with trial_file.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    with (ROOT / "F9_GRAB_GATE_TRIALS.csv").open(encoding="utf-8", newline="") as stream:
        old = {(r["phase"], r["trial_id"]): r for r in csv.DictReader(stream)}
    with (ROOT / "GRAB_USER_PREDICTIONS.csv").open(encoding="utf-8", newline="") as stream:
        frozen = {(r["phase"], r["trial_id"]): r for r in csv.DictReader(stream)
                  if r["arm"] == "F0v2"}
    assert len(rows) == len(old) == len(frozen) == 112
    assert {(r["phase"], r["trial_id"]) for r in rows} == set(old) == set(frozen)
    for phase in ("validation", "final"):
        subset = [r for r in rows if r["phase"] == phase]
        assert len(subset) == 56
        correct_rejected = errors_rejected = baseline_errors = 0
        for row in subset:
            key = (phase, row["trial_id"])
            prior, prediction = old[key], frozen[key]
            assert row["subject"] == prior["subject"] == prediction["subject"]
            assert row["gesture"] == prior["gesture"] == prediction["gesture"]
            scores = np.array([float(prediction[f"p_{label}"]) for label in CLASSES])
            correct = CLASSES[int(scores.argmax())] == int(row["gesture"])
            rejected = row["rejected"] == "True"
            assert row["f0_correct"] == str(correct)
            assert row["parent_rejected"] == prior["rejected"]
            assert rejected == (int(row["structural_windows"]) > 0)
            baseline_errors += not correct
            correct_rejected += rejected and correct
            errors_rejected += rejected and not correct
        summary = result["phases"][phase]["pooled"]
        assert summary["baseline_errors"] == baseline_errors
        assert summary["correct_rejected"] == correct_rejected
        assert summary["errors_rejected"] == errors_rejected
        assert summary["rejected"] == correct_rejected + errors_rejected
        assert summary["coverage"] == (56 - summary["rejected"]) / 56
