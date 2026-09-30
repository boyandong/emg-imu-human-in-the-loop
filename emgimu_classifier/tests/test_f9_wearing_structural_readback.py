"""Read back re-wearing F9 trial flags against frozen F0v2 probabilities."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1] / "benchmarks" / "new_bank_v2"
CLASSES = (0, 1, 2, 3, 4)


def test_wearing_structural_trial_inventory_and_scores():
    protocol = json.loads((ROOT / "F9_WEARING_STRUCTURAL_PROTOCOL.json").read_text(encoding="utf-8"))
    result = json.loads((ROOT / "F9_WEARING_STRUCTURAL_RESULTS.json").read_text(encoding="utf-8"))
    parent = json.loads((ROOT / "WEARING_RESULTS.json").read_text(encoding="utf-8"))
    assert hashlib.sha256((ROOT / "F9_WEARING_STRUCTURAL_PROTOCOL.json").read_bytes()).hexdigest() == result["protocol_sha256"]
    assert hashlib.sha256((ROOT / "WEARING_RESULTS.json").read_bytes()).hexdigest() == protocol["parent_results_sha256"]
    assert result["archive_sha256"] == parent["archive_sha256"]
    assert result["availability"] == {"adc_clipping": False, "line_noise": False,
                                      "low_frequency_pre_highpass": False}
    trial_file = ROOT / "F9_WEARING_STRUCTURAL_TRIALS.csv"
    assert hashlib.sha256(trial_file.read_bytes()).hexdigest() == result["trial_rows_sha256"]
    with trial_file.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    with (ROOT / "WEARING_TRIAL_PREDICTIONS.csv").open(encoding="utf-8", newline="") as stream:
        frozen = {(r["phase"], r["trial_id"]): r for r in csv.DictReader(stream)
                  if r["arm"] == "F0v2"}
    assert len(rows) == len(frozen) == 240
    assert {(r["phase"], r["trial_id"]) for r in rows} == set(frozen)
    for phase in ("validation", "final"):
        phase_rows = [r for r in rows if r["phase"] == phase]
        assert len(phase_rows) == 120
        for domain in ("trial_1", "trial_2", "trial_3", "trial_4"):
            subset = [r for r in phase_rows if r["domain"] == domain]
            assert len(subset) == 30
            errors = correct_rejected = errors_rejected = synthetic = 0
            for row in subset:
                parent_row = frozen[(phase, row["trial_id"])]
                assert row["subject"] == parent_row["subject"]
                assert row["domain"] == parent_row["domain"]
                assert row["label"] == parent_row["label"]
                p = np.array([float(parent_row[f"p_{c}"]) for c in CLASSES])
                correct = CLASSES[int(p.argmax())] == int(row["label"])
                rejected = row["rejected"] == "True"
                assert row["f0_correct"] == str(correct)
                assert rejected == (int(row["structural_windows"]) > 0)
                assert 0 <= int(row["structural_windows"]) <= int(row["windows"])
                errors += not correct
                correct_rejected += correct and rejected
                errors_rejected += not correct and rejected
                synthetic += row["synthetic_constant_ch1_detected"] == "True"
            summary = result["phases"][phase]["domains"][domain]
            assert summary["baseline_errors"] == errors
            assert summary["correct_rejected"] == correct_rejected
            assert summary["errors_rejected"] == errors_rejected
            assert summary["synthetic_constant_channel_trials_detected"] == synthetic
            assert summary["coverage"] == (30 - summary["rejected"]) / 30
        pooled = result["phases"][phase]["pooled"]
        assert pooled["trials"] == 120
        assert pooled["baseline_errors"] == sum(result["phases"][phase]["domains"][d]["baseline_errors"]
                                               for d in ("trial_1", "trial_2", "trial_3", "trial_4"))
