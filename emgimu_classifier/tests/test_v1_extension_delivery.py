"""Independently replay saved cross-dataset first-stage family screens."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.grabmyo_crossday.run import score
from benchmarks.new_bank_v2.roam_posture_run import sha256


ROOT = Path(__file__).resolve().parents[1] / "benchmarks" / "new_bank_v2"


@pytest.mark.parametrize("prefix,parent,count,classes,label", [
    ("ROAM_V1_EXTENSION", "ROAM_POSTURE", 1800, [0, 1, 2], "label"),
    ("GRAB_V1_EXTENSION", "GRAB_USER", 560, [4, 15, 16, 17], "gesture"),
])
def test_saved_extension_replays_scores_and_frozen_baseline(prefix, parent, count, classes, label):
    result = json.loads((ROOT / f"{prefix}_RESULTS.json").read_text(encoding="utf-8"))
    protocol = json.loads((ROOT / f"{prefix}_PROTOCOL.json").read_text(encoding="utf-8"))
    with (ROOT / f"{prefix}_PREDICTIONS.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert result["protocol"] == protocol
    assert result["protocol_sha256"] == sha256(ROOT / f"{prefix}_PROTOCOL.json")
    assert result["prediction_sha256"] == sha256(ROOT / f"{prefix}_PREDICTIONS.csv")
    assert len(rows) == count
    assert set(row["arm"] for row in rows) == set(protocol["arms"])
    keys = [(row["arm"], row["phase"], row["trial_id"]) for row in rows]
    assert len(set(keys)) == count
    assert not (set(result["source_trial_ids"]) &
                (set(result["validation_trial_ids"]) | set(result["final_trial_ids"])))
    with (ROOT / f"{parent}_PREDICTIONS.csv").open(newline="", encoding="utf-8") as stream:
        old = {(r["phase"], r["trial_id"]): r for r in csv.DictReader(stream)
               if r["arm"] == "F0v2"}
    baseline = [r for r in rows if r["arm"] == "F0v2"]
    assert len(baseline) == count // len(protocol["arms"]) == len(old)
    for row in baseline:
        prior = old[(row["phase"], row["trial_id"])]
        assert int(row[label]) == int(prior[label])
        np.testing.assert_allclose([float(row[f"p_{c}"]) for c in classes],
                                   [float(prior[f"p_{c}"]) for c in classes],
                                   atol=1e-10, rtol=0)
    for arm in protocol["arms"]:
        for phase in ("validation", "final"):
            group = [r for r in rows if r["arm"] == arm and r["phase"] == phase]
            expected_ids = set(result[f"{phase}_trial_ids"])
            assert {r["trial_id"] for r in group} == expected_ids
            y = np.asarray([int(r[label]) for r in group])
            subjects = np.asarray([int(r["subject"]) for r in group])
            p = np.asarray([[float(r[f"p_{c}"]) for c in classes] for r in group])
            assert np.isfinite(p).all() and (p >= 0).all()
            np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-12, rtol=0)
            replay = score(y, p, np.asarray(classes), subjects)
            recorded = result["scores"][arm][phase]
            for key in ("accuracy", "macro_f1", "log_loss", "brier",
                        "minimum_subject_macro_f1"):
                assert replay[key] == pytest.approx(recorded[key], abs=1e-12)
            for key in ("per_class_recall", "per_subject_macro_f1"):
                assert replay[key] == pytest.approx(recorded[key], abs=1e-12)
            if prefix == "ROAM_V1_EXTENSION":
                for posture in protocol["target_postures"]:
                    mask = np.asarray([r["condition"] == posture for r in group])
                    posture_replay = score(y[mask], p[mask], np.asarray(classes), subjects[mask])
                    for key in ("macro_f1", "log_loss", "brier"):
                        assert posture_replay[key] == pytest.approx(
                            recorded["by_posture"][posture][key], abs=1e-12)
