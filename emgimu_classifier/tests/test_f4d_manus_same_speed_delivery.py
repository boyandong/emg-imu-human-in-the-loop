"""Read back the matched-speed F4d protocol without rerunning training."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from benchmarks.grabmyo_crossday.run import score
from benchmarks.new_bank_v2.roam_posture_run import sha256

ROOT = Path(__file__).resolve().parents[1] / "benchmarks/new_bank_v2"


def read_rows(name: str) -> list[dict[str, str]]:
    with (ROOT / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_same_speed_f4d_native_trials_are_disjoint_and_scores_read_back() -> None:
    protocol_path = ROOT / "F4D_MANUS_SAME_SPEED_PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    result = json.loads((ROOT / "F4D_MANUS_SAME_SPEED_RESULTS.json").read_text(encoding="utf-8"))
    predictions_path = ROOT / "F4D_MANUS_SAME_SPEED_PREDICTIONS.csv"
    rows = read_rows(predictions_path.name)
    previous = read_rows("F4D_MANUS_PREDICTIVE_PREDICTIONS.csv")
    previous_by_key = {(row["phase"], row["user"], row["trial_id"], row["arm"]): row
                       for row in previous}
    assert result["protocol_sha256"] == sha256(protocol_path)
    assert protocol["parent_predictive_protocol_sha256"] == sha256(
        ROOT / protocol["parent_predictive_protocol"])
    assert result["parent_result_sha256"] == sha256(ROOT / "F4D_MANUS_SESSION_RESULTS.json")
    assert result["prediction_sha256"] == sha256(predictions_path)
    assert result["prediction_rows"] == len(rows) == 432
    classes = np.arange(6)
    for row in rows:
        trial = row["trial_id"]
        calibration = row["calibration_trial_ids"].split("|")
        assert len(calibration) == len(set(calibration)) == 5
        assert trial not in calibration
        user, session, speed = row["user"], "2" if row["phase"] == "validation" else "3", row["speed"]
        for identifier in (trial, *calibration):
            assert f"/u_{user}/s_{session}/" in identifier
            assert f"/recording_{speed}_" in identifier
        gestures = [identifier.split("/")[3] for identifier in (trial, *calibration)]
        assert len(set(gestures)) == 6
        if row["arm"] in ("F0v2", "F0v2+F4d_long"):
            older = previous_by_key[(row["phase"], user, trial, row["arm"])]
            np.testing.assert_allclose([float(row[f"p_{c}"]) for c in classes],
                                       [float(older[f"p_{c}"]) for c in classes], atol=1e-10)

    for phase in ("validation", "final"):
        for arm in ("F0v2", "F0v2+F4d_long", "F0v2+F4d_same_speed"):
            selected = [row for row in rows if row["phase"] == phase and row["arm"] == arm]
            assert len(selected) == 72
            y = np.asarray([int(row["label"]) for row in selected])
            p = np.asarray([[float(row[f"p_{c}"]) for c in classes] for row in selected])
            users = np.asarray([int(row["user"]) for row in selected])
            np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-10)
            observed = score(y, p, classes, users)
            saved = result["scores"][phase][arm]["pooled"]
            for key in ("macro_f1", "log_loss", "brier", "minimum_subject_macro_f1"):
                assert abs(observed[key] - saved[key]) < 1e-12
