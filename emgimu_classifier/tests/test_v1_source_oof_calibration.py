"""Guard source-subject OOF calibration and its frozen-target read-back."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.new_bank_v2.roam_posture_run import sha256


ROOT = Path(__file__).resolve().parents[1] / "benchmarks" / "new_bank_v2"
SPECS = {"roam_posture": ("ROAM_V1_EXTENSION", [0, 1, 2], 18, 162, 180),
         "grab_user": ("GRAB_V1_EXTENSION", [4, 15, 16, 17], 4, 112, 56),
         "grab_day": ("GRAB_DAY_V1_EXTENSION", [4, 15, 16, 17], 8, 224, 224)}


def test_source_subject_folds_and_temperature_grid_replay():
    result = json.loads((ROOT / "V1_SOURCE_OOF_CAL_RESULTS.json").read_text(encoding="utf-8"))
    predictions = ROOT / "V1_SOURCE_OOF_CAL_PREDICTIONS.csv"
    assert result["protocol_sha256"] == sha256(ROOT / "V1_SOURCE_OOF_CAL_PROTOCOL.json")
    assert result["prediction_sha256"] == sha256(predictions)
    with predictions.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == result["prediction_rows"] == 7090
    assert len({(r["study"], r["partition"], r["arm"], r["trial_id"]) for r in rows}) == 7090
    candidates = np.asarray(result["temperature_candidates"])
    for study, (prefix, classes, subject_count, source_count, target_count) in SPECS.items():
        parent = json.loads((ROOT / f"{prefix}_RESULTS.json").read_text(encoding="utf-8"))
        folds = result["studies"][study]["source_folds"]
        assert len(folds) == subject_count
        assert {fold["held_subject"] for fold in folds} == set(result["studies"][study]["source_subjects"])
        assert all(fold["held_subject"] not in fold["train_subjects"] for fold in folds)
        assert {identity for fold in folds for identity in fold["held_trial_ids"]} == set(
            parent["source_trial_ids"])
        assert sum(fold["held_trials"] for fold in folds) == source_count
        assert not (set(parent["source_trial_ids"]) &
                    (set(parent["validation_trial_ids"]) | set(parent["final_trial_ids"])))
        for arm, outcome in result["studies"][study]["arms"].items():
            source = [r for r in rows if r["study"] == study and r["partition"] == "source_oof"
                      and r["arm"] == arm]
            assert len(source) == source_count
            assert {r["trial_id"] for r in source} == set(parent["source_trial_ids"])
            y = np.asarray([int(r["label"]) for r in source])
            raw = np.asarray([[float(r[f"raw_p_{c}"]) for c in classes] for r in source])
            assert np.isfinite(raw).all() and (raw > 0).all()
            np.testing.assert_allclose(raw.sum(axis=1), 1, atol=1e-12, rtol=0)
            loss = []
            for temperature in candidates:
                power = np.clip(raw, 1e-12, 1) ** (1 / temperature)
                p = power / power.sum(axis=1, keepdims=True)
                indices = np.searchsorted(classes, y)
                loss.append((float(-np.log(p[np.arange(len(y)), indices]).mean()),
                             abs(float(np.log(temperature))), float(temperature)))
            assert min(loss)[2] == outcome["temperature"]
            assert outcome["source_oof_calibrated"]["log_loss"] <= (
                outcome["source_oof_raw"]["log_loss"] + 1e-12)
            for phase in ("validation", "final"):
                target = [r for r in rows if r["study"] == study and r["partition"] == phase
                          and r["arm"] == arm]
                assert len(target) == target_count
                assert {r["trial_id"] for r in target} == set(parent[f"{phase}_trial_ids"])
                assert outcome["target"][phase]["raw"]["macro_f1"] == pytest.approx(
                    outcome["target"][phase]["calibrated"]["macro_f1"], abs=1e-12)


def test_source_oof_calibration_delivery_matches_readback():
    verification = json.loads((ROOT / "V1_SOURCE_OOF_CAL_VERIFICATION.json").read_text(
        encoding="utf-8"))
    delivery = json.loads((ROOT / "V1_SOURCE_OOF_CAL_DELIVERY_AUDIT.json").read_text(
        encoding="utf-8"))
    cells = ROOT / "V1_SOURCE_OOF_CAL_CELLS.csv"
    assert verification["status"] == delivery["status"] == "ok"
    assert verification["source_result_sha256"] == sha256(ROOT / "V1_SOURCE_OOF_CAL_RESULTS.json")
    assert verification["source_prediction_sha256"] == sha256(
        ROOT / "V1_SOURCE_OOF_CAL_PREDICTIONS.csv")
    assert verification["cell_sha256"] == delivery["source_cells_sha256"] == sha256(cells)
    assert delivery["source_verification_sha256"] == sha256(
        ROOT / "V1_SOURCE_OOF_CAL_VERIFICATION.json")
    assert verification["target_score_cells"] == delivery["rows"] == 440
    source = ROOT.parents[1] / "feature_bank" / "results" / "calibration_curve.csv"
    # The delivery hash is a snapshot of the whole source table at export time.
    # Later independent experiments append rows, so verify this run's actual
    # canonical source rows against its immutable cells instead.
    with source.open(newline="", encoding="utf-8") as stream:
        exported = [r for r in csv.DictReader(stream) if r["run_id"] == delivery["run_id"]]
    assert len(exported) == 440
    assert {r["shots_per_class"] for r in exported} == {"0"}
    assert {r["method"] for r in exported} == {"uncalibrated", "source_oof_temperature"}
    with cells.open(newline="", encoding="utf-8") as stream:
        expected = list(csv.DictReader(stream))
    def key(row):
        return (row["phase"], row["subject"], row["feature_bank"] if "feature_bank" in row else row["arm"],
                row["method"], row["aggregation"] if "aggregation" in row else row["scope"],
                row["macro_f1"], row["log_loss"], row["brier"], row["ece"])
    assert len({key(row) for row in exported}) == 440
    assert {key(row) for row in exported} == {key(row) for row in expected}
