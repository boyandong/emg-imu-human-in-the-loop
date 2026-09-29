"""Independently replay saved cross-dataset first-stage family screens."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.grabmyo_crossday.run import score
from benchmarks.new_bank_v2.roam_posture_run import sha256


ROOT = Path(__file__).resolve().parents[1] / "benchmarks" / "new_bank_v2"


@pytest.mark.parametrize("prefix,parent,count,classes,label,parent_arm,parent_phase,tolerance", [
    ("ROAM_V1_EXTENSION", "ROAM_POSTURE", 1800, [0, 1, 2], "label", "F0v2", "phase", 1e-10),
    ("GRAB_V1_EXTENSION", "GRAB_USER", 560, [4, 15, 16, 17], "gesture", "F0v2", "phase", 1e-10),
    ("GRAB_DAY_V1_EXTENSION", "GRABMYO_TRIAL", 2240, [4, 15, 16, 17], "gesture", "F0", "split", 1e-8),
])
def test_saved_extension_replays_scores_and_frozen_baseline(
        prefix, parent, count, classes, label, parent_arm, parent_phase, tolerance):
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
    parent_file = ("GRABMYO_TRIAL_PREDICTIONS.csv" if parent == "GRABMYO_TRIAL"
                   else f"{parent}_PREDICTIONS.csv")
    with (ROOT / parent_file).open(newline="", encoding="utf-8") as stream:
        old = {(r[parent_phase], r["trial_id"]): r for r in csv.DictReader(stream)
               if r["arm"] == parent_arm}
    baseline = [r for r in rows if r["arm"] == "F0v2"]
    assert len(baseline) == count // len(protocol["arms"]) == len(old)
    for row in baseline:
        prior = old[(row["phase"], row["trial_id"])]
        assert int(row[label]) == int(prior[label])
        np.testing.assert_allclose([float(row[f"p_{c}"]) for c in classes],
                                   [float(prior[f"p_{c}"]) for c in classes],
                                   atol=tolerance, rtol=0)
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


def test_cross_study_paired_cells_have_matched_error_accounting():
    audit = json.loads((ROOT / "V1_EXTENSION_PAIRED_AUDIT.json").read_text(encoding="utf-8"))
    path = ROOT / "V1_EXTENSION_PAIRED.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert audit["status"] == "ok"
    assert audit["paired_csv_sha256"] == sha256(path)
    assert len(rows) == audit["rows"] == 176
    assert audit["group_counts"] == {"roam_posture": 80, "grab_user": 24, "grab_day": 72}
    for study, prefix in (("roam_posture", "ROAM_V1_EXTENSION"),
                          ("grab_user", "GRAB_V1_EXTENSION"),
                          ("grab_day", "GRAB_DAY_V1_EXTENSION")):
        assert audit["source_hashes"][study]["prediction_sha256"] == sha256(
            ROOT / f"{prefix}_PREDICTIONS.csv")
        assert audit["source_hashes"][study]["result_sha256"] == sha256(
            ROOT / f"{prefix}_RESULTS.json")
    for row in rows:
        n = int(row["n_trials"])
        corrected = int(row["base_wrong_added_right"])
        created = int(row["base_right_added_wrong"])
        both_wrong = int(row["both_wrong"])
        both_right = int(row["both_right"])
        disagreement = int(row["prediction_disagreement"])
        assert n == corrected + created + both_wrong + both_right
        assert corrected + created <= disagreement <= n
        assert float(row["delta_macro_f1"]) == pytest.approx(
            float(row["added_macro_f1"]) - float(row["base_macro_f1"]), abs=1e-12)
        assert float(row["log_loss_improvement"]) == pytest.approx(
            float(row["base_log_loss"]) - float(row["added_log_loss"]), abs=1e-12)
        assert float(row["brier_improvement"]) == pytest.approx(
            float(row["base_brier"]) - float(row["added_brier"]), abs=1e-12)


def test_paired_cells_are_in_canonical_source_tables():
    audit = json.loads((ROOT / "V1_EXTENSION_DELIVERY_AUDIT.json").read_text(encoding="utf-8"))
    assert audit["status"] == "ok"
    assert audit["paired_csv_sha256"] == sha256(ROOT / "V1_EXTENSION_PAIRED.csv")
    assert audit["paired_audit_sha256"] == sha256(ROOT / "V1_EXTENSION_PAIRED_AUDIT.json")
    source = ROOT.parents[1] / "feature_bank" / "results"
    for name in ("conditional_incremental.csv", "error_complementarity.csv"):
        path = source / name
        assert audit["exports"][name]["sha256"] == sha256(path)
        with path.open(newline="", encoding="utf-8") as stream:
            rows = [r for r in csv.DictReader(stream) if r["run_id"] == audit["run_id"]]
        assert len(rows) == audit["exports"][name]["rows"] == 176
        assert len({(r["dataset"], r["session/domain"], r["phase"], r["scope"], r["subject"],
                     r["condition"], r.get("added_family", r.get("family_b")))
                    for r in rows}) == 176


def test_ring_frequency_four_arm_interaction_replays_and_exports():
    result = json.loads((ROOT / "RING_FREQ_INTERACTION_RESULTS.json").read_text(encoding="utf-8"))
    audit = json.loads((ROOT / "RING_FREQ_INTERACTION_AUDIT.json").read_text(encoding="utf-8"))
    delivery = json.loads((ROOT / "RING_FREQ_INTERACTION_DELIVERY_AUDIT.json").read_text(encoding="utf-8"))
    predictions = ROOT / "RING_FREQ_INTERACTION_PREDICTIONS.csv"
    table = ROOT / "RING_FREQ_INTERACTION.csv"
    assert result["protocol_sha256"] == sha256(ROOT / "RING_FREQ_INTERACTION_PROTOCOL.json")
    assert result["prediction_sha256"] == audit["prediction_sha256"] == sha256(predictions)
    assert audit["result_sha256"] == sha256(ROOT / "RING_FREQ_INTERACTION_RESULTS.json")
    assert audit["table_sha256"] == delivery["source_sha256"] == sha256(table)
    assert audit["rows"] == delivery["rows"] == 44
    with predictions.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    with table.open(newline="", encoding="utf-8") as stream:
        cells = list(csv.DictReader(stream))
    assert len(rows) == result["prediction_rows"] == 3680
    assert len({(r["study"], r["arm"], r["phase"], r["trial_id"]) for r in rows}) == 3680
    for cell in cells:
        assert float(cell["S_macro_f1"]) == pytest.approx(
            float(cell["joint_macro_f1"]) - float(cell["ring_macro_f1"])
            - float(cell["frequency_macro_f1"]) + float(cell["base_macro_f1"]), abs=1e-12)
        assert float(cell["S_negative_logloss"]) == pytest.approx(
            float(cell["ring_log_loss"]) + float(cell["frequency_log_loss"])
            - float(cell["joint_log_loss"]) - float(cell["base_log_loss"]), abs=1e-12)
        assert float(cell["joint_log_loss_improvement"]) == pytest.approx(
            float(cell["base_log_loss"]) - float(cell["joint_log_loss"]), abs=1e-12)
    source = ROOT.parents[1] / "feature_bank" / "results" / "interaction_results.csv"
    assert delivery["output_sha256"] == sha256(source)
    with source.open(newline="", encoding="utf-8") as stream:
        exported = [r for r in csv.DictReader(stream) if r["run_id"] == delivery["run_id"]]
    assert len(exported) == 44


def test_ring_frequency_full_bank_lofo_replays_and_delivers():
    result = json.loads((ROOT / "RING_FREQ_LOFO_RESULTS.json").read_text(encoding="utf-8"))
    verify = json.loads((ROOT / "RING_FREQ_LOFO_VERIFICATION.json").read_text(encoding="utf-8"))
    delivery = json.loads((ROOT / "RING_FREQ_LOFO_DELIVERY_AUDIT.json").read_text(encoding="utf-8"))
    predictions = ROOT / "RING_FREQ_LOFO_PREDICTIONS.csv"
    cells = ROOT / "RING_FREQ_LOFO_CELLS.csv"
    parent = ROOT / "RING_FREQ_INTERACTION_PREDICTIONS.csv"
    assert result["protocol_sha256"] == verify["protocol_sha256"] == sha256(
        ROOT / "RING_FREQ_LOFO_PROTOCOL.json")
    assert result["prediction_sha256"] == verify["prediction_sha256"] == sha256(predictions)
    assert result["parent_prediction_sha256"] == verify["parent_prediction_sha256"] == sha256(parent)
    assert verify["cell_sha256"] == delivery["source_sha256"] == sha256(cells)
    assert verify["prediction_rows"] == result["prediction_rows"] == 3680
    assert verify["parent_copied_rows"] == 2760
    assert verify["score_cells"] == delivery["rows"] == 176
    with predictions.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len({(r["study"], r["arm"], r["phase"], r["trial_id"]) for r in rows}) == 3680
    with cells.open(newline="", encoding="utf-8") as stream:
        scored = list(csv.DictReader(stream))
    full = {(r["study"], r["phase"], r["group_type"], r["group"]): r
            for r in scored if r["removed_family"] == "NONE"}
    assert len(full) == 44
    for row in scored:
        reference = full[(row["study"], row["phase"], row["group_type"], row["group"])]
        assert float(row["full_minus_removed_macro_f1"]) == pytest.approx(
            float(reference["macro_f1"]) - float(row["macro_f1"]), abs=1e-12)
        assert float(row["removed_minus_full_log_loss"]) == pytest.approx(
            float(row["log_loss"]) - float(reference["log_loss"]), abs=1e-12)
        assert float(row["removed_minus_full_brier"]) == pytest.approx(
            float(row["brier"]) - float(reference["brier"]), abs=1e-12)
    source = ROOT.parents[1] / "feature_bank" / "results" / "ablation_full_bank.csv"
    assert delivery["output_sha256"] == sha256(source)
    with source.open(newline="", encoding="utf-8") as stream:
        exported = [r for r in csv.DictReader(stream) if r["run_id"] == delivery["run_id"]]
    assert len(exported) == 176


def test_three_study_independent_family_screen_is_delivered():
    audit = json.loads((ROOT / "V1_EXTENSION_FAMILY_AUDIT.json").read_text(encoding="utf-8"))
    delivery = json.loads((ROOT / "V1_EXTENSION_FAMILY_DELIVERY_AUDIT.json").read_text(
        encoding="utf-8"))
    table = ROOT / "V1_EXTENSION_FAMILY_SCREEN.csv"
    assert audit["status"] == delivery["status"] == "ok"
    assert audit["table_sha256"] == delivery["screen_sha256"] == sha256(table)
    assert delivery["screen_audit_sha256"] == sha256(ROOT / "V1_EXTENSION_FAMILY_AUDIT.json")
    assert audit["rows"] == delivery["rows"] == 220
    assert audit["study_rows"] == {"roam_posture": 100, "grab_user": 30, "grab_day": 90}
    for study, prefix in (("roam_posture", "ROAM_V1_EXTENSION"),
                          ("grab_user", "GRAB_V1_EXTENSION"),
                          ("grab_day", "GRAB_DAY_V1_EXTENSION")):
        assert audit["sources"][study]["result_sha256"] == sha256(ROOT / f"{prefix}_RESULTS.json")
        assert audit["sources"][study]["prediction_sha256"] == sha256(
            ROOT / f"{prefix}_PREDICTIONS.csv")
    with table.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len({(r["study"], r["phase"], r["scope"], r["subject"],
                 r["condition"], r["feature_family"]) for r in rows}) == 220
    for row in rows:
        assert 0 <= float(row["ece"]) <= 1
        assert int(row["feature_dimension"]) > 0 and int(row["evaluation_trials"]) > 0
        classes = {"0", "1", "2"} if row["study"] == "roam_posture" else {"4", "15", "16", "17"}
        assert set(json.loads(row["per_class_f1_json"])) == classes
    source = ROOT.parents[1] / "feature_bank" / "results" / "feature_family_results.csv"
    assert delivery["output_sha256"] == sha256(source)
    with source.open(newline="", encoding="utf-8") as stream:
        exported = [r for r in csv.DictReader(stream) if r["run_id"] == delivery["run_id"]]
    assert len(exported) == 220
