"""Independent native-trial and metric readback for three V3 F2c axes."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, log_loss


ROOT = Path(__file__).resolve().parents[1] / "benchmarks"
V2 = ROOT / "new_bank_v2"
V3 = ROOT / "new_bank_v3"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_v3_f2c_wearing_readback() -> None:
    protocol = json.loads((V3 / "SPEC_F2C_CROSS_AXIS_PROTOCOL.json").read_text(encoding="utf-8"))
    result = json.loads((V3 / "SPEC_F2C_WEARING_RESULTS.json").read_text(encoding="utf-8"))
    assert result["protocol_sha256"] == digest(V3 / "SPEC_F2C_CROSS_AXIS_PROTOCOL.json")
    assert result["prediction_sha256"] == digest(V3 / "SPEC_F2C_WEARING_PREDICTIONS.csv")
    assert result["parent_result_sha256"] == digest(V2 / "F2A_DOCUMENT_WEARING_RESULTS.json")
    assert result["f0_parent_prediction_sha256"] == digest(V2 / "WEARING_TRIAL_PREDICTIONS.csv")
    assert result["feature_dimensions"] == {"F0v2": 48, "F2c_spec": 36}
    rows = read_rows(V3 / "SPEC_F2C_WEARING_PREDICTIONS.csv")
    parent = json.loads((V2 / "F2A_DOCUMENT_WEARING_RESULTS.json").read_text(encoding="utf-8"))
    assert len(rows) == result["prediction_rows"] == 240
    assert len({r["trial_id"] for r in rows}) == 240
    for phase in ("validation", "final"):
        selected = [r for r in rows if r["phase"] == phase]
        expected = {trial for key, part in parent["split_trial_ids"].items()
                    if key.startswith(phase + "_") for trial in part["target"]}
        assert {r["trial_id"] for r in selected} == expected
        assert {int(r["subject"]) for r in selected} == set(
            [15, 16, 17] if phase == "validation" else [18, 19, 20])
        labels = np.asarray([int(r["label"]) for r in selected])
        p = np.asarray([[float(r[f"p_{c}"]) for c in range(5)] for r in selected])
        assert np.isfinite(p).all() and (p >= 0).all()
        np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-12, rtol=0)
        score = result["scores"][phase]["pooled"]
        np.testing.assert_allclose(score["macro_f1"], f1_score(
            labels, p.argmax(axis=1), labels=range(5), average="macro"))
        np.testing.assert_allclose(score["log_loss"], log_loss(labels, p, labels=range(5)))
    assert protocol["candidate"] == "F0v2+F2c_spec"


def test_v3_f2c_transfer_readback() -> None:
    result = json.loads((V3 / "SPEC_F2C_TRANSFER_RESULTS.json").read_text(encoding="utf-8"))
    assert result["protocol_sha256"] == digest(V3 / "SPEC_F2C_CROSS_AXIS_PROTOCOL.json")
    assert result["prediction_sha256"] == digest(V3 / "SPEC_F2C_TRANSFER_PREDICTIONS.csv")
    rows = read_rows(V3 / "SPEC_F2C_TRANSFER_PREDICTIONS.csv")
    assert len(rows) == result["prediction_rows"] == 328
    assert len({(r["axis"], r["trial_id"]) for r in rows}) == 328
    parent_manus = json.loads((V2 / "F2_MANUS_CANDIDATES_RESULTS.json").read_text(encoding="utf-8"))
    parent_grab = json.loads((V2 / "F2_GRAB_CANDIDATES_RESULTS.json").read_text(encoding="utf-8"))
    assert {key: result["parent_result_sha256"][key] for key in ("manus_session", "grab_unseen_user")} == {
        "manus_session": digest(V2 / "F2_MANUS_CANDIDATES_RESULTS.json"),
        "grab_unseen_user": digest(V2 / "F2_GRAB_CANDIDATES_RESULTS.json"),
    }
    assert result["feature_dimensions"] == {"F0v2": 48, "F2c_spec": 36}
    for axis, parent, classes, count in (
        ("manus_session", parent_manus, [0, 1, 2, 3, 4, 5], 108),
        ("grab_unseen_user", parent_grab, [4, 15, 16, 17], 56),
    ):
        for phase in ("validation", "final"):
            chosen = [r for r in rows if r["axis"] == axis and r["phase"] == phase]
            assert len(chosen) == count
            expected = (parent["split_trial_ids"][phase]["target_trials"]
                        if axis == "manus_session" else parent[f"{phase}_trial_ids"])
            assert {r["trial_id"] for r in chosen} == set(expected)
            assert {r["class_order"] for r in chosen} == {"|".join(map(str, classes))}
            labels = np.asarray([int(r["label"]) for r in chosen])
            p = np.asarray([[float(r[f"p_{i}"]) for i in range(len(classes))]
                            for r in chosen])
            assert np.isfinite(p).all() and (p >= 0).all()
            np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-12, rtol=0)
            score = result["scores"][axis][phase]
            if axis == "manus_session":
                score = score["pooled"]
            np.testing.assert_allclose(score["macro_f1"], f1_score(
                labels, np.asarray(classes)[p.argmax(axis=1)], labels=classes,
                average="macro"))
            np.testing.assert_allclose(score["log_loss"], log_loss(
                labels, p, labels=classes))


def test_v3_f2c_cross_axis_decision() -> None:
    audit = json.loads((V3 / "SPEC_F2C_CROSS_AXIS_AUDIT.json").read_text(encoding="utf-8"))
    cells = read_rows(V3 / "SPEC_F2C_CROSS_AXIS_CELLS.csv")
    assert audit["protocol_sha256"] == digest(V3 / "SPEC_F2C_CROSS_AXIS_PROTOCOL.json")
    assert audit["cells_sha256"] == digest(V3 / "SPEC_F2C_CROSS_AXIS_CELLS.csv")
    assert len(cells) == audit["cells"] == 6
    assert {(r["axis"], r["phase"]) for r in cells} == {
        (a, p) for a in ("wearing_shift", "manus_session", "grab_unseen_user")
        for p in ("validation", "final")}
    violations = []
    for row in cells:
        np.testing.assert_allclose(float(row["delta_macro_f1"]),
            float(row["f2c_macro_f1"]) - float(row["f0_macro_f1"]))
        np.testing.assert_allclose(float(row["delta_log_loss"]),
            float(row["f2c_log_loss"]) - float(row["f0_log_loss"]))
        if row["phase"] == "validation":
            if float(row["delta_macro_f1"]) < -1e-12:
                violations.append(f"{row['axis']}:macro_f1")
            if float(row["delta_log_loss"]) > 1e-12:
                violations.append(f"{row['axis']}:log_loss")
    assert violations == audit["validation_violations"]
    assert not audit["candidate_eligible_for_public_default"]
    assert audit["three_axis_default"] == "F0v2"
