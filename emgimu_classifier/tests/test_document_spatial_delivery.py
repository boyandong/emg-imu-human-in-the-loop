"""Read back native F2c/F3c document-consistent results and decision."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import f1_score, log_loss


ROOT=Path(__file__).resolve().parents[1]/"benchmarks"/"new_bank_v2"


def load(name):
    return json.loads((ROOT/name).read_text(encoding="utf-8"))


def digest(name):
    return hashlib.sha256((ROOT/name).read_bytes()).hexdigest()


def rows(name):
    with (ROOT/name).open(encoding="utf-8",newline="") as stream:
        return list(csv.DictReader(stream))


def test_document_spatial_wearing_matches_native_trials_and_scores():
    protocol=load("DOCUMENT_SPATIAL_WEARING_PROTOCOL.json")
    result=load("DOCUMENT_SPATIAL_WEARING_RESULTS.json")
    parent=load("F2A_DOCUMENT_WEARING_RESULTS.json")
    data=rows("DOCUMENT_SPATIAL_WEARING_PREDICTIONS.csv")
    assert result["protocol_sha256"]==digest("DOCUMENT_SPATIAL_WEARING_PROTOCOL.json")
    assert result["prediction_sha256"]==digest("DOCUMENT_SPATIAL_WEARING_PREDICTIONS.csv")
    assert result["parent_result_sha256"]==protocol["parent_result_sha256"]==digest("F2A_DOCUMENT_WEARING_RESULTS.json")
    assert result["parent_prediction_sha256"]==protocol["parent_prediction_sha256"]==digest("F2A_DOCUMENT_WEARING_PREDICTIONS.csv")
    assert result["feature_dimensions"]=={"F0v2":48,"F2c_document":36,"F3c_document":20}
    assert len(data)==result["prediction_rows"]==480
    for phase in ("validation","final"):
        expected={trial for key,part in parent["split_trial_ids"].items()
                  if key.startswith(phase+"_") for trial in part["target"]}
        for arm in protocol["arms"]:
            selected=[row for row in data if row["phase"]==phase and row["arm"]==arm]
            assert {row["trial_id"] for row in selected}==expected
            y=np.array([int(row["label"]) for row in selected])
            p=np.array([[float(row[f"p_{c}"]) for c in range(5)] for row in selected])
            assert np.isfinite(p).all() and np.all(p>=0)
            np.testing.assert_allclose(p.sum(1),1.,atol=1e-12)
            summary=result["scores"][phase][arm]["pooled"]
            assert f1_score(y,p.argmax(1),labels=range(5),average="macro")==pytest.approx(summary["macro_f1"])
            assert log_loss(y,p,labels=range(5))==pytest.approx(summary["log_loss"])


def test_document_f2c_transfer_matches_native_trials_and_scores():
    protocol=load("F2C_DOCUMENT_TRANSFER_PROTOCOL.json")
    result=load("F2C_DOCUMENT_TRANSFER_RESULTS.json")
    manus=load("F2_MANUS_CANDIDATES_RESULTS.json")
    grab=load("F2_GRAB_CANDIDATES_RESULTS.json")
    data=rows("F2C_DOCUMENT_TRANSFER_PREDICTIONS.csv")
    assert result["protocol_sha256"]==digest("F2C_DOCUMENT_TRANSFER_PROTOCOL.json")
    assert result["prediction_sha256"]==digest("F2C_DOCUMENT_TRANSFER_PREDICTIONS.csv")
    assert result["parent_result_sha256"]==protocol["parent_result_sha256"]==digest("F2A_DOCUMENT_TRANSFER_RESULTS.json")
    assert result["parent_prediction_sha256"]==protocol["parent_prediction_sha256"]==digest("F2A_DOCUMENT_TRANSFER_PREDICTIONS.csv")
    assert len(data)==result["prediction_rows"]==328
    for axis,parent,classes in (("manus_session",manus,[0,1,2,3,4,5]),
                                ("grab_unseen_user",grab,[4,15,16,17])):
        for phase in ("validation","final"):
            selected=[row for row in data if row["axis"]==axis and row["phase"]==phase]
            expected=(parent["split_trial_ids"][phase]["target_trials"] if axis=="manus_session"
                      else parent[f"{phase}_trial_ids"])
            assert {row["trial_id"] for row in selected}==set(expected)
            assert {row["class_order"] for row in selected}=={"|".join(map(str,classes))}
            y=np.array([int(row["label"]) for row in selected])
            p=np.array([[float(row[f"p_{i}"]) for i in range(len(classes))] for row in selected])
            assert np.isfinite(p).all() and np.all(p>=0)
            np.testing.assert_allclose(p.sum(1),1.,atol=1e-12)
            score=result["scores"][axis][phase]
            if axis=="manus_session":score=score["pooled"]
            assert f1_score(y,np.array(classes)[p.argmax(1)],labels=classes,average="macro")==pytest.approx(score["macro_f1"])
            assert log_loss(y,p,labels=classes)==pytest.approx(score["log_loss"])


def test_document_spatial_guard_uses_validation_and_bounds_ring_role():
    audit=load("DOCUMENT_SPATIAL_CROSS_AXIS_AUDIT.json")
    data=rows("DOCUMENT_SPATIAL_CROSS_AXIS_CELLS.csv")
    assert audit["cells_sha256"]==digest("DOCUMENT_SPATIAL_CROSS_AXIS_CELLS.csv")
    assert audit["wearing_protocol_sha256"]==digest("DOCUMENT_SPATIAL_WEARING_PROTOCOL.json")
    assert audit["transfer_protocol_sha256"]==digest("F2C_DOCUMENT_TRANSFER_PROTOCOL.json")
    for name,expected in audit["source_results_sha256"].items():
        assert expected==digest(name)
    assert len(data)==audit["cells"]==8
    for row in data:
        assert float(row["delta_macro_f1"])==pytest.approx(
            float(row["candidate_macro_f1"])-float(row["f0_macro_f1"]),abs=1e-12)
        assert float(row["delta_log_loss"])==pytest.approx(
            float(row["candidate_log_loss"])-float(row["f0_log_loss"]),abs=1e-12)
    f2c=[row for row in data if row["phase"]=="validation" and row["arm"]=="F0v2+F2c_document"]
    assert {row["axis"] for row in f2c}=={"wearing_shift","manus_session","grab_unseen_user"}
    violations=[f"{row['axis']}:{metric}" for row in f2c
                for metric,failed in (("macro_f1",float(row["delta_macro_f1"])<-1e-12),
                                      ("log_loss",float(row["delta_log_loss"])>1e-12)) if failed]
    assert violations==audit["f2c_validation_violations"]
    assert audit["f2c_universal_default_eligible"] is False
    f3c=[row for row in data if row["arm"]=="F0v2+F3c_document"]
    assert {row["axis"] for row in f3c}=={"wearing_shift"}
    assert audit["three_axis_default"]=="F0v2"
