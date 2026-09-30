"""Read back the native document-formula F2a wearing experiment."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, log_loss


ROOT = Path(__file__).resolve().parents[1] / "benchmarks" / "new_bank_v2"


def test_document_f2a_wearing_rows_match_frozen_trials_and_scores():
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    protocol_path=ROOT/"F2A_DOCUMENT_WEARING_PROTOCOL.json"
    protocol=json.loads(protocol_path.read_text(encoding="utf-8"))
    result=json.loads((ROOT/"F2A_DOCUMENT_WEARING_RESULTS.json").read_text(encoding="utf-8"))
    parent=json.loads((ROOT/"F2_AC_WEARING_RESULTS.json").read_text(encoding="utf-8"))
    rows_path=ROOT/"F2A_DOCUMENT_WEARING_PREDICTIONS.csv"
    assert result["protocol_sha256"]==digest(protocol_path)
    assert result["parent_result_sha256"]==protocol["parent_result_sha256"]==digest(ROOT/"F2_AC_WEARING_RESULTS.json")
    assert result["parent_prediction_sha256"]==protocol["parent_prediction_sha256"]==digest(ROOT/"F2_AC_WEARING_PREDICTIONS.csv")
    assert result["prediction_sha256"]==digest(rows_path)
    assert result["feature_dimensions"]=={"F0v2":48,"F2a_uncentered":36}
    with rows_path.open(newline="",encoding="utf-8") as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==result["prediction_rows"]==240
    assert len({row["trial_id"] for row in rows})==240
    for phase in ("validation","final"):
        selected=[row for row in rows if row["phase"]==phase]
        expected={trial for subject in protocol["subjects"][phase]
                  for trial in parent["split_trial_ids"][f"{phase}_{subject}"]["target"]}
        assert {row["trial_id"] for row in selected}==expected
        y=np.asarray([int(row["label"]) for row in selected])
        p=np.asarray([[float(row[f"p_{c}"]) for c in range(5)] for row in selected])
        assert np.isfinite(p).all() and np.all(p>=0)
        np.testing.assert_allclose(p.sum(1),1.,atol=1e-12)
        score=result["scores"][phase]["pooled"]
        np.testing.assert_allclose(f1_score(y,p.argmax(1),labels=range(5),average="macro"),
                                   score["macro_f1"],atol=1e-12)
        np.testing.assert_allclose(log_loss(y,p,labels=range(5)),score["log_loss"],atol=1e-12)
    centered=parent["scores"]["validation"]["F0v2+F2a_trace"]["pooled"]
    uncentered=result["scores"]["validation"]["pooled"]
    assert uncentered["macro_f1"]<centered["macro_f1"]
    assert uncentered["log_loss"]>centered["log_loss"]
