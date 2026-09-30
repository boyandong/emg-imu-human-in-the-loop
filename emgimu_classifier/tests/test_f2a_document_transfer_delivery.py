"""Read back document-exact F2a matched MANUS and GRAB transfer rows."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, log_loss


ROOT=Path(__file__).resolve().parents[1]/"benchmarks"/"new_bank_v2"


def test_document_f2a_transfer_rows_match_frozen_target_ids_and_scores():
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    protocol_path=ROOT/"F2A_DOCUMENT_TRANSFER_PROTOCOL.json"
    protocol=json.loads(protocol_path.read_text(encoding="utf-8"))
    result=json.loads((ROOT/"F2A_DOCUMENT_TRANSFER_RESULTS.json").read_text(encoding="utf-8"))
    manus=json.loads((ROOT/"F2_MANUS_CANDIDATES_RESULTS.json").read_text(encoding="utf-8"))
    grab=json.loads((ROOT/"F2_GRAB_CANDIDATES_RESULTS.json").read_text(encoding="utf-8"))
    rows_path=ROOT/"F2A_DOCUMENT_TRANSFER_PREDICTIONS.csv"
    assert result["protocol_sha256"]==digest(protocol_path)
    assert result["prediction_sha256"]==digest(rows_path)
    for name,key in (("F2_MANUS_CANDIDATES_RESULTS.json","manus_parent_result_sha256"),
                     ("F2_MANUS_CANDIDATES_PREDICTIONS.csv","manus_parent_prediction_sha256"),
                     ("F2_GRAB_CANDIDATES_RESULTS.json","grab_parent_result_sha256"),
                     ("F2_GRAB_CANDIDATES_PREDICTIONS.csv","grab_parent_prediction_sha256")):
        assert protocol[key]==digest(ROOT/name)
    with rows_path.open(newline="",encoding="utf-8") as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==result["prediction_rows"]==328
    assert len({row["trial_id"] for row in rows})==328
    for axis,source,labels in (("manus_session",manus,[0,1,2,3,4,5]),
                               ("grab_unseen_user",grab,[4,15,16,17])):
        for phase in ("validation","final"):
            selected=[row for row in rows if row["axis"]==axis and row["phase"]==phase]
            expected=(source["split_trial_ids"][phase]["target_trials"] if axis=="manus_session"
                      else source[f"{phase}_trial_ids"])
            assert {row["trial_id"] for row in selected}==set(expected)
            assert len(selected)==len(expected)==(108 if axis=="manus_session" else 56)
            assert {row["class_order"] for row in selected}=={"|".join(map(str,labels))}
            y=np.array([int(row["label"]) for row in selected])
            p=np.array([[float(row[f"p_{i}"]) for i in range(len(labels))] for row in selected])
            assert set(y)==set(labels) and np.isfinite(p).all() and np.all(p>=0)
            np.testing.assert_allclose(p.sum(1),1.,atol=1e-12)
            score=result["scores"][axis][phase]
            if axis=="manus_session":score=score["pooled"]
            np.testing.assert_allclose(f1_score(y,np.array(labels)[p.argmax(1)],labels=labels,average="macro"),
                                       score["macro_f1"],atol=1e-12)
            np.testing.assert_allclose(log_loss(y,p,labels=labels),score["log_loss"],atol=1e-12)
