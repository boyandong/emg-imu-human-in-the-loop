"""Frozen validation-only eligibility of historical uncentered F2c/F3c alternatives."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from benchmarks.new_bank_v2.roam_posture_run import sha256


ROOT=Path(__file__).resolve().parent
CELLS=ROOT/"DOCUMENT_SPATIAL_CROSS_AXIS_CELLS.csv"
AUDIT=ROOT/"DOCUMENT_SPATIAL_CROSS_AXIS_AUDIT.json"


def decide() -> dict:
    names=("DOCUMENT_SPATIAL_WEARING_RESULTS.json","F2C_DOCUMENT_TRANSFER_RESULTS.json",
           "F2B_WEARING_RESULTS.json","F2_MANUS_CANDIDATES_RESULTS.json",
           "F2_GRAB_CANDIDATES_RESULTS.json")
    sources={name:json.loads((ROOT/name).read_text(encoding="utf-8")) for name in names}
    wearing,transfer,wear_parent,manus_parent,grab_parent=(sources[name] for name in names)
    if (wearing["parent_result_sha256"]!=sha256(ROOT/"F2A_DOCUMENT_WEARING_RESULTS.json") or
            transfer["parent_result_sha256"]!=sha256(ROOT/"F2A_DOCUMENT_TRANSFER_RESULTS.json")):
        raise AssertionError("Frozen document spatial parents changed")
    rows=[]
    for phase in ("validation","final"):
        available=(
            ("wearing_shift","F0v2+F2c_document",
             wear_parent["scores"][phase]["F0v2"]["pooled"],
             wearing["scores"][phase]["F0v2+F2c_document"]["pooled"]),
            ("manus_session","F0v2+F2c_document",
             manus_parent["scores"][phase]["F0v2"]["pooled"],
             transfer["scores"]["manus_session"][phase]["pooled"]),
            ("grab_unseen_user","F0v2+F2c_document",
             grab_parent["scores"]["F0v2"][phase],
             transfer["scores"]["grab_unseen_user"][phase]),
            ("wearing_shift","F0v2+F3c_document",
             wear_parent["scores"][phase]["F0v2"]["pooled"],
             wearing["scores"][phase]["F0v2+F3c_document"]["pooled"]),
        )
        for axis,arm,baseline,candidate in available:
            if baseline["trials"]!=candidate["trials"]:
                raise AssertionError("Matched native trial count changed")
            rows.append({"phase":phase,"axis":axis,"arm":arm,"trials":baseline["trials"],
                         "f0_macro_f1":baseline["macro_f1"],
                         "candidate_macro_f1":candidate["macro_f1"],
                         "delta_macro_f1":candidate["macro_f1"]-baseline["macro_f1"],
                         "f0_log_loss":baseline["log_loss"],
                         "candidate_log_loss":candidate["log_loss"],
                         "delta_log_loss":candidate["log_loss"]-baseline["log_loss"]})
    with CELLS.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n")
        writer.writeheader();writer.writerows(rows)
    f2c=[row for row in rows if row["phase"]=="validation" and row["arm"]=="F0v2+F2c_document"]
    f2c_violations=[f"{row['axis']}:{metric}" for row in f2c
                    for metric,failed in (("macro_f1",row["delta_macro_f1"]<-1e-12),
                                          ("log_loss",row["delta_log_loss"]>1e-12)) if failed]
    f3c=next(row for row in rows if row["phase"]=="validation" and row["arm"]=="F0v2+F3c_document")
    result={"status":"document_spatial_public_eligibility",
            "wearing_protocol_sha256":sha256(ROOT/"DOCUMENT_SPATIAL_WEARING_PROTOCOL.json"),
            "transfer_protocol_sha256":sha256(ROOT/"F2C_DOCUMENT_TRANSFER_PROTOCOL.json"),
            "existing_guard_protocol_sha256":sha256(ROOT/"V1_CROSS_AXIS_DECISION_PROTOCOL.json"),
            "source_results_sha256":{name:sha256(ROOT/name) for name in names},
            "cells_sha256":sha256(CELLS),"cells":len(rows),
            "f2c_validation_violations":f2c_violations,
            "f2c_universal_default_eligible":not f2c_violations,
            "f3c_wearing_validation_f1_gain":f3c["delta_macro_f1"],
            "f3c_wearing_validation_loss_change":f3c["delta_log_loss"],
            "f3c_role":"wearing-only exploratory candidate under assumed saved-column circular order; dataset column-to-electrode adjacency unverified",
            "three_axis_default":"F0v2" if f2c_violations else "F0v2+F2c_document",
            "boundary":"Public datasets were previously inspected. F2c uses the pre-existing three-axis validation guard. The Myo armband is physically circular, but this dataset does not attest that saved columns follow physical adjacency; F3c is therefore an index-order exploratory screen, not verified anatomical ring evidence. Finals are descriptive and own-device validation is absent."}
    AUDIT.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"cells":len(rows),"three_axis_default":result["three_axis_default"],
                      "f3c_wearing_validation_f1_gain":f3c["delta_macro_f1"]}))
    return result


if __name__=="__main__":decide()
