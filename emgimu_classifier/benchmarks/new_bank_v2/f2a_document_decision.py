"""Apply the existing validation-only default guard to document F2a."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from benchmarks.new_bank_v2.roam_posture_run import sha256


ROOT=Path(__file__).resolve().parent
CELLS=ROOT/"F2A_DOCUMENT_CROSS_AXIS_CELLS.csv"
AUDIT=ROOT/"F2A_DOCUMENT_CROSS_AXIS_AUDIT.json"


def decide() -> dict:
    names=("F2A_DOCUMENT_WEARING_RESULTS.json","F2A_DOCUMENT_TRANSFER_RESULTS.json",
           "F2B_WEARING_RESULTS.json","F2_MANUS_CANDIDATES_RESULTS.json",
           "F2_GRAB_CANDIDATES_RESULTS.json")
    loaded={name:json.loads((ROOT/name).read_text(encoding="utf-8")) for name in names}
    wear,transfer,wear_parent,manus_parent,grab_parent=(loaded[name] for name in names)
    if (wear["parent_result_sha256"]!=sha256(ROOT/"F2_AC_WEARING_RESULTS.json") or
            transfer["manus_parent_result_sha256"]!=sha256(ROOT/"F2_MANUS_CANDIDATES_RESULTS.json") or
            transfer["grab_parent_result_sha256"]!=sha256(ROOT/"F2_GRAB_CANDIDATES_RESULTS.json")):
        raise AssertionError("Document F2a matched parents changed")
    rows=[]
    for phase in ("validation","final"):
        measures={
            "wearing_shift":(wear_parent["scores"][phase]["F0v2"]["pooled"],
                             wear["scores"][phase]["pooled"]),
            "manus_session":(manus_parent["scores"][phase]["F0v2"]["pooled"],
                             transfer["scores"]["manus_session"][phase]["pooled"]),
            "grab_unseen_user":(grab_parent["scores"]["F0v2"][phase],
                                transfer["scores"]["grab_unseen_user"][phase]),
        }
        for axis,(baseline,candidate) in measures.items():
            if baseline["trials"]!=candidate["trials"]:
                raise AssertionError(f"Unmatched {axis} trial counts")
            rows.append({"phase":phase,"axis":axis,"trials":baseline["trials"],
                         "f0_macro_f1":baseline["macro_f1"],
                         "candidate_macro_f1":candidate["macro_f1"],
                         "delta_macro_f1":candidate["macro_f1"]-baseline["macro_f1"],
                         "f0_log_loss":baseline["log_loss"],
                         "candidate_log_loss":candidate["log_loss"],
                         "delta_log_loss":candidate["log_loss"]-baseline["log_loss"]})
    with CELLS.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n")
        writer.writeheader();writer.writerows(rows)
    validation=rows[:3]
    violations=[f"{row['axis']}:{metric}" for row in validation
                for metric,failed in (("macro_f1",row["delta_macro_f1"]<-1e-12),
                                      ("log_loss",row["delta_log_loss"]>1e-12)) if failed]
    eligible=(not violations and any(row["delta_macro_f1"]>1e-12 or row["delta_log_loss"]<-1e-12
                                     for row in validation))
    result={"status":"document_f2a_validation_only_public_guard",
            "wearing_protocol_sha256":sha256(ROOT/"F2A_DOCUMENT_WEARING_PROTOCOL.json"),
            "transfer_protocol_sha256":sha256(ROOT/"F2A_DOCUMENT_TRANSFER_PROTOCOL.json"),
            "existing_guard_protocol_sha256":sha256(ROOT/"V1_CROSS_AXIS_DECISION_PROTOCOL.json"),
            "source_results_sha256":{name:sha256(ROOT/name) for name in names},
            "cells_sha256":sha256(CELLS),"cells":len(rows),
            "validation_violations":violations,
            "candidate_eligible_for_public_default":eligible,
            "three_axis_default":"F0v2+F2a_uncentered" if eligible else "F0v2",
            "boundary":"Existing validation guard applied to an added formula candidate on previously inspected public datasets; descriptive finals do not enter selection. No own-device or independent-cohort claim."}
    AUDIT.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"cells":len(rows),"three_axis_default":result["three_axis_default"]}))
    return result


if __name__=="__main__":decide()
