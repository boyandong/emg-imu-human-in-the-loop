"""Matched wearing screen for document-consistent uncentered F2c and F3c."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmarks.new_bank_v2 import wearing_v1_extension_run as wearing
from benchmarks.new_bank_v2.roam_posture_run import sha256
from emgimu.datasets.electrode_shift import PATH_RE
from emgimu.feature_bank.new_bank_v2 import (
    DocumentRingRelativeCovarianceV2, DocumentSpdTangentV2, RestNoiseDetailV2,
)


ROOT=Path(__file__).resolve().parent
PROTOCOL_PATH=ROOT/"DOCUMENT_SPATIAL_WEARING_PROTOCOL.json"
PREDICTIONS=ROOT/"DOCUMENT_SPATIAL_WEARING_PREDICTIONS.csv"
RESULT=ROOT/"DOCUMENT_SPATIAL_WEARING_RESULTS.json"


def evaluate() -> dict:
    protocol=json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if (sha256(ROOT/"F2A_DOCUMENT_WEARING_RESULTS.json")!=protocol["parent_result_sha256"] or
            sha256(ROOT/"F2A_DOCUMENT_WEARING_PREDICTIONS.csv")!=protocol["parent_prediction_sha256"] or
            sha256(wearing.ARCHIVE)!=protocol["archive_sha256"]):
        raise AssertionError("Frozen document-F2a wearing parent changed")
    wearing.check_protocol()
    parent=json.loads((ROOT/"F2A_DOCUMENT_WEARING_RESULTS.json").read_text(encoding="utf-8"))
    root_protocol=json.loads((ROOT/"F2A_DOCUMENT_WEARING_PROTOCOL.json").read_text(encoding="utf-8"))
    classes=np.asarray(root_protocol["classes"])
    rows=[];partitions={};dimensions={}
    for phase in ("validation","final"):
        for subject in root_protocol["subjects"][phase]:
            source=wearing.load(wearing.ARCHIVE,subject,(root_protocol["source_domain"],))
            target=wearing.load(wearing.ARCHIVE,subject,tuple(root_protocol["target_domains"]))
            if source.batch.channels!=8 or source.batch.sample_rate_hz!=200 or source.batch.emg.shape[1]!=40:
                raise AssertionError("Native wearing sensor/window contract changed")
            families={"F0v2":RestNoiseDetailV2(rest_label=2).fit(source.batch,source.labels),
                      "F2c_document":DocumentSpdTangentV2(shrinkage=.05).fit(source.batch),
                      "F3c_document":DocumentRingRelativeCovarianceV2(ring_topology=True,
                                                                       shrinkage=.05).fit(source.batch)}
            xs,ys,source_users,source_trials=wearing.vectors(source,families)
            xt,yt,target_users,target_trials=wearing.vectors(target,families)
            key=f"{phase}_{subject}";frozen=parent["split_trial_ids"][key]
            if (set(source_trials)!=set(frozen["source"]) or set(target_trials)!=set(frozen["target"]) or
                    set(source_trials)&set(target_trials) or set(source_users)!={subject} or
                    set(target_users)!={subject}):
                raise AssertionError("Native wearing partition changed")
            partitions[key]={"source":source_trials.tolist(),"target":target_trials.tolist()}
            for name,values in xs.items():
                if name in dimensions and dimensions[name]!=values.shape[1]:
                    raise AssertionError("Feature dimension changed across subjects")
                dimensions[name]=int(values.shape[1])
            for arm in protocol["arms"]:
                names=arm.split("+")
                model=make_pipeline(StandardScaler(),LogisticRegression(
                    C=1.,class_weight="balanced",max_iter=2000,random_state=20260924))
                model.fit(np.concatenate([xs[name] for name in names],axis=1),ys)
                np.testing.assert_array_equal(model[-1].classes_,classes)
                if np.max(model[-1].n_iter_)>=2000:
                    raise RuntimeError(f"Document spatial classifier did not converge: {key}/{arm}")
                probabilities=model.predict_proba(np.concatenate([xt[name] for name in names],axis=1))
                for trial,label,p in zip(target_trials,yt,probabilities):
                    identity=PATH_RE.fullmatch(str(trial))
                    if identity is None or int(identity["subject"])!=subject or int(identity["label"])!=label:
                        raise AssertionError("Native wearing trial metadata changed")
                    rows.append({"phase":phase,"subject":subject,"domain":identity["domain"],
                                 "arm":arm,"trial_id":str(trial),"label":int(label),
                                 **{f"p_{c}":float(value) for c,value in zip(classes,p)}})
            print(f"Document spatial wearing {key}: {len(target_trials)} held-out trials",flush=True)
    scores={phase:{arm:wearing._score(rows,phase,arm) for arm in protocol["arms"]}
            for phase in ("validation","final")}
    with PREDICTIONS.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n")
        writer.writeheader();writer.writerows(rows)
    result={"status":"document_uncentered_spatial_native_wearing",
            "protocol_sha256":sha256(PROTOCOL_PATH),
            "parent_result_sha256":protocol["parent_result_sha256"],
            "parent_prediction_sha256":protocol["parent_prediction_sha256"],
            "archive_sha256":protocol["archive_sha256"],
            "prediction_sha256":sha256(PREDICTIONS),"prediction_rows":len(rows),
            "feature_dimensions":dimensions,"split_trial_ids":partitions,
            "scores":scores,"scope":protocol["boundary"]}
    RESULT.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    for arm in protocol["arms"]:
        print(f"{arm}: validation F1={scores['validation'][arm]['pooled']['macro_f1']:.4f}; "
              f"final F1={scores['final'][arm]['pooled']['macro_f1']:.4f}",flush=True)
    return result


if __name__=="__main__":evaluate()
