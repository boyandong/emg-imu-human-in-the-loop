from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.preprocessing import StandardScaler

from emgimu.datasets.unibo_baseline import HAND_CLASSES, HAND_NAMES, hierarchical_segment_weights
from emgimu.datasets.unibo_physiology import _as_feature_windows, _subset_raw, chronological_fold, load_chronological_raw_windows
from emgimu.feature_bank import FeatureBatch

from .families import BodyContextFamily
from .screening import FAMILY_FACTORIES, SEED, expected_calibration_error


FACTORIES = {name: factory for name, factory in FAMILY_FACTORIES.items() if name != "F3_Ring"}
FACTORIES["F6_Posture"] = BodyContextFamily


def _metrics(y: np.ndarray, probability: np.ndarray, weights: np.ndarray) -> dict:
    prediction = probability.argmax(1); one_hot = np.eye(len(HAND_CLASSES))[y]
    per_class = f1_score(y, prediction, labels=HAND_CLASSES, average=None, sample_weight=weights, zero_division=0)
    return {"macro_f1": float(f1_score(y, prediction, labels=HAND_CLASSES, average="macro", sample_weight=weights, zero_division=0)),
        "accuracy": float(accuracy_score(y, prediction, sample_weight=weights)),
        "log_loss": float(log_loss(y, probability, labels=HAND_CLASSES, sample_weight=weights)),
        "brier": float(np.average(np.mean((probability-one_hot)**2,axis=1),weights=weights)),
        "ece": expected_calibration_error(y, probability, weights),
        "per_class_f1_json": json.dumps({HAND_NAMES[int(label)]:float(value) for label,value in zip(HAND_CLASSES,per_class)},separators=(",",":"))}


def run(root: Path, output: Path, phase: str, frozen_added_family: str | None) -> None:
    print(f"[1/4] loading UniBo chronological {phase} split", flush=True)
    if phase == "validation":
        raw = load_chronological_raw_windows(root, range(1,7)); train,target = chronological_fold(raw,(1,2,3,4,5),6,6)
        active = FACTORIES
    else:
        if frozen_added_family not in FACTORIES or frozen_added_family == "F0":
            raise ValueError("final phase requires the frozen added family selected on Day 6")
        raw = load_chronological_raw_windows(root,(1,2,3,4,5,7,8)); train,unused = chronological_fold(raw,(1,2,3,4,5),7,6)
        days=np.asarray([int(str(value)[1:]) for value in raw.session_id]); target=_subset_raw(raw,np.flatnonzero(np.isin(days,(7,8))))
        active={"F0":FACTORIES["F0"],frozen_added_family:FACTORIES[frozen_added_family]}
    train_batch=FeatureBatch(train.emg,200.0,posture=train.posture.astype(str));target_batch=FeatureBatch(target.emg,200.0,posture=target.posture.astype(str))
    train_features={};target_features={};dimensions={}
    for index,(name,factory) in enumerate(active.items(),1):
        print(f"[2/4] feature {index}/{len(active)} {name}",flush=True)
        family=factory();train_features[name]=family.fit_transform(train_batch,train.labels);target_features[name]=family.transform(target_batch);dimensions[name]=train_features[name].shape[1]
    specs={"B_F0":("F0",),**{f"B_plus_{name}":("F0",name) for name in active if name!="F0"}}
    train_weights=hierarchical_segment_weights(_as_feature_windows(train,np.empty((len(train),0),dtype=np.float32)))
    target_weights=hierarchical_segment_weights(_as_feature_windows(target,np.empty((len(target),0),dtype=np.float32)))
    rows=[]
    target_days=np.asarray([int(str(value)[1:]) for value in target.session_id])
    for index,(model_name,members) in enumerate(specs.items(),1):
        print(f"[3/4] classifier {index}/{len(specs)} {model_name}",flush=True)
        a=np.concatenate([train_features[x] for x in members],axis=1);b=np.concatenate([target_features[x] for x in members],axis=1)
        scaler=StandardScaler().fit(a,sample_weight=train_weights);model=LogisticRegression(C=1.0,class_weight="balanced",max_iter=1000,random_state=SEED)
        probability=model.fit(scaler.transform(a),train.labels,sample_weight=train_weights).predict_proba(scaler.transform(b))
        cells=[("ALL","ALL",np.ones(len(target),dtype=bool))]
        cells += [("ALL",f"posture_{p}",target.posture==p) for p in range(1,5)]
        cells += [(subject,"ALL",target.subject_id==subject) for subject in sorted(np.unique(target.subject_id))]
        cells += [("ALL",f"day_{day}",target_days==day) for day in sorted(np.unique(target_days))]
        for subject,condition,selected in cells:
            rows.append({"phase":phase,"dataset":"unibo_inail","subject":subject,"condition":condition,"feature_family":"+".join(members),
                "calibration_budget":0,**_metrics(target.labels[selected],probability[selected],target_weights[selected]),
                "feature_dimension":sum(dimensions[x] for x in members)})
    print("[4/4] writing UniBo Feature Bank evidence",flush=True);output.mkdir(parents=True,exist_ok=False)
    with (output/"feature_family_results.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/"run_manifest.json").write_text(json.dumps({"phase":phase,"seed":SEED,"train_days":[1,2,3,4,5],"target_days":[6] if phase=="validation" else [7,8],
        "families":list(active),"frozen_added_family":frozen_added_family,"ring_features_excluded":"UniBo has named muscles, not circular electrodes",
        "F6_interpretation":"oracle posture upper bound","dimensions":dimensions},indent=2),encoding="utf-8")
    print(json.dumps({"status":"ok","phase":phase,"train_windows":len(train),"target_windows":len(target),"output":str(output)}),flush=True)


def main() -> None:
    parser=argparse.ArgumentParser(description="UniBo Feature Bank chronological study");parser.add_argument("dataset",type=Path);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--phase",choices=("validation","final"),required=True);parser.add_argument("--frozen-added-family")
    args=parser.parse_args();run(args.dataset,args.output,args.phase,args.frozen_added_family)


if __name__ == "__main__":main()
