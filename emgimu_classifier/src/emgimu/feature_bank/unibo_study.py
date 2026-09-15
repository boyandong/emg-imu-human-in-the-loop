from __future__ import annotations

import argparse
import csv
import json
import pickle
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
    train_features={};target_features={};dimensions={};family_states={};diagnostics=[]
    for index,(name,factory) in enumerate(active.items(),1):
        print(f"[2/4] feature {index}/{len(active)} {name}",flush=True)
        family=factory();train_features[name]=family.fit_transform(train_batch,train.labels);target_features[name]=family.transform(target_batch);dimensions[name]=train_features[name].shape[1]
        family_states[name]=family
        diagnostic_scaler=StandardScaler().fit(train_features[name])
        source_z=diagnostic_scaler.transform(train_features[name]);target_z=diagnostic_scaler.transform(target_features[name])
        for subject in np.unique(target.subject_id):
            displacement=[];posture_distance=[];separation=[]
            centers={}
            for posture in range(1,5):
                for label in HAND_CLASSES:
                    source_mask=(train.subject_id==subject)&(train.posture==posture)&(train.labels==label)
                    target_mask=(target.subject_id==subject)&(target.posture==posture)&(target.labels==label)
                    if np.any(source_mask) and np.any(target_mask):
                        center=target_z[target_mask].mean(0);centers[(posture,label)]=center
                        displacement.append(np.linalg.norm(center-source_z[source_mask].mean(0)))
            for label in HAND_CLASSES:
                cs=[centers[(p,label)] for p in range(1,5) if (p,label) in centers]
                posture_distance.extend(np.linalg.norm(a-b) for i,a in enumerate(cs) for b in cs[i+1:])
            for posture in range(1,5):
                cs=[centers[(posture,h)] for h in HAND_CLASSES if (posture,h) in centers]
                separation.extend(np.linalg.norm(a-b) for i,a in enumerate(cs) for b in cs[i+1:])
            dg=float(np.mean(separation))
            for factor,values in (('day',displacement),('posture',posture_distance)):
                dn=float(np.mean(values))
                diagnostics.append({'dataset':'unibo_inail','phase':phase,'subject':subject,'family':name,'factor':factor,
                    'D_nuisance':dn,'D_gesture':dg,'J':dg/(dn+1e-12),'feature_dimension':dimensions[name],
                    'definition':'source-standardized window-pooled class/posture centroids; day matched by posture and class'})
    specs={"B_F0":("F0",),**{f"B_plus_{name}":("F0",name) for name in active if name!="F0"}}
    train_weights=hierarchical_segment_weights(_as_feature_windows(train,np.empty((len(train),0),dtype=np.float32)))
    target_weights=hierarchical_segment_weights(_as_feature_windows(target,np.empty((len(target),0),dtype=np.float32)))
    rows=[];predictions={};classifier_states={}
    target_days=np.asarray([int(str(value)[1:]) for value in target.session_id])
    for index,(model_name,members) in enumerate(specs.items(),1):
        print(f"[3/4] classifier {index}/{len(specs)} {model_name}",flush=True)
        a=np.concatenate([train_features[x] for x in members],axis=1);b=np.concatenate([target_features[x] for x in members],axis=1)
        scaler=StandardScaler().fit(a,sample_weight=train_weights);model=LogisticRegression(C=1.0,class_weight="balanced",max_iter=1000,random_state=SEED)
        probability=model.fit(scaler.transform(a),train.labels,sample_weight=train_weights).predict_proba(scaler.transform(b))
        predictions[model_name]=probability;classifier_states[model_name]=(scaler,model,members)
        cells=[("ALL","ALL",np.ones(len(target),dtype=bool))]
        cells += [("ALL",f"posture_{p}",target.posture==p) for p in range(1,5)]
        cells += [(subject,"ALL",target.subject_id==subject) for subject in sorted(np.unique(target.subject_id))]
        cells += [("ALL",f"day_{day}",target_days==day) for day in sorted(np.unique(target_days))]
        for subject,condition,selected in cells:
            rows.append({"phase":phase,"dataset":"unibo_inail","subject":subject,"condition":condition,"feature_family":"+".join(members),
                "calibration_budget":0,**_metrics(target.labels[selected],probability[selected],target_weights[selected]),
                "feature_dimension":sum(dimensions[x] for x in members)})
    print("[4/4] writing UniBo Feature Bank evidence",flush=True);output.mkdir(parents=True,exist_ok=False)
    with (output/'fitted_states.pkl').open('wb') as handle:pickle.dump((family_states,classifier_states),handle)
    np.savez_compressed(output/'heldout_predictions.npz',**predictions,labels=target.labels,
                        trials=target.trial_id,subjects=target.subject_id,days=target.session_id,posture=target.posture)
    (output/'split_trial_ids.json').write_text(json.dumps({'train':sorted(set(train.trial_id.tolist())),
        'validation' if phase=='validation' else 'test':sorted(set(target.trial_id.tolist()))},indent=2),encoding='utf-8')
    with (output/'cross_day_posture_diagnostics.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(diagnostics[0]));writer.writeheader();writer.writerows(diagnostics)
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
