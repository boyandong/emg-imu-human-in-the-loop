from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,log_loss
from sklearn.preprocessing import StandardScaler

from emgimu.datasets.emg_fmg import GESTURES,EmgFmgWindows,load_emg_fmg_windows
from .screening import FAMILY_FACTORIES,SEED,expected_calibration_error


def _trial_probabilities(probability:np.ndarray,data:EmgFmgWindows)->tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    trials=np.unique(data.trials);rows=[];labels=[];loads=[];positions=[]
    for trial in trials:
        selected=data.trials==trial;rows.append(probability[selected].mean(0));labels.append(np.unique(data.labels[selected]).item())
        loads.append(np.unique(data.loads[selected]).item());positions.append(np.unique(data.positions[selected]).item())
    return np.stack(rows),np.asarray(labels),np.asarray(loads),np.asarray(positions)


def _metrics(y:np.ndarray,p:np.ndarray)->dict:
    pred=p.argmax(1);one=np.eye(4)[y];w=np.ones(len(y));pc=f1_score(y,pred,labels=np.arange(4),average=None,zero_division=0)
    return {"macro_f1":float(f1_score(y,pred,labels=np.arange(4),average="macro",zero_division=0)),"accuracy":float(accuracy_score(y,pred)),
        "log_loss":float(log_loss(y,p,labels=np.arange(4))),"brier":float(np.mean((p-one)**2)),"ece":expected_calibration_error(y,p,w),
        "per_class_f1_json":json.dumps({name:float(value) for name,value in zip(GESTURES,pc)},separators=(",",":"))}


def run(archive:Path,output:Path,phase:str,load_family:str|None,position_family:str|None)->None:
    subjects=(1,2,3) if phase=="validation" else (4,5,6)
    if phase=="final" and (load_family not in FAMILY_FACTORIES or position_family not in FAMILY_FACTORIES):raise ValueError("final requires frozen families")
    print(f"[1/4] loading EMG-FMG {phase} subjects {subjects}",flush=True);collected=[]
    for si,subject in enumerate(subjects,1):
        all_data=load_emg_fmg_windows(archive,subjects=(subject,),loads=(0,250,500,750,1000),positions=range(1,9))
        scenarios=(("load_shift",all_data.loads==0,all_data.loads>0),("position_shift",all_data.positions==1,all_data.positions>1))
        for scenario,train_mask,target_mask in scenarios:
            train=all_data.take(np.flatnonzero(train_mask));target=all_data.take(np.flatnonzero(target_mask))
            frozen=load_family if scenario=="load_shift" else position_family
            active=FAMILY_FACTORIES if phase=="validation" else {"F0":FAMILY_FACTORIES["F0"],frozen:FAMILY_FACTORIES[frozen]}
            train_features={};target_features={};dimensions={}
            for fi,(name,factory) in enumerate(active.items(),1):
                print(f"[2/4] subject {si}/{len(subjects)} {scenario} feature {fi}/{len(active)} {name}",flush=True)
                family=factory();train_features[name]=family.fit_transform(train.batch,train.labels);target_features[name]=family.transform(target.batch);dimensions[name]=train_features[name].shape[1]
            specs={"B_F0":("F0",),**{f"B_plus_{name}":("F0",name) for name in active if name!="F0"}}
            for mi,(model_name,members) in enumerate(specs.items(),1):
                print(f"[3/4] subject {si}/{len(subjects)} {scenario} model {mi}/{len(specs)} {model_name}",flush=True)
                a=np.concatenate([train_features[x] for x in members],1);b=np.concatenate([target_features[x] for x in members],1)
                scaler=StandardScaler().fit(a,sample_weight=train.sample_weight);model=LogisticRegression(C=1,class_weight="balanced",max_iter=1000,random_state=SEED)
                p=model.fit(scaler.transform(a),train.labels,sample_weight=train.sample_weight).predict_proba(scaler.transform(b));p,y,loads,positions=_trial_probabilities(p,target)
                cells=[("ALL",np.ones(len(y),dtype=bool))]
                values=sorted(np.unique(loads if scenario=="load_shift" else positions))
                cells += [(f"load_{value}" if scenario=="load_shift" else f"position_{value}",(loads if scenario=="load_shift" else positions)==value) for value in values]
                for condition,selected in cells:
                    collected.append({"phase":phase,"dataset":"emg_fmg","scenario":scenario,"subject":subject,"condition":condition,
                        "feature_family":"+".join(members),"calibration_budget":0,**_metrics(y[selected],p[selected]),"feature_dimension":sum(dimensions[x] for x in members)})
    print("[4/4] aggregating and writing load-position evidence",flush=True)
    aggregates=[]
    keys=sorted({(row["scenario"],row["condition"],row["feature_family"]) for row in collected})
    for scenario,condition,family in keys:
        selected=[row for row in collected if (row["scenario"],row["condition"],row["feature_family"])==(scenario,condition,family)]
        aggregates.append({"phase":phase,"dataset":"emg_fmg","scenario":scenario,"subject":"ALL","condition":condition,"feature_family":family,"calibration_budget":0,
            **{key:float(np.mean([float(row[key]) for row in selected])) for key in ("macro_f1","accuracy","log_loss","brier","ece")},
            "per_class_f1_json":"per-subject macro aggregation","feature_dimension":selected[0]["feature_dimension"]})
    rows=collected+aggregates;output.mkdir(parents=True,exist_ok=False)
    with (output/"feature_family_results.csv").open("w",newline="",encoding="utf-8") as handle:writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/"run_manifest.json").write_text(json.dumps({"phase":phase,"seed":SEED,"subjects":list(subjects),"evaluation_unit":"18-second trial",
        "EMG_columns":"9-16 only","load_shift":{"train_load_g":0,"target_load_g":[250,500,750,1000]},"position_shift":{"train_position":1,"target_positions":[2,3,4,5,6,7,8]},
        "frozen_families":{"load_shift":load_family,"position_shift":position_family}},indent=2),encoding="utf-8")
    print(json.dumps({"status":"ok","phase":phase,"rows":len(rows),"output":str(output)}),flush=True)


def main()->None:
    parser=argparse.ArgumentParser(description="EMG-FMG load and limb-position Feature Bank study");parser.add_argument("archive",type=Path);parser.add_argument("--output",type=Path,required=True);parser.add_argument("--phase",choices=("validation","final"),required=True)
    parser.add_argument("--load-family");parser.add_argument("--position-family");args=parser.parse_args();run(args.archive,args.output,args.phase,args.load_family,args.position_family)


if __name__=="__main__":main()
