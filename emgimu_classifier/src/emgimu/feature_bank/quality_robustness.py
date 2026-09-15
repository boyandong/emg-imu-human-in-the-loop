from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from emgimu.datasets.electrode_shift import ShiftWindows, load_electrode_shift_windows
from emgimu.feature_bank import FeatureBatch

from .electrode_shift_study import _metrics
from .families import LocalDetailFamily, QualityFamily, RingGeometryFamily
from .screening import SEED


def _corrupt(emg: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    output = np.asarray(emg, dtype=float).copy(); selected = np.arange(len(output)) % 4 == 0
    if kind == "clean": return output
    if kind == "channel_dropout": output[selected, :, 0] = 0
    elif kind == "clipping": output[selected] = np.clip(output[selected], -32, 32)
    elif kind == "line_noise":
        tone = np.sin(2*np.pi*50*np.arange(output.shape[1])/200.0)[None,:,None]
        amplitude = np.std(output[selected], axis=1, keepdims=True)*2.0; output[selected] += amplitude*tone
    elif kind == "motion_burst":
        burst = np.sin(np.linspace(0,np.pi,output.shape[1]))[None,:,None]
        amplitude = np.std(output[selected],axis=1,keepdims=True)*rng.uniform(4,8,size=(selected.sum(),1,1)); output[selected] += amplitude*burst
    else: raise ValueError(kind)
    return output


def _aggregate(features: np.ndarray, data: ShiftWindows, quality: np.ndarray | None = None) -> tuple[np.ndarray,np.ndarray]:
    rows=[];labels=[]
    for trial in np.unique(data.trials):
        selected=data.trials==trial; weights=np.ones(selected.sum()) if quality is None else np.maximum(quality[selected],1e-3)
        rows.append(np.average(features[selected],axis=0,weights=weights));labels.append(np.unique(data.labels[selected]).item())
    return np.stack(rows),np.asarray(labels)


def run(archive: Path, output: Path) -> None:
    subjects=(15,16,17);corruptions=("clean","channel_dropout","clipping","line_noise","motion_burst");rows=[];rng=np.random.default_rng(SEED)
    print("[1/4] loading quality robustness subjects",flush=True)
    for subject_index,subject in enumerate(subjects,1):
        train=load_electrode_shift_windows(archive,subjects=(subject,),domains=("training",));target=load_electrode_shift_windows(archive,subjects=(subject,),domains=("trial_1","trial_2","trial_3","trial_4"))
        f0=LocalDetailFamily();f3=RingGeometryFamily();qf=QualityFamily(adc_min=-128,adc_max=127)
        train_feature=np.concatenate((f0.fit_transform(train.batch,train.labels),f3.fit_transform(train.batch,train.labels)),axis=1);qf.fit(train.batch,train.labels)
        train_trial,train_y=_aggregate(train_feature,train);scaler=StandardScaler().fit(train_trial);model=LogisticRegression(C=1,class_weight="balanced",max_iter=1000,random_state=SEED).fit(scaler.transform(train_trial),train_y)
        for corruption_index,kind in enumerate(corruptions,1):
            print(f"[2/4] subject {subject_index}/{len(subjects)} corruption {corruption_index}/{len(corruptions)} {kind}",flush=True)
            corrupted=FeatureBatch(_corrupt(target.batch.emg,kind,rng),200.0)
            features=np.concatenate((f0.transform(corrupted),f3.transform(corrupted)),axis=1);quality_features=qf.transform(corrupted)
            quality=quality_features[:,qf.feature_names.index("F9.mean_quality")]
            for mode,score in (("uniform",None),("F9_quality_weighted",quality)):
                trial_x,y=_aggregate(features,target,score);probability=model.predict_proba(scaler.transform(trial_x))
                rows.append({"dataset":"libemg_electrode_shift","subject":subject,"corruption":kind,"aggregation":mode,
                    "synthetic":kind!="clean","mean_quality":float(np.mean(quality)),**_metrics(y,probability)})
    print("[3/4] aggregating robustness cells",flush=True)
    for kind in corruptions:
        for mode in ("uniform","F9_quality_weighted"):
            selected=[row for row in rows if row["corruption"]==kind and row["aggregation"]==mode]
            rows.append({"dataset":"libemg_electrode_shift","subject":"ALL","corruption":kind,"aggregation":mode,"synthetic":kind!="clean",
                **{key:float(np.mean([float(row[key]) for row in selected])) for key in ("mean_quality","macro_f1","accuracy","log_loss","brier","ece")},
                "per_class_f1_json":"per-subject macro aggregation"})
    print("[4/4] writing synthetic quality evidence",flush=True);output.mkdir(parents=True,exist_ok=False)
    with (output/"quality_robustness.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/"run_manifest.json").write_text(json.dumps({"seed":SEED,"subjects":list(subjects),"corrupt_fraction_of_windows":0.25,
        "corruptions":list(corruptions),"quality_family":"F9","feature_model":"F0+F3_Ring","synthetic_claim_only":True},indent=2),encoding="utf-8")
    print(json.dumps({"status":"ok","rows":len(rows),"output":str(output)}),flush=True)


def main() -> None:
    parser=argparse.ArgumentParser(description="F9 synthetic corruption robustness study");parser.add_argument("archive",type=Path);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();run(args.archive,args.output)


if __name__ == "__main__":main()
