from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import re
from typing import Iterable
import zipfile

import numpy as np

from emgimu.feature_bank import FeatureBatch


PATH_RE=re.compile(r"Data/Par (?P<subject>\d+)/(?P<gesture>Key|Pinch|Power|Tripod)/(?P<load>0|250|500|750|1000)/[^/]+ (?P=load) (?P<position>[1-8])\.csv$")
GESTURES=("Key","Pinch","Power","Tripod")


@dataclass(frozen=True,slots=True)
class EmgFmgWindows:
    batch:FeatureBatch;labels:np.ndarray;subjects:np.ndarray;loads:np.ndarray;positions:np.ndarray;trials:np.ndarray;sample_weight:np.ndarray

    def take(self,indices:np.ndarray)->"EmgFmgWindows":
        index=np.asarray(indices)
        return EmgFmgWindows(self.batch.take(index),self.labels[index],self.subjects[index],self.loads[index],self.positions[index],self.trials[index],self.sample_weight[index])


def load_emg_fmg_windows(archive:str|Path,*,subjects:Iterable[int],loads:Iterable[int],positions:Iterable[int],
                         window_ms:float=200.0,maximum_windows_per_trial:int=8)->EmgFmgWindows:
    subject_set={int(x) for x in subjects};load_set={int(x) for x in loads};position_set={int(x) for x in positions};size=round(2000*window_ms/1000)
    rows={key:[] for key in ("emg","labels","subjects","loads","positions","trials","weight")}
    with zipfile.ZipFile(archive) as handle:
        for member in sorted(handle.namelist()):
            match=PATH_RE.match(member)
            if match is None:continue
            subject=int(match.group("subject"));load=int(match.group("load"));position=int(match.group("position"))
            if subject not in subject_set or load not in load_set or position not in position_set:continue
            values=np.loadtxt(BytesIO(handle.read(member)),delimiter=",",skiprows=1,dtype=np.float32,usecols=range(8,16))
            if values.shape!=(36000,8) or not np.all(np.isfinite(values)):raise ValueError(f"invalid EMG-FMG CSV: {member} {values.shape}")
            # Published recordings last 18 s; use evenly spaced windows from the central 9 s hold region.
            low,high=len(values)//4,3*len(values)//4
            candidates=np.arange(low,high-size+1,size)
            starts=candidates[np.linspace(0,len(candidates)-1,maximum_windows_per_trial).round().astype(int)] if len(candidates)>maximum_windows_per_trial else candidates
            for start in starts:
                rows["emg"].append(values[start:start+size]);rows["labels"].append(GESTURES.index(match.group("gesture")))
                rows["subjects"].append(subject);rows["loads"].append(load);rows["positions"].append(position);rows["trials"].append(member);rows["weight"].append(1/len(starts))
    if not rows["emg"]:raise ValueError("no EMG-FMG files matched")
    weights=np.asarray(rows["weight"],dtype=float);weights*=len(weights)/weights.sum()
    return EmgFmgWindows(FeatureBatch(np.stack(rows["emg"]),2000.0),np.asarray(rows["labels"]),np.asarray(rows["subjects"]),
        np.asarray(rows["loads"]),np.asarray(rows["positions"]),np.asarray(rows["trials"]),weights)
