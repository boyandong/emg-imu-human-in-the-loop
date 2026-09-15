"""Native full recordings; recording intent is not an interval/trial label."""
from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np

MOVEMENTS=('PS','3F','PP','FL','EX','RD','UD','PR','SU')
NAME_RE=re.compile(r'ID(?P<subject>\d+)_(?P<movement>PS|3F|PP|FL|EX|RD|UD|PR|SU)_(?P<position>P[123])\.txt')


def parse_recording_name(name):
    match=NAME_RE.fullmatch(Path(name).name)
    if match is None or int(match['subject']) not in range(1,11):
        raise ValueError('Invalid native subject/movement/position recording name')
    return int(match['subject']),match['movement'],match['position']


@dataclass(frozen=True,slots=True)
class ElectrodeReplacementRecording:
    emg:np.ndarray
    subject:int
    movement:str
    position:str
    recording_id:str
    sample_rate_hz:float=1000.
    interval_labels:None=None
    repetition_boundaries:None=None


def load_electrode_replacement_recording(path):
    path=Path(path);subject,movement,position=parse_recording_name(path.name)
    emg=np.loadtxt(path,dtype=np.float64)
    if emg.ndim!=2 or emg.shape[1]!=8 or len(emg)<3 or not np.isfinite(emg).all():
        raise ValueError('Expected finite native eight-channel full EMG recording')
    return ElectrodeReplacementRecording(emg,subject,movement,position,path.stem)
