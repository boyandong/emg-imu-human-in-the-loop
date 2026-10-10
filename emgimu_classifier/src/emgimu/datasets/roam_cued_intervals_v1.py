"""Complete native ROAM cue intervals, with labels separate from model input.

Ground-truth cue changes supply oracle interval boundaries. They are not
physiological onset/release annotations or autonomous detector outputs.
"""
from dataclasses import dataclass
import csv
import hashlib
import io
import numpy as np
from emgimu.feature_bank.personal_temporal_bouts_v1 import TemporalBoutBatchV1

CHANNELS=tuple(f'emg{i}' for i in range(8))
CLASSES=('close','open','relax')
LABELS={0:'relax',1:'open',2:'close'}
PREPROCESSING='roam_native_myo200_no_added_filter_v1'


@dataclass(frozen=True)
class RoamCuedIntervalsV1:
    batch: TemporalBoutBatchV1
    labels: dict
    receipt: dict


def load_roam_cued_intervals(archive,member):
    payload=archive.read(member)
    reader=csv.DictReader(io.StringIO(payload.decode('utf8')))
    if not set((*CHANNELS,'gt','time_elapsed'))<=set(reader.fieldnames or ()):
        raise ValueError('Missing native ROAM channel, cue or time columns')
    rows=[];labels=[];times=[]
    for row in reader:
        rows.append([float(row[c]) for c in CHANNELS]);labels.append(int(row['gt']));times.append(float(row['time_elapsed']))
    x=np.asarray(rows,dtype=np.float32);y=np.asarray(labels,dtype=int);time=np.asarray(times,float)
    if (x.ndim!=2 or x.shape[1]!=8 or len(x)<200 or not np.isfinite(x).all()
            or not np.isfinite(time).all() or not set(y)<=set(LABELS)
            or not .8*len(x)/200<=time[-1]-time[0]<=1.2*len(x)/200):
        raise ValueError('Native ROAM sensor, cue, length or nominal clock contract differs')
    edges=np.r_[0,np.flatnonzero(y[1:]!=y[:-1])+1,len(y)]
    sequences=[];ids=[];starts=[];mapping={};excluded=[];intervals=[]
    for i,(a,b) in enumerate(zip(edges[:-1],edges[1:])):
        identity=f'{member}:cue{i}';label=LABELS[int(y[a])]
        row=dict(trial_id=identity,start=int(a),end=int(b),samples=int(b-a),class_name=label)
        if not 1.<=(b-a)/200<=30.:
            excluded.append(dict(row,reason='outside_complete_path_1_to_30_second_contract'));continue
        sequences.append(x[a:b].copy());ids.append(identity);starts.append(int(a));mapping[identity]=label;intervals.append(row)
    if not sequences:raise ValueError('No eligible complete native cue intervals')
    batch=TemporalBoutBatchV1(tuple(sequences),tuple(ids),(member,)*len(ids),tuple(starts),200.,CHANNELS,
        PREPROCESSING,'complete_cued').validate()
    receipt=dict(member=member,member_sha256=hashlib.sha256(payload).hexdigest(),native_samples=len(x),
        archive_sample_rate_hz=200.,channel_ids=CHANNELS,cue_intervals=intervals,excluded_intervals=excluded,
        nominal_seconds=len(x)/200.,observed_clock_span=float(time[-1]-time[0]),
        source_annotation='gt contiguous cue-label intervals; oracle boundaries',
        labels_separate_from_inference_batch=True,resampled=False,biological_boundaries_proven=False,
        autonomous_segmentation_proven=False,physical_validation_proven=False)
    return RoamCuedIntervalsV1(batch,mapping,receipt)


def take_roam_intervals(data,trial_ids):
    ids=tuple(trial_ids)
    if not ids or len(set(ids))!=len(ids) or not set(ids)<=set(data.batch.trial_ids):
        raise ValueError('Explicit unique existing native cue interval identities required')
    positions=[data.batch.trial_ids.index(t) for t in ids];b=data.batch
    batch=TemporalBoutBatchV1(tuple(b.sequences[i] for i in positions),ids,
        tuple(b.recording_ids[i] for i in positions),tuple(b.starts[i] for i in positions),b.sample_rate_hz,
        b.channel_ids,b.preprocessing_id,b.boundary_kind)
    return RoamCuedIntervalsV1(batch,{t:data.labels[t] for t in ids},data.receipt)
