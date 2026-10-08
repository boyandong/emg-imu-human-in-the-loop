"""Independent source-only window classifier on full chronological ROAM files."""
import argparse
import csv
import hashlib
import json
import pickle
import zipfile
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, log_loss
from benchmarks.new_bank_v2.roam_posture_run import read_native, sha256
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.new_bank_v2 import RestNoiseDetailV2
from emgimu.feature_bank.causal_window_recognition_v1 import CausalWindowRecognizerV1, transition_hold_diagnostics

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PROTOCOL=HERE/'ROAM_CAUSAL_WINDOW_V1_PROTOCOL.json'
RESULT=HERE/'ROAM_CAUSAL_WINDOW_V1_RESULTS.json'
PREDICTIONS=HERE/'ROAM_CAUSAL_WINDOW_V1_EMISSIONS.csv'


def prepare():
    if PROTOCOL.exists(): raise FileExistsError('Frozen protocol exists')
    parent=json.loads((ROOT/'benchmarks/new_bank_v2/ROAM_POSTURE_PROTOCOL.json').read_text(encoding='utf8'))
    sources=[Path(__file__),ROOT/'src/emgimu/feature_bank/causal_window_recognition_v1.py',
             ROOT/'src/emgimu/feature_bank/new_bank_v2.py',ROOT/'src/emgimu/feature_bank/new_bank_v1.py',
             ROOT/'src/emgimu/feature_bank/core.py',ROOT/'benchmarks/new_bank_v2/roam_posture_run.py',
             ROOT/'benchmarks/new_bank_v2/ROAM_POSTURE_PROTOCOL.json']
    p={'schema':'roam_causal_window_v1','archive':parent['archive'],'archive_sha256':parent['archive_sha256'],
       'source_users':parent['source_subjects'],'validation_users':parent['validation_subjects'],
       'descriptive_final_users':parent['final_subjects'],'source_posture':'resting','target_postures':parent['target_postures'],
       'sample_rate_hz':200,'window_samples':40,'hop_samples':10,'chunk_samples':257,
       'reaction_half_buffer_samples':100,'source_bout_edge_exclusion_samples':40,
       'classifier':'Source-only StandardScaler and LogisticRegression C1,max_iter2000,seed20261008; equal class/bout mass sample weights. No tuning.',
       'feature':'Reference F0v2 six local metrics,48 coordinates; Rest thresholds from source-only windows. Not prior full-bout aggregate classifier.',
       'emission':'Trailing observed40-sample window, every10 samples. No future context. Hold latest probability between emissions; first39 samples explicitly unknown.',
       'evaluation':'All target samples and native cue transitions; no target labels used in fitting, segmentation or inference. Labels used only after chronological inference for metrics.',
       'scope':'New versioned descriptive public experiment on previously inspected cohorts. Native200Hz Myo three-class/static postures only; not blind new-user, biological onset, hardware throughput or own-device efficacy. Reaction/maintenance metric is independently specified, not exact ReactEMG reproduction.',
       'source_sha256':{path.relative_to(ROOT).as_posix():sha256(path) for path in sources},'default_promoted':False}
    PROTOCOL.write_text(json.dumps(p,indent=2)+'\n',encoding='utf8')
    print('Protocol frozen; no native signal read',flush=True)


def rle(values):
    changes=np.r_[0,np.flatnonzero(np.diff(values))+1,len(values)]
    return [[int(a),int(b),int(values[a])] for a,b in zip(changes[:-1],changes[1:])]


def run():
    if RESULT.exists() or PREDICTIONS.exists(): raise FileExistsError('Refuse to overwrite completed run')
    p=json.loads(PROTOCOL.read_text(encoding='utf8'))
    if any(sha256(ROOT/path)!=digest for path,digest in p['source_sha256'].items()): raise ValueError('Frozen implementation changed')
    if sha256(Path(p['archive']))!=p['archive_sha256']: raise ValueError('Archive changed')
    windows=[];labels=[];bout_ids=[];native_sources={}
    print('1/3 Read source-only resting recordings',flush=True)
    with zipfile.ZipFile(p['archive']) as archive:
        for user in p['source_users']:
            name=f'data/ROAM_EMG/s{user}/s{user}_static_resting.csv'
            native_sources[name]=hashlib.sha256(archive.read(name)).hexdigest()
            with archive.open(name) as handle: emg,y=read_native(handle,name)
            for bout,(a,b,label) in enumerate(rle(y)):
                a+=40;b-=40
                for start in range(a,b-39,40):
                    windows.append(emg[start:start+40]);labels.append(label);bout_ids.append(f'{name}:bout{bout}')
    batch=FeatureBatch(np.stack(windows),200);y=np.asarray(labels)
    family=RestNoiseDetailV2(rest_label=0).fit(batch,y)
    x=family.transform(batch);ids=np.asarray(bout_ids)
    weights=np.empty(len(y))
    for label in range(3):
        unique=np.unique(ids[y==label])
        for identity in unique:
            mask=ids==identity;weights[mask]=1/(len(unique)*mask.sum())
    weights*=len(y)/weights.sum()
    model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000,random_state=20261008))
    model.fit(x,y,standardscaler__sample_weight=weights,logisticregression__sample_weight=weights)
    if model[-1].n_iter_.max()>=2000:raise ValueError('Source model failed to converge')
    if not np.array_equal(model.classes_,[0,1,2]):raise ValueError('Source class axis differs')
    frozen=pickle.dumps((family,model));records=[]
    columns=['native_file','emission_sample','p_0','p_1','p_2']
    print('2/3 Chronological target inference, no target-label segmentation',flush=True)
    with PREDICTIONS.open('w',encoding='utf8',newline='') as out,zipfile.ZipFile(p['archive']) as archive:
        writer=csv.DictWriter(out,fieldnames=columns,lineterminator='\n');writer.writeheader()
        for phase,key in [('validation','validation_users'),('descriptive_final','descriptive_final_users')]:
            for user in p[key]:
                for posture in p['target_postures']:
                    name=f'data/ROAM_EMG/s{user}/s{user}_static_{posture}.csv'
                    digest=hashlib.sha256(archive.read(name)).hexdigest()
                    with archive.open(name) as handle:emg,truth=read_native(handle,name)
                    stream=CausalWindowRecognizerV1(family,model,sample_rate_hz=200,hop_samples=10)
                    ends=[];probabilities=[]
                    for start in range(0,len(emg),p['chunk_samples']):
                        e,q=stream.push(emg[start:start+p['chunk_samples']]);ends.extend(e);probabilities.extend(q)
                    ends=np.asarray(ends);probabilities=np.asarray(probabilities)
                    for end,q in zip(ends,probabilities):writer.writerow(dict(zip(columns,[name,int(end),*q.tolist()])))
                    dense=np.full((len(truth),3),np.nan);predicted=np.full(len(truth),-1,dtype=int)
                    for i,end in enumerate(ends):
                        stop=int(ends[i+1]) if i+1<len(ends) else len(truth)
                        dense[end:stop]=probabilities[i];predicted[end:stop]=probabilities[i].argmax()
                    valid=predicted>=0;yy=truth[valid];qq=dense[valid]
                    score={'accuracy':float(np.mean(predicted[valid]==yy)),
                           'macro_f1':float(f1_score(yy,predicted[valid],labels=range(3),average='macro',zero_division=0)),
                           'log_loss':float(log_loss(yy,qq,labels=range(3))),
                           'brier':float(np.mean((qq-np.eye(3)[yy])**2))}
                    diag=transition_hold_diagnostics(truth,predicted,rate_hz=200,half_buffer_samples=p['reaction_half_buffer_samples'])
                    records.append({'native_file':name,'native_sha256':digest,'user':user,'posture':posture,'phase':phase,
                                    'samples':len(truth),'unknown_warmup_samples':int(np.sum(~valid)),
                                    'emissions':len(ends),'truth_rle':rle(truth),'prediction_rle':rle(predicted),
                                    'scores':score,'transition_hold':diag})
                print(f'User{user}: all4 postures complete',flush=True)
    if frozen!=pickle.dumps((family,model)):raise ValueError('Target inference changed source predictor')
    summaries={}
    for phase in ('validation','descriptive_final'):
        summaries[phase]={}
        for posture in ['ALL',*p['target_postures']]:
            group=[r for r in records if r['phase']==phase and (posture=='ALL' or r['posture']==posture)]
            eligible=sum(r['transition_hold']['eligible_transitions'] for r in group)
            correct=sum(r['transition_hold']['correct_transitions'] for r in group)
            summaries[phase][posture]={'recordings':len(group),'equal_recording_macro_f1':float(np.mean([r['scores']['macro_f1'] for r in group])),
                                      'eligible_transitions':eligible,'correct_transitions':correct,
                                      'transition_hold_accuracy':correct/eligible if eligible else None,
                                      'maintenance_switches':sum(r['transition_hold']['maintenance_switches'] for r in group)}
    result={'schema':p['schema'],'protocol_sha256':sha256(PROTOCOL),'emissions_sha256':sha256(PREDICTIONS),
            'source_native_sha256':native_sources,'source_bout_ids':sorted(set(bout_ids)),
            'source_windows':len(y),'source_class_weight_mass':{str(c):float(weights[y==c].sum()) for c in range(3)},
            'source_state_immutable':True,'records':records,'summaries':summaries,
            'default_promoted':False,'physical_validation_proven':False,'scope':p['scope']}
    RESULT.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print('3/3 Saved full chronological control; no default promotion',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    prepare() if args.prepare else run()
