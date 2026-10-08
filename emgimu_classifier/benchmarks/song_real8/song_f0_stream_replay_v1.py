"""Frozen full-recording Song stream replay; no new model fitting."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import h5py
import numpy as np
from benchmarks.song_real8_study import _filter_emg,parse_label
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.song_f0_stream_v1 import SongF0StreamV1

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PROTOCOL=HERE/'SONG_F0_STREAM_V1_PROTOCOL.json'
RESULT=HERE/'SONG_F0_STREAM_V1_RESULTS.json'
ARRAYS=HERE/'SONG_F0_STREAM_V1_EMISSIONS.npz'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def metrics(truth,prediction,trials):
    eligible=truth>=0;y=truth[eligible];pred=prediction[eligible];ids=trials[eligible]
    unique,counts=np.unique(ids,return_counts=True);weights=np.array([1/dict(zip(unique,counts))[t] for t in ids])
    cm=np.zeros((4,5));index=np.where(pred<0,4,pred)
    np.add.at(cm,(y,index),weights)
    tp=np.diag(cm[:,:4]);den=cm.sum(1)+cm[:,:4].sum(0)
    f1=np.divide(2*tp,den,out=np.zeros(4),where=den>0)
    holds=[]
    for trial in unique:
        p=pred[ids==trial];target=y[ids==trial]
        if len(set(target))!=1:raise ValueError('Mixed native trial labels')
        holds.append({'trial_id':int(trial),'windows':len(p),'correct_windows':int(np.sum(p==target)),
                      'whole_stable_hold_correct':bool(np.all(p==target)),
                      'switches':int(np.sum(np.diff(p)!=0))})
    return {'eligible_windows':len(y),'eligible_trials':len(unique),'unknown_windows':int(np.sum(pred<0)),
        'trial_balanced_accuracy':float(np.sum(weights*(y==pred))/len(unique)),
        'trial_balanced_macro_f1':float(f1.mean()),'trial_balanced_confusion':cm.tolist(),
        'whole_stable_hold_correct':sum(r['whole_stable_hold_correct'] for r in holds),
        'within_stable_trial_switches':sum(r['switches'] for r in holds),'trials':holds}


def prepare(source):
    if PROTOCOL.exists():raise FileExistsError('Frozen protocol already exists')
    native=json.loads((HERE/'F9_DOCUMENT_V3_PROTOCOL.json').read_text(encoding='utf8'))
    package=ROOT/'feature_bank/models/song_f0_250hz_v1.pkl'
    paths=[Path(__file__),ROOT/'src/emgimu/feature_bank/song_f0_stream_v1.py',
        ROOT/'src/emgimu/feature_bank/song_f0_runtime_v1.py',ROOT/'src/emgimu/feature_bank/causal_label_debounce_v1.py',
        ROOT/'src/emgimu/feature_bank/families.py',ROOT/'src/emgimu/feature_bank/document_signal.py',
        ROOT/'src/emgimu/feature_bank/core.py',ROOT/'benchmarks/song_real8_study.py',
        ROOT/'tests/test_song_f0_stream_v1.py',ROOT/'tests/test_song_f0_stream_v1_delivery.py']
    p={'schema':'song_f0_stream_v1','source_folder':str(source),'target_sessions':['S03','S04'],
       'hdf5_sha256':{s:native['expected_session_sha256'][s] for s in ['S03','S04']},
       'package_path':package.relative_to(ROOT).as_posix(),'package_sha256':sha(package),
       'sample_rate_hz':250,'channels':8,'window_samples':50,'hop_samples':10,'confirmations':2,
       'chunk_samples':4096,'source_sha256':{v.relative_to(ROOT).as_posix():sha(v) for v in paths},
       'scope':'Retrospective full raw Song recording replay of both fixed F0 arms; no new fits, target tuning or default selection. Labels enter scoring only. Every emission uses trailing50 samples and original zero-start continuous causal filter. Two-consecutive-class confirmation starts UNKNOWN. Only windows wholly inside valid completed formal stable intervals are scored; all other emissions remain unscored, never fake Neutral. Equal native trial mass and whole-stable-interval hold counts are reported. Cue intervals are not biological transition labels, autonomous onset detection or hardware latency. One participant/day, readiness failures and previous inspection persist.',
       'default_promoted':False}
    PROTOCOL.write_text(json.dumps(p,indent=2)+'\n',encoding='utf8',newline='\n')
    print('Frozen Song250Hz stream/confirmation replay; no target raw signals loaded',flush=True)


def run():
    if RESULT.exists() or ARRAYS.exists():raise FileExistsError('Do not overwrite completed stream replay')
    p=json.loads(PROTOCOL.read_text(encoding='utf8'))
    for name,digest in p['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen implementation changed')
    if sha(ROOT/p['package_path'])!=p['package_sha256']:raise ValueError('Source checkpoint changed')
    runtime=pickle.loads((ROOT/p['package_path']).read_bytes());before=pickle.dumps(runtime);arrays={};records=[]
    for session in p['target_sessions']:
        path=Path(p['source_folder'])/f'2026-09-18_{session}'/'session.h5'
        if sha(path)!=p['hdf5_sha256'][session]:raise ValueError('Native recording changed')
        with h5py.File(path) as h:raw=h['streams/emg/raw'][:];annotations=h['trials'][:]
        dense=np.full(len(raw),-1,int);trials=np.full(len(raw),-1,int)
        for row in annotations:
            if row['trial_kind']!=b'formal' or not row['valid'] or row['completion_status']!=b'completed':continue
            a,b=int(row['stable_start_sample']),int(row['stable_end_sample'])
            if not 0<=a<b<=len(raw):continue
            if np.any(trials[a:b]>=0):raise ValueError('Overlapping native stable trials')
            hand=parse_label(row['label'].decode('utf8'))[1]
            dense[a:b]=runtime.classes_.index(hand);trials[a:b]=int(row['trial_id'])
        ends=np.arange(49,len(raw),10);eligible=(trials[ends]>=0)&(trials[ends]==trials[ends-49])
        truth=np.where(eligible,dense[ends],-1);trial=np.where(eligible,trials[ends],-1)
        arrays[session+'_ends']=ends;arrays[session+'_truth']=truth;arrays[session+'_trials']=trial
        filtered=_filter_emg(raw,'causal');windows=np.stack([filtered[e-49:e+1] for e in ends])
        for arm in runtime.models_:
            stream=SongF0StreamV1(runtime,arm=arm,recording_id=session,hop_samples=p['hop_samples'],confirmations=p['confirmations'])
            outputs=[]
            for start in range(0,len(raw),p['chunk_samples']):
                outputs+=stream.push(raw[start:start+p['chunk_samples']],first_sample_index=start,sample_rate_hz=250.)
            if [o['end_sample'] for o in outputs]!=ends.tolist():raise ValueError('Stream grid differs')
            q=np.stack([o['probabilities'] for o in outputs]);family,model=runtime.models_[arm]
            columns=[list(model[-1].classes_).index(c) for c in runtime.classes_]
            reference=model.predict_proba(family.transform(FeatureBatch(windows,250.)))[:,columns]
            error=float(np.max(abs(q-reference)))
            if error>1e-12:raise ValueError('Stream differs from independent full-recording oracle')
            raw_labels=np.array([o['raw_class_index'] for o in outputs]);confirmed=np.array([o['confirmed_class_index'] for o in outputs])
            arrays[session+'_'+arm+'_probabilities']=q
            arrays[session+'_'+arm+'_raw']=raw_labels;arrays[session+'_'+arm+'_confirmed']=confirmed
            records.append({'session':session,'arm':arm,'raw_samples':len(raw),'emissions':len(ends),
                'unscored_emissions':int(np.sum(~eligible)),'one_pass_max_probability_error':error,
                'raw':metrics(truth,raw_labels,trial),'confirmed':metrics(truth,confirmed,trial)})
            print(f'{session}/{arm}: complete recording replayed; no training',flush=True)
        del windows,filtered
    if before!=pickle.dumps(runtime):raise ValueError('Stream changed source-fitted runtime')
    np.savez_compressed(ARRAYS,**arrays)
    out={'schema':'song_f0_stream_v1','protocol_sha256':sha(PROTOCOL),'emissions_sha256':sha(ARRAYS),
        'classes':list(runtime.classes_),'records':records,'source_state_immutable':True,
        'default_promoted':False,'physical_validation_proven':False,'completion_proven':False,'scope':p['scope']}
    RESULT.write_text(json.dumps(out,indent=2)+'\n',encoding='utf8',newline='\n')
    print('Saved continuous probabilities/labels and stable-hold scoring; no device efficacy claim',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',type=Path);args=parser.parse_args()
    prepare(args.prepare) if args.prepare else run()
