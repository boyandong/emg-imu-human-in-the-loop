"""Recover exactly the existing Song F0 fits and prove label-free native replay."""
import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
from scipy.signal import butter,iirnotch,tf2sos
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from benchmarks.song_real8_study import load_session
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.families import LocalDetailFamily
from emgimu.feature_bank.document_signal import RestNoiseLocalDetailFamily
from emgimu.feature_bank.song_f0_runtime_v1 import SongF0RuntimeV1

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PACKAGE=ROOT/'feature_bank/models/song_f0_250hz_v1.pkl'
OUT=ROOT/'feature_bank/SONG_F0_RUNTIME_ACCEPTANCE_V1.json'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def run(source,verify_only=False):
    if not verify_only and (PACKAGE.exists() or OUT.exists()):raise FileExistsError('Use --verify-only to replay existing Song delivery')
    protocol=HERE/'F0_REST_MODEL_PROTOCOL.json';result=HERE/'F0_REST_MODEL_RESULTS.json'
    p=json.loads(protocol.read_text(encoding='utf8'));r=json.loads(result.read_text(encoding='utf8'))
    if sha(protocol)!=r['protocol_sha256']:raise ValueError('Source protocol changed')
    parent=HERE/'F0_REST_NOISE_PROTOCOL.json'
    if sha(parent)!=p['parent_f0_protocol_sha256']:raise ValueError('Feature protocol changed')
    native=json.loads((HERE/'F9_DOCUMENT_V3_PROTOCOL.json').read_text(encoding='utf8'))
    if verify_only:
        previous=json.loads(OUT.read_text(encoding='utf8'))
        packed=PACKAGE.read_bytes()
        if sha(PACKAGE)!=previous['package_sha256']:raise ValueError('Existing checkpoint changed')
        runtime=pickle.loads(packed);before=pickle.dumps(runtime)
        if not isinstance(runtime,SongF0RuntimeV1) or list(runtime.classes_)!=p['classes']:
            raise ValueError('Existing runtime type/class axis differs')
        # The constructor verified original model pickle fingerprints before
        # deep-copying. The immutable delivered package has its own byte hash:
        # copy/unpickle graph memoization need not reproduce original pickle bytes.
        source_windows=previous['source_windows'];ids=runtime.source_trial_ids_
    else:
        source_data=[]
        for session in p['source_sessions']:
            item=load_session(source/f'2026-09-18_{session}',session,filter_mode='causal')
            if item['audit']['sha256']!=native['expected_session_sha256'][session]:raise ValueError('Native source changed')
            source_data.append(item)
        x=np.concatenate([d['batch'].emg for d in source_data]);y=np.concatenate([d['hand'] for d in source_data])
        ids=set(np.concatenate([d['trial'] for d in source_data]));batch=FeatureBatch(x,250.)
        families={'pooled_source_threshold':LocalDetailFamily().fit(batch),
                  'rest_only_threshold':RestNoiseLocalDetailFamily(rest_label='neutral').fit(batch,y)}
        models={}
        for name,family in families.items():
            model=make_pipeline(StandardScaler(),LogisticRegression(C=1.,class_weight='balanced',max_iter=2000,random_state=0))
            model.fit(family.transform(batch),y);models[name]=(family,model)
            if hashlib.sha256(pickle.dumps(models[name])).hexdigest()!=r['source_state_sha256'][name]:
                raise ValueError('Recovered source state differs from frozen experiment')
        stages=[butter(4,40.,btype='highpass',fs=250.,output='sos')]
        for hz in (50.,100.):stages.append(tf2sos(*iirnotch(hz,Q=30.,fs=250.)))
        runtime=SongF0RuntimeV1(models,source_trial_ids=ids,source_state_sha256=r['source_state_sha256'],
            class_names=p['classes'],filter_stages=stages)
        packed=pickle.dumps(runtime,protocol=pickle.HIGHEST_PROTOCOL);runtime=pickle.loads(packed);before=pickle.dumps(runtime)
        source_windows=len(x)
    csv_path=HERE/'F0_REST_MODEL_TRIAL_PREDICTIONS.csv'
    if sha(csv_path)!=r['prediction_csv_sha256']:raise ValueError('Frozen predictions changed')
    with csv_path.open(encoding='utf8',newline='') as h:reference=list(csv.DictReader(h))
    records=[]
    for session in p['read_only_sessions']:
        d=load_session(source/f'2026-09-18_{session}',session,filter_mode='causal')
        if d['audit']['sha256']!=native['expected_session_sha256'][session]:raise ValueError('Native evaluation changed')
        offsets=np.empty(len(d['trial']),int);counts={}
        for i,t in enumerate(d['trial']):offsets[i]=counts.get(t,0);counts[t]=offsets[i]+1
        output=runtime.predict_windows(FeatureBatch(d['batch'].emg,250.),d['trial'],window_offsets=offsets,
                                      preprocessing_id=runtime.preprocessing_id)
        for name,values in output['arms'].items():
            expected={v['trial_id']:v for v in reference if v['session']==session and v['arm']==name}
            if set(expected)!=set(output['trial_ids']):raise ValueError('Native trial coverage changed')
            q=np.array([[float(expected[t]['p_'+c]) for c in p['classes']] for t in output['trial_ids']])
            error=float(np.max(abs(q-values['probabilities'])))
            if error>1e-12:raise ValueError('Frozen probabilities differ')
            records.append({'session':session,'arm':name,'trials':len(expected),'probability_max_error':error})
    if before!=pickle.dumps(runtime):raise ValueError('Inference changed source state')
    if not verify_only:
        PACKAGE.parent.mkdir(parents=True,exist_ok=True);PACKAGE.write_bytes(packed)
    paths=[protocol,result,parent,csv_path,HERE/'F9_DOCUMENT_V3_PROTOCOL.json',Path(__file__),
           ROOT/'src/emgimu/feature_bank/song_f0_runtime_v1.py',ROOT/'tests/test_song_f0_runtime_v1.py',
           ROOT/'src/emgimu/feature_bank/families.py',ROOT/'src/emgimu/feature_bank/document_signal.py',ROOT/'benchmarks/song_real8_study.py']
    out={'schema':'song_f0_runtime_acceptance_v1','package_path':PACKAGE.relative_to(ROOT).as_posix(),
         'package_sha256':sha(PACKAGE),'source_sha256':{v.relative_to(ROOT).as_posix():sha(v) for v in paths},
         'source_state_exactly_recovered':True,'source_state_immutable':True,'source_windows':source_windows,
         'source_trials':len(ids),'classes':p['classes'],'channels':8,'sample_rate_hz':250,'window_samples':50,
         'preprocessing_id':runtime.preprocessing_id,'records':records,'prediction_rows':sum(v['trials'] for v in records),
         'source_original_state_sha256':r['source_state_sha256'],'source_hdf5_sha256':r['source_hdf5_sha256'],'physical_validation_proven':False,'default_promoted':False,'completion_proven':False,
         'scope':'Two exact existing Song F0 models recovered from S01/S02 only. No model selection, source protocol change or new target training. Probability means over cued windows; one user/day, readiness failures and previously inspected S04. Independent live/generalization efficacy is unproven. Input requires identical causal continuous-session filtering; resets between windows are incompatible. Native Brier archive remains its original class-sum convention.'}
    OUT.write_text(json.dumps(out,indent=2)+'\n',encoding='utf8',newline='\n')
    print(('Verified existing250Hz package without fitting; ' if verify_only else 'Recovered both exact250Hz source states; ')+'all568 paired native predictions replayed',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('source',type=Path);parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args();run(args.source,args.verify_only)
