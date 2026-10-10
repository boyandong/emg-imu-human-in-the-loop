"""Precommitted native8/200Hz cue-interval joint study; inspected public users.

No target search, invented Rest, interval interpolation or biological boundary
claim. Posture recordings are different domains, not days or redonnings.
"""
import argparse
import csv
import hashlib
import json
import pickle
import platform
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import scipy
from scipy.special import softmax
import sklearn
from sklearn.metrics import f1_score,confusion_matrix

from emgimu.datasets.roam_cued_intervals_v1 import (
    load_roam_cued_intervals,take_roam_intervals,RoamCuedIntervalsV1,CHANNELS,CLASSES,PREPROCESSING)
from emgimu.feature_bank.personal_temporal_bouts_v1 import TemporalBoutBatchV1
from emgimu.feature_bank.native_bout_window_adapter_v2 import native_windows
from emgimu.feature_bank.native_document_window_v2 import fit_native_document_source,NativeDocumentWindowDecisionV2
from emgimu.feature_bank.native_joint_bout_workflow_v2 import NativeJointBoutWorkflowV2
from emgimu.feature_bank.document_window_composition_v1 import GROUPS,FAMILY_GROUPS,source_trial_features
from emgimu.feature_bank.frozen_emg_provider_bank_v1 import FrozenEmgProviderBankV1
from emgimu.feature_bank.personal_session_workflow_v1 import PersonalSessionWorkflowV1
from emgimu.feature_bank.force_nested_oof import fit_temperature,temperature_probability

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
REPOSITORY=ROOT.parent
PROTOCOL=HERE/'ROAM_NATIVE_JOINT_V1_PROTOCOL.json'
RESULT=HERE/'ROAM_NATIVE_JOINT_V1_RESULTS.json'
OUT=HERE/'roam_native_joint_v1'
ARCHIVE=Path('D:/emg-imu-benchmarks/data/raw/roam_emg/data.zip')
ARCHIVE_SHA='c9de0e25c187c216e4771d17ccfa19acb0e4028f907ea726b2a87a18d2db5343'
SEQUENCE=('relax','open','close','open','relax','open','relax','close','relax')
ZERO_COST=('population','uniform','single_F0','single_CSP')
SCOPE=('Retrospective native8-channel200Hz three-class ROAM oracle cue-interval classification. '
       'Source users1..18 resting; target long resting and current hanging recordings are separate from '
       'unsupported/reaching queries. Recording domains are not chronological sessions, days or redonnings. '
       'Every eligible interval native sample enters full-path processing; overlapping window coverage and tails '
       'are reported. Previously inspected19..28 users, no new blind cohort. No pinch, autonomous segmentation, '
       'physiological action boundaries, physical ring/F6, ADC-bound F9, device latency or efficacy claim.')


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    def convert(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        raise TypeError(type(v).__name__)
    with Path(path).open('x',encoding='utf8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,default=convert);stream.write('\n')


def merge(items):
    batches=[i.batch for i in items]
    batch=TemporalBoutBatchV1(*(tuple(v for b in batches for v in getattr(b,name))
        for name in ('sequences','trial_ids','recording_ids','starts')),200.,CHANNELS,PREPROCESSING,'complete_cued').validate()
    labels={t:c for i in items for t,c in i.labels.items()}
    if len(labels)!=len(batch.trial_ids):raise ValueError('Repeated native identity')
    return RoamCuedIntervalsV1(batch,labels,{})


def select(data,shots):
    ids=tuple(t for c in CLASSES for t in [t for t in data.batch.trial_ids if data.labels[t]==c][:shots])
    if len(ids)!=shots*len(CLASSES):raise ValueError('Insufficient independent cued calibration trials')
    return take_roam_intervals(data,ids)


def load():
    data={}
    with zipfile.ZipFile(ARCHIVE) as z:
        for u in range(1,29):
            for condition in (('resting',) if u<19 else ('resting','hanging','unsupported','reaching')):
                m=f'data/ROAM_EMG/s{u}/s{u}_static_{condition}.csv'
                d=load_roam_cued_intervals(z,m)
                if tuple(d.labels.values())!=SEQUENCE or d.receipt['excluded_intervals']:
                    raise ValueError('Frozen nine-cue coverage differs; do not silently change trial denominator')
                data[u,condition]=d
    return data


def score(labels,q):
    truth=np.asarray([CLASSES.index(c) for c in labels]);q=np.asarray(q,float)
    if q.shape!=(len(truth),3) or not np.isfinite(q).all() or np.any(q<0) or not np.allclose(q.sum(1),1.,rtol=0,atol=1e-12):
        raise ValueError('Finite matched three-class probability axis required')
    pred=q.argmax(1);cm=confusion_matrix(truth,pred,labels=np.arange(3))
    return dict(trials=len(truth),macro_f1=float(f1_score(truth,pred,labels=np.arange(3),average='macro',zero_division=0)),
        accuracy=float(np.mean(truth==pred)),log_loss=float(-np.log(np.maximum(q[np.arange(len(q)),truth],1e-15)).mean()),
        brier=float(np.mean((q-np.eye(3)[truth])**2)),confusion=cm.tolist(),
        recall={c:float(cm[i,i]/cm[i].sum()) for i,c in enumerate(CLASSES)})


def fit_item(item,*,forbidden=()):
    b,ids,_,_=native_windows(item.batch,40,40)
    y=np.array([item.labels[t] for t in ids])
    return fit_native_document_source(b,y,ids,rest_label='relax',forbidden_trials=forbidden,seed=20261011)


def source_read(families,models,item):
    b,ids,_,_=native_windows(item.batch,40,40);raw={}
    for g in GROUPS:
        x,axis=source_trial_features(families[g],b,ids);s,m=models[g]
        raw[g]=m.predict_proba(s.transform(x))
    return raw,axis,np.array([item.labels[t] for t in axis])


def build(bank):
    base=PersonalSessionWorkflowV1(bank,channel_ids=CHANNELS,preprocessing_id=PREPROCESSING,
        rest_label='relax',quality_options={})
    return NativeJointBoutWorkflowV2(NativeDocumentWindowDecisionV2(base))


def prepare():
    if sha(ARCHIVE)!=ARCHIVE_SHA:raise ValueError('Native archive checksum differs')
    data=load();sources=list((ROOT/'src/emgimu/feature_bank').glob('*.py'))
    sources += [ROOT/'src/emgimu/datasets/roam_cued_intervals_v1.py',Path(__file__),
        HERE/'verify_roam_native_joint_v1.py',ROOT/'tests/test_native_joint_bout_workflow_v2.py',
        ROOT/'tests/test_roam_cued_intervals_v1.py',ROOT/'tests/test_roam_native_joint_study_v1.py']
    write(PROTOCOL,dict(schema='roam_native_joint_v1',archive=str(ARCHIVE),archive_sha256=ARCHIVE_SHA,
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sources},
        native_recordings=[d.receipt for d in data.values()],source_users=list(range(1,19)),
        source_folds=[list(range(1,7)),list(range(7,13)),list(range(13,19))],
        validation_users=list(range(19,24)),descriptive_users=list(range(24,29)),current_shots=[0,1,2],
        source_posture='resting',long_posture='resting',long_shots=2,current_posture='hanging',
        query_postures=['unsupported','reaching'],classes=CLASSES,groups=GROUPS,channels=CHANNELS,
        sample_rate_hz=200.,source_window_samples=40,source_hop_samples=40,inference_hop_samples=8,
        feature_dimensions=[48,8,72,8,72,57,12],source_fit='All seven families, scalers and balanced logistic C1 models independently inside3 user folds, then joined source1..18. F0 source-Rest noise; CSP source-only3-class. Equal cue means of disjoint200ms source windows; source tails are not classifier samples.',
        probability_policy='Provider temperatures from concatenated source OOF probabilities; population softmax(-OOF logloss). n0=4 and reliability_temperature=.5 fixed without target search. OOF-fitted policy training loss is not an unbiased generalization estimate.',
        calibration='First2 cues/class from target resting for long registration; first0/1/2 cues/class from separate hanging for current domain. Unused cues from both calibration recordings excluded from queries. Same unsupported+reaching18 queries at every budget. Six long plus0/3/6 current independent cues, charged once across branches.',
        composition='Window F7 anchor_mix=.5, F8 calibration routing, full-cue32 RMS-bin DTW/signature mixtures. Full=.75 window+.125 DTW+.125 signature. Fixed-weight provider omissions renormalize existing window weights; whole-F2 removes covariance/tangent and CSP. Whole-F5 removes windowF5 and both temporal branches. WindowF7/F8 removals do not remove temporal calibration.',
        primary='At2 current cues/class, joint_full versus window_full on validation19..23: lower pooled logloss/Brier, nonworse macroF1/all3class recalls and at least3/5 user logloss wins.24..28 and pooled results descriptive; no default promotion even if guards pass.',
        brier_definition='mean squared probability error over trials and3 classes',scope=SCOPE,
        raw_quality_policy_available=False,default_promoted=False,completion_proven=False))


def check_frozen(p):
    for name,digest in p['source_sha256'].items():
        path=ROOT/name
        if sha(path)!=digest:raise ValueError('Frozen implementation changed: '+name)
    for path in [PROTOCOL,*[ROOT/n for n in p['source_sha256']]]:
        relative=path.relative_to(REPOSITORY).as_posix()
        committed=subprocess.run(['git','show','HEAD:'+relative],cwd=REPOSITORY,check=True,capture_output=True).stdout
        if committed!=path.read_bytes():raise ValueError('Protocol/source must be byte-exactly committed before native fitting: '+relative)
    if sha(ARCHIVE)!=p['archive_sha256']:raise ValueError('Native archive checksum differs')


def window_omission(window,active):
    names=tuple(window['decision_provider_probabilities']);weights=np.array(window['weights'],float)
    positions=[names.index(g) for g in names if g in active];weights=weights[positions];weights/=weights.sum()
    q=sum(w*window['decision_provider_probabilities'][names[i]] for w,i in zip(weights,positions))
    return q/q.sum(1,keepdims=True)


def derive(joint,other_windows,source):
    w=joint['window'];order=[w['trial_ids'].index(t) for t in joint['trial_ids']]
    t=joint['temporal']['arms'];dtw,sig=t['DTW_blended'],t['signature_blended']
    combine=lambda q:.75*q+.125*dtw+.125*sig
    arms=dict(source)
    arms.update({name:r['probabilities'][[r['trial_ids'].index(t) for t in joint['trial_ids']]] for name,r in other_windows.items()})
    arms['window_full']=w['probabilities'][order]
    arms.update(joint_full=joint['probabilities'],joint_DTW=.75*arms['window_full']+.25*dtw,
        joint_signature=.75*arms['window_full']+.25*sig,joint_uniform=.75*arms['window_full']+.25/3,
        joint_long_templates=.75*arms['window_full']+.125*t['DTW_long']+.125*t['signature_long'],
        joint_minus_window_F7=combine(arms['window_F8']),joint_minus_window_F8=combine(arms['window_F7']),
        joint_minus_temporal=arms['window_full'])
    for g in GROUPS:
        arms['minus_provider_'+g]=combine(window_omission(w,tuple(n for n in GROUPS if n!=g))[order])
    for g,members in FAMILY_GROUPS.items():
        q=window_omission(w,tuple(n for n in GROUPS if n not in members))[order]
        # Whole F5 includes both temporal branches, unlike a window-provider omission.
        arms['minus_family_'+g]=q if g=='F5' else combine(q)
    return arms


def run():
    if RESULT.exists() or OUT.exists():raise FileExistsError('Native study has started or completed; do not rerun')
    p=json.loads(PROTOCOL.read_text(encoding='utf8'));check_frozen(p);data=load()
    if json.loads(json.dumps([d.receipt for d in data.values()]))!=p['native_recordings']:
        raise ValueError('Native cue identities/coverage changed')
    OUT.mkdir();folds=[];target_ids=tuple(t for (u,_),d in data.items() if u>=19 for t in d.batch.trial_ids)
    print('1/4 Native source-user OOF fits (three folds and final source bank)',flush=True)
    for i,held in enumerate(p['source_folds']):
        train=merge([data[u,'resting'] for u in p['source_users'] if u not in held])
        valid=merge([data[u,'resting'] for u in held])
        f,m,meta=fit_item(train,forbidden=(*valid.batch.trial_ids,*target_ids))
        package=OUT/f'source_fold{i}.pkl';package.write_bytes(pickle.dumps((f,m),protocol=pickle.HIGHEST_PROTOCOL))
        raw,axis,labels=source_read(f,m,valid)
        folds.append(dict(held_users=held,fit_ids=sorted(train.batch.trial_ids),ids=axis,labels=labels,raw=raw,
            path=package.relative_to(ROOT).as_posix(),sha256=sha(package)))
        print(f'  source fold {i+1}/3 complete',flush=True)
    labels=np.concatenate([f['labels'] for f in folds]);encoded=np.array([CLASSES.index(c) for c in labels])
    temperatures={g:fit_temperature(np.concatenate([f['raw'][g] for f in folds]),encoded) for g in GROUPS}
    losses={g:score(labels,temperature_probability(np.concatenate([f['raw'][g] for f in folds]),temperatures[g]))['log_loss'] for g in GROUPS}
    population=softmax(-np.array([losses[g] for g in GROUPS]));joined=merge([data[u,'resting'] for u in p['source_users']])
    f,m,meta=fit_item(joined,forbidden=target_ids)
    bank=FrozenEmgProviderBankV1(f,m,temperatures,classes=CLASSES,class_names=CLASSES,
        source_trial_ids=tuple(sorted(joined.batch.trial_ids)),source_policy_id=sha(PROTOCOL),
        sample_rate_hz=200.,window_samples=40,channels=8,population=population,n0=4.,reliability_temperature=.5)
    w=build(bank);frozen=pickle.dumps(w);package=OUT/'source_bank.pkl';package.write_bytes(pickle.dumps(bank))
    write(OUT/'policy.json',dict(schema='roam_native_joint_v1',bank_sha256=sha(package),bank_id=bank.bank_id_,
        window_policy_id=w.window.policy_id,joint_contract_id=w.contract_id,channels=CHANNELS,preprocessing=PREPROCESSING,
        rest_label='relax',sample_rate_hz=200.,quality_policy_available=False))
    rows=[];cells=[];arrays={};subjects=[];roundtrips=[]
    print('2/4 Shared long/current registration and fixed native queries',flush=True)
    for u in range(19,29):
        user=f'ROAM_s{u}';long=select(data[u,'resting'],2)
        query=merge([data[u,c] for c in p['query_postures']]);ey=[query.labels[t] for t in query.batch.trial_ids]
        forbidden=dict(forbidden_trial_ids=query.batch.trial_ids,forbidden_recording_ids=query.batch.recording_ids)
        personal=w.enroll(long.batch,long.labels,user_id=user,session_id='resting',**forbidden)
        pp=OUT/f's{u}_personal.zip';w.save_profile(personal,pp)
        loaded=w.load_profile(pp,user_id=user)
        qb,qi,qo,tails=native_windows(query.batch,40,8)
        source=bank.predict_providers(qb,qi,window_offsets=qo,user_id=user)
        order=[source['trial_ids'].index(t) for t in query.batch.trial_ids]
        source_probs={g:q[order] for g,q in source['probabilities'].items()}
        controls=dict(population=sum(population[i]*source_probs[g] for i,g in enumerate(GROUPS)),
            uniform=np.mean(list(source_probs.values()),axis=0),single_F0=source_probs['F0'],single_CSP=source_probs['F2b'])
        for g,q in source_probs.items():arrays[f'u{u}_source_{g}']=q
        subject=dict(user=u,query_ids=query.batch.trial_ids,query_labels=ey,query_recordings=query.batch.recording_ids,
            query_samples=[len(x) for x in query.batch.sequences],query_window_tail_samples=tails,
            personal_path=pp.relative_to(ROOT).as_posix(),personal_id=personal.profile_id,
            personal_calibration_ids=long.batch.trial_ids,personal_cost=personal.calibration_cost,budgets={})
        for shots in p['current_shots']:
            session=None;loaded_session=None;sp=None;current=None
            if shots:
                current=select(data[u,'hanging'],shots)
                session=w.enroll(current.batch,current.labels,user_id=user,session_id='hanging',personal=personal,**forbidden)
                sp=OUT/f's{u}_current{shots}.zip';w.save_profile(session,sp)
                loaded_session=w.load_profile(sp,user_id=user,session_id='hanging',personal=loaded)
            before=pickle.dumps((personal,session))
            common=dict(personal=personal,user_id=user,session_id='hanging',session=session)
            joint=w.predict(query.batch,**common)
            replay=w.predict(query.batch,personal=loaded,user_id=user,session_id='hanging',session=loaded_session)
            if not np.array_equal(joint['probabilities'],replay['probabilities']):raise ValueError('Saved joint profile changed native predictions')
            roundtrips.append(dict(user=u,shots=shots,maximum_probability_error=0.))
            window_common=dict(window_offsets=qo,personal=personal.window,session=None if session is None else session.window,
                user_id=user,session_id='hanging',observed_channel_ids=CHANNELS,preprocessing_id=PREPROCESSING)
            other={name:w.window.predict(qb,qi,**window_common,**options) for name,options in dict(
                window_reliability=dict(use_anchor=False,use_session_routing=False),
                window_F7=dict(use_session_routing=False),window_F8=dict(use_anchor=False)).items()}
            arms=derive(joint,other,controls);prefix=f'u{u}_s{shots}_'
            arrays[prefix+'window_weights']=np.asarray(joint['window']['weights'])
            for g,q in joint['window']['decision_provider_probabilities'].items():arrays[prefix+'decision_'+g]=q[order]
            for name in ('DTW_blended','signature_blended','DTW_long','signature_long'):
                arrays[prefix+name]=joint['temporal']['arms'][name]
            for arm,q in arms.items():
                arrays[prefix+arm]=q
                count=0 if arm in ZERO_COST else 6+3*shots
                cells.append(dict(user=u,phase='validation' if u<=23 else 'descriptive_final',shots=shots,arm=arm,
                    long_calibration_trials=0 if arm in ZERO_COST else 6,current_calibration_trials=0 if arm in ZERO_COST else 3*shots,
                    unique_calibration_trials=count,**score(ey,q)))
                for trial,label,prob in zip(query.batch.trial_ids,ey,q):
                    rows.append(dict(user=u,shots=shots,arm=arm,trial_id=trial,true_label=label,predicted=CLASSES[prob.argmax()],
                        **{f'p_{c}':float(prob[i]) for i,c in enumerate(CLASSES)}))
            if pickle.dumps((personal,session))!=before:raise ValueError('Prediction mutated shared calibration')
            subject['budgets'][str(shots)]=dict(session_path=None if sp is None else sp.relative_to(ROOT).as_posix(),
                session_id=None if session is None else session.profile_id,
                current_calibration_ids=[] if current is None else current.batch.trial_ids,
                current_cost=None if session is None else session.calibration_cost,arms=tuple(arms))
        subjects.append(subject);print(f'  target user {u-18}/10 complete',flush=True)
    if pickle.dumps(w)!=frozen:raise ValueError('Native source workflow mutated')
    print('3/4 Paired budget and whole-family readouts',flush=True)
    aggregates=[]
    for phase,users in [('validation',range(19,24)),('descriptive_final',range(24,29)),('descriptive_all',range(19,29))]:
        ey=[c for s in subjects if s['user'] in users for c in s['query_labels']]
        for shots in p['current_shots']:
            for arm in subjects[0]['budgets'][str(shots)]['arms']:
                q=np.concatenate([arrays[f'u{u}_s{shots}_{arm}'] for u in users])
                aggregates.append(dict(phase=phase,shots=shots,arm=arm,**score(ey,q)))
    find=lambda arm:next(c for c in aggregates if c['phase']=='validation' and c['shots']==2 and c['arm']==arm)
    base,full=find('window_full'),find('joint_full')
    wins=sum(next(c for c in cells if c['user']==u and c['shots']==2 and c['arm']=='joint_full')['log_loss']<
        next(c for c in cells if c['user']==u and c['shots']==2 and c['arm']=='window_full')['log_loss'] for u in range(19,24))
    guards=dict(lower_log_loss=full['log_loss']<base['log_loss'],lower_brier=full['brier']<base['brier'],
        nonworse_macro_f1=full['macro_f1']>=base['macro_f1'],nonworse_all_class_recalls=all(full['recall'][c]>=base['recall'][c] for c in CLASSES),
        at_least3_of5_user_loss_wins=wins>=3)
    np.savez_compressed(OUT/'readouts.npz',**arrays)
    with (OUT/'predictions.csv').open('x',encoding='utf8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    artifacts={p.relative_to(ROOT).as_posix():sha(p) for p in OUT.iterdir() if p.is_file()}
    write(RESULT,dict(schema='roam_native_joint_v1',protocol_sha256=sha(PROTOCOL),source_oof=folds,
        source_temperatures=temperatures,source_oof_losses=losses,source_population=population,source_metadata=meta,
        source_bank_id=bank.bank_id_,source_trial_ids=bank.policy_.source_trials,subjects=subjects,cells=cells,aggregates=aggregates,
        artifacts_sha256=artifacts,profile_roundtrips=roundtrips,source_unchanged=True,
        primary_guards=guards,primary_pass=all(guards.values()),validation_user_loss_wins=wins,
        prediction_rows=len(rows),independent_query_trials=180,source_cue_trials=162,native_recordings=58,
        environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__),
        scope=SCOPE,default_promoted=False,physical_validation_proven=False,completion_proven=False))
    print('4/4 Saved native results; independent verification remains required',flush=True)
    print(json.dumps(dict(cells=len(cells),predictions=len(rows),primary_guards=guards),ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    prepare() if args.prepare else run()
