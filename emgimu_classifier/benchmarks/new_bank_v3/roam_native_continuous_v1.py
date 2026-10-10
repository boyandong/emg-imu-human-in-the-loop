"""Frozen eight-channel automatic detection/classification, without query cues.

References are active *cue* intervals, not physiological onset ground truth.
Existing native classifiers, calibration profiles and detector thresholds are
reused exactly. There is no fitting, target selection or policy search here.
"""
import argparse
import csv
import json
import pickle
import subprocess
import zipfile
from pathlib import Path
import numpy as np
from emgimu.datasets.roam_cued_intervals_v1 import load_roam_cued_intervals, CLASSES, CHANNELS, PREPROCESSING
from emgimu.feature_bank.native_joint_bout_stream_v1 import NativeJointBoutStreamV1
from benchmarks.new_bank_v3.roam_native_joint_v1 import sha, write, build, ARCHIVE, ARCHIVE_SHA

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
REPO=ROOT.parent
PROTOCOL=HERE/'ROAM_NATIVE_CONTINUOUS_V1_PROTOCOL.json'
RESULT=HERE/'ROAM_NATIVE_CONTINUOUS_V1_RESULTS.json'
OUT=HERE/'roam_native_continuous_v1'
OLD=HERE/'ROAM_NATIVE_JOINT_V1_RESULTS.json'
ARMS=('source_window','window_full','joint_full')
ACTIVE=('close','open')
SCOPE=('Retrospective native eight-channel200Hz continuous ROAM query recordings; no query '
       'labels or cue boundaries enter detection/classification. Saved target calibration profiles and '
       'their neutral-only fixed detectors are reused, not fitted. Active gt cue intervals are evaluation '
       'references, not physiological action boundaries. Postures are not days/redonnings; inspected '
       'users19..28 are not a new blind cohort. No pinch, own-device250Hz efficacy, F6/ring/raw-quality '
       'validation, physical latency or default promotion. Misses and unmatched detections remain visible.')


def matching(events, references, rate=200.):
    candidates=[]
    for i,e in enumerate(events):
        for j,r in enumerate(references):
            overlap=max(0,min(e['end'],r['end'])-max(e['start'],r['start']))
            iou=overlap/(max(e['end'],r['end'])-min(e['start'],r['start']))
            if iou>=.5:candidates.append((-iou,i,j))
    used_e=set();used_r=set();pairs=[]
    for negative,i,j in sorted(candidates):
        if i in used_e or j in used_r:continue
        used_e.add(i);used_r.add(j)
        pairs.append(dict(event_index=i,reference_index=j,iou=-negative,
            onset_error_s=(events[i]['start']-references[j]['start'])/rate,
            offset_error_s=(events[i]['end']-references[j]['end'])/rate))
    return pairs


def metrics(records, arm):
    refs=[r for x in records for r in x['references']]
    events=[e for x in records for e in x['events']]
    matched=[];onset=[];offset=[]
    for x in records:
        for m in x['matches']:
            matched.append((x['references'][m['reference_index']],x['events'][m['event_index']]))
            onset.append(abs(m['onset_error_s']));offset.append(abs(m['offset_error_s']))
    correct=sum(r['class_name']==e['labels'][arm] for r,e in matched)
    recalls={};f1=[]
    for c in ACTIVE:
        n=sum(r['class_name']==c for r in refs)
        tp=sum(r['class_name']==c==e['labels'][arm] for r,e in matched)
        pred=sum(e['labels'][arm]==c for e in events)
        recalls[c]=tp/n if n else 0.
        f1.append(2*tp/(n+pred) if n+pred else 0.)
    if matched:
        truth=np.array([CLASSES.index(r['class_name']) for r,e in matched])
        q=np.array([e['probabilities'][arm] for r,e in matched])
        loss=float(-np.log(np.maximum(q[np.arange(len(q)),truth],1e-15)).mean())
        brier=float(((q-np.eye(3)[truth])**2).mean())
    else:loss=brier=None
    return dict(references=len(refs),detections=len(events),matched=len(matched),
        missed=len(refs)-len(matched),unmatched_detections=len(events)-len(matched),correct=correct,
        detection_recall=len(matched)/len(refs) if refs else 0.,
        detection_precision=len(matched)/len(events) if events else 0.,
        end_to_end_success=correct/len(refs) if refs else 0.,
        active_event_macro_f1=float(np.mean(f1)),active_recall=recalls,
        conditional_accuracy=correct/len(matched) if matched else None,
        conditional_log_loss=loss,conditional_brier=brier,
        onset_cue_mae_s=float(np.mean(onset)) if onset else None,
        offset_cue_mae_s=float(np.mean(offset)) if offset else None,
        eof_censored_recordings=sum(x['censored_end'] for x in records))


def prepare():
    old=json.loads(OLD.read_text(encoding='utf8'))
    old_protocol=json.loads((HERE/'ROAM_NATIVE_JOINT_V1_PROTOCOL.json').read_text(encoding='utf8'))
    code=dict(old_protocol['source_sha256'])
    for p in (Path(__file__),HERE/'verify_roam_native_continuous_v1.py',
              HERE/'verify_calibration_rest_continuous_unibo_v1.py',
              ROOT/'benchmarks/song_real8/verify_integrated_decision_v1.py',
              ROOT/'src/emgimu/feature_bank/native_joint_bout_stream_v1.py',
              ROOT/'tests/test_native_joint_bout_stream_v1.py',ROOT/'tests/test_roam_native_continuous_v1.py'):
        code[p.relative_to(ROOT).as_posix()]=sha(p)
    artifacts={OLD.relative_to(ROOT).as_posix():sha(OLD),
        'benchmarks/new_bank_v3/roam_native_joint_v1/source_bank.pkl':
            old['artifacts_sha256']['benchmarks/new_bank_v3/roam_native_joint_v1/source_bank.pkl']}
    for s in old['subjects']:
        for n in [s['personal_path'],*[b['session_path'] for b in s['budgets'].values() if b['session_path']]]:
            artifacts[n]=old['artifacts_sha256'][n]
    recordings=[]
    with zipfile.ZipFile(ARCHIVE) as z:
        for u in range(19,29):
            for c in ('unsupported','reaching'):
                d=load_roam_cued_intervals(z,f'data/ROAM_EMG/s{u}/s{u}_static_{c}.csv')
                if d.receipt['excluded_intervals']:raise ValueError('Incomplete continuous source coverage')
                recordings.append(d.receipt)
    if sha(ARCHIVE)!=ARCHIVE_SHA:raise ValueError('Archive differs')
    write(PROTOCOL,dict(schema='roam_native_continuous_v1',source_sha256=code,artifact_sha256=artifacts,
        archive=str(ARCHIVE),archive_sha256=ARCHIVE_SHA,native_recordings=recordings,
        validation_users=list(range(19,24)),descriptive_users=list(range(24,29)),current_shots=[0,1,2],
        arms=ARMS,chunk_samples=[17,251,64,389],sample_rate_hz=200.,
        detector='Reuse saved personal detector at0 current shots, saved current detector otherwise. Fixed25ms RMS,80ms onset,120ms release,100ms preroll,1..30s. No EOF flush or query fitting.',
        reference='All close/open contiguous gt cue intervals; relax intervals remain stream input and may produce unmatched detections. Same100 unique active references at every budget.',
        matching='Within each recording, greedy descending IoU>=.5; ties event index then reference index; one-to-one; independent of predicted gesture. Unmatched detections and every missed reference retained.',
        calibration_cost='Every arm uses shared6 long plus0/3/6 current native cues and same selected detector. Population classifier does not make the detector/calibration bundle zero-cost.',
        primary='2 current cues/class on validation19..23: joint_full vs source_window higher end-to-end success, nonworse both active recalls and active event F1, lower matched log loss/Brier, at least3/5 user success wins; shared detector/matches. Empty matches fail probability guards.',
        classifier_refitted=False,detector_refitted=False,scope=SCOPE,default_promoted=False))


def run():
    if RESULT.exists() or OUT.exists():raise FileExistsError('Started/completed native experiment; do not rerun')
    p=json.loads(PROTOCOL.read_text(encoding='utf8'))
    for rel,h in {**p['source_sha256'],**p['artifact_sha256'],PROTOCOL.relative_to(ROOT).as_posix():sha(PROTOCOL)}.items():
        path=ROOT/rel
        if sha(path)!=h or subprocess.check_output(['git','show','HEAD:'+path.relative_to(REPO).as_posix()],cwd=REPO)!=path.read_bytes():
            raise ValueError('Commit exact frozen code/inputs before native inference: '+rel)
    if sha(ARCHIVE)!=p['archive_sha256']:raise ValueError('Archive differs')
    OUT.mkdir();old=json.loads(OLD.read_text(encoding='utf8'));bank=pickle.loads((HERE/'roam_native_joint_v1/source_bank.pkl').read_bytes())
    w=build(bank);source_records=tuple(t.rsplit(':cue',1)[0] for t in bank.policy_.source_trials)
    frozen=pickle.dumps(w);records=[];cells=[];subjects=[]
    with zipfile.ZipFile(ARCHIVE) as z:
        for s in old['subjects']:
            u=s['user'];user=f'ROAM_s{u}';personal=w.load_profile(ROOT/s['personal_path'],user_id=user)
            for shots in p['current_shots']:
                sp=s['budgets'][str(shots)]['session_path']
                session=None if sp is None else w.load_profile(ROOT/sp,user_id=user,session_id='hanging',personal=personal)
                before=pickle.dumps((personal,session));block=[]
                for condition in ('unsupported','reaching'):
                    member=f'data/ROAM_EMG/s{u}/s{u}_static_{condition}.csv';d=load_roam_cued_intervals(z,member)
                    x=np.concatenate(d.batch.sequences)
                    if (d.batch.starts[0]!=0 or len(x)!=d.receipt['native_samples']
                            or any(a+len(v)!=b for a,v,b in zip(d.batch.starts,d.batch.sequences,d.batch.starts[1:]))):
                        raise ValueError('Native continuous reconstruction has gaps')
                    stream=NativeJointBoutStreamV1(w,personal=personal,session=session,user_id=user,session_id='hanging',
                        source_recording_ids=source_records,sample_rate_hz=200.,channel_ids=CHANNELS,preprocessing_id=PREPROCESSING)
                    events=[];pos=0;j=0
                    while pos<len(x):
                        n=p['chunk_samples'][j%len(p['chunk_samples'])];end=min(pos+n,len(x))
                        events.extend(stream.feed(x[pos:end],pos,recording_id=member));pos=end;j+=1
                    censored=stream.finish()
                    # Evaluation labels and cue edges first enter here, after inference.
                    refs=[r for r in d.receipt['cue_intervals'] if r['class_name'] in ACTIVE]
                    row=dict(user=u,shots=shots,member=member,member_sha256=d.receipt['member_sha256'],native_samples=len(x),
                        references=refs,events=events,matches=matching(events,refs),censored_end=censored,
                        personal_path=s['personal_path'],session_path=sp,
                        detector_profile_id=(session or personal).profile_id,
                        unique_calibration_trials=6+3*shots)
                    records.append(row);block.append(row)
                if pickle.dumps((personal,session))!=before:raise ValueError('Immutable profiles changed')
                cells.extend(dict(user=u,shots=shots,arm=arm,metrics=metrics(block,arm),unique_calibration_trials=6+3*shots) for arm in ARMS)
            print(f'Native continuous user {u-18}/10 complete',flush=True)
    if pickle.dumps(w)!=frozen:raise ValueError('Frozen source workflow changed')
    aggregates=[]
    for phase,users in (('validation',range(19,24)),('descriptive_final',range(24,29)),('all',range(19,29))):
        for shots in p['current_shots']:
            block=[r for r in records if r['user'] in users and r['shots']==shots]
            aggregates.extend(dict(phase=phase,shots=shots,arm=a,metrics=metrics(block,a)) for a in ARMS)
    lookup={(r['phase'],r['shots'],r['arm']):r['metrics'] for r in aggregates}
    base=lookup['validation',2,'source_window'];full=lookup['validation',2,'joint_full']
    users={(r['user'],r['shots'],r['arm']):r['metrics'] for r in cells}
    wins=sum(users[u,2,'joint_full']['end_to_end_success']>users[u,2,'source_window']['end_to_end_success'] for u in range(19,24))
    guards=dict(higher_end_to_end_success=full['end_to_end_success']>base['end_to_end_success'],
        nonworse_active_recalls=all(full['active_recall'][c]>=base['active_recall'][c] for c in ACTIVE),
        nonworse_active_event_f1=full['active_event_macro_f1']>=base['active_event_macro_f1'],
        lower_conditional_log_loss=full['conditional_log_loss'] is not None and full['conditional_log_loss']<base['conditional_log_loss'],
        lower_conditional_brier=full['conditional_brier'] is not None and full['conditional_brier']<base['conditional_brier'],at_least3_user_success_wins=wins>=3)
    write(OUT/'recordings.json',records)
    with (OUT/'events.csv').open('x',encoding='utf8',newline='') as f:
        fields=('user','shots','member','event_id','start','end','reference_id','reference_label','arm','predicted','probabilities_json')
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in records:
            by_event={m['event_index']:r['references'][m['reference_index']] for m in r['matches']}
            for i,e in enumerate(r['events']):
                ref=by_event.get(i)
                for a in ARMS:writer.writerow(dict(user=r['user'],shots=r['shots'],member=r['member'],event_id=e['trial_id'],start=e['start'],end=e['end'],
                    reference_id='' if ref is None else ref['trial_id'],reference_label='' if ref is None else ref['class_name'],arm=a,predicted=e['labels'][a],probabilities_json=json.dumps(e['probabilities'][a],separators=(',',':'))))
    artifacts={v.relative_to(ROOT).as_posix():sha(v) for v in OUT.iterdir()}
    write(RESULT,dict(schema='roam_native_continuous_v1',protocol_sha256=sha(PROTOCOL),artifacts_sha256=artifacts,
        recording_budget_blocks=len(records),distinct_query_recordings=20,independent_active_references=100,
        cells=cells,aggregates=aggregates,primary_guards=guards,primary_pass=all(guards.values()),validation_user_success_wins=wins,
        classifier_refitted=False,detector_refitted=False,source_profiles_unchanged=True,
        scope=SCOPE,default_promoted=False,physical_validation_proven=False,completion_proven=False))
    print(json.dumps(dict(primary_pass=all(guards.values()),validation_user_success_wins=wins)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    prepare() if args.prepare else run()
