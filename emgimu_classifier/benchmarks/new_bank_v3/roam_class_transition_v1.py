"""Source-selected class-aware native segmentation; frozen source/target phases."""
import argparse
import itertools
import json
import pickle
import subprocess
import zipfile
from pathlib import Path
import numpy as np
from emgimu.datasets.roam_cued_intervals_v1 import CHANNELS,CLASSES,PREPROCESSING,load_roam_cued_intervals
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.class_transition_bouts_v1 import ConfirmedClassBoutDetectorV1
from emgimu.feature_bank.native_class_transition_stream_v1 import window_scores
from emgimu.feature_bank.native_bout_window_adapter_v2 import native_windows
from emgimu.feature_bank.personal_temporal_bouts_v1 import TemporalBoutBatchV1
from benchmarks.new_bank_v3.roam_native_joint_v1 import ROOT,HERE,REPOSITORY,ARCHIVE,ARCHIVE_SHA,sha,write,build
from benchmarks.new_bank_v3.roam_native_continuous_v1 import matching,metrics,ARMS,ACTIVE
from benchmarks.new_bank_v3.roam_source_fusion_v1 import source_data

PROTOCOL=HERE/'ROAM_CLASS_TRANSITION_V1_PROTOCOL.json'
SOURCE=HERE/'ROAM_CLASS_TRANSITION_V1_SOURCE_RESULTS.json'
TARGET=HERE/'ROAM_CLASS_TRANSITION_V1_TARGET_RESULTS.json'
OUT=HERE/'roam_class_transition_v1'
INPUT=HERE/'ROAM_SOURCE_FUSION_V1_SOURCE_INPUT.json'
SOURCE_RECEIPT=ROOT/'feature_bank/ROAM_CLASS_TRANSITION_V1_SOURCE_ACCEPTANCE.json'
TARGET_RECEIPT=ROOT/'feature_bank/ROAM_CLASS_TRANSITION_V1_ACCEPTANCE.json'
SCOPE=('Retrospective native8/200Hz ROAM source-selected class-transition segmentation, including active-to-active '
       'changes, followed by frozen window/joint classification. Source grid selection uses18 users/72 recordings, '
       'held out from each source representation fold, with360 independent active cue references. Existing source '
       'OOF temperatures/population used all source users; source grid scores are policy training, not unbiased '
       'generalization. Source policy is committed before target inference. Previously inspected19..28 users '
       'and active gt cue references do not establish blind/prospective or physiological boundary truth, '
       'pinch, physical ring/F6/quality,250Hz device efficacy or a universal/default benefit.')


def grid():
    return [dict(confirmations=n,smoothing_windows=s,confidence=c)
        for n,s,c in itertools.product((1,3,5),(1,5),(0.,.5,.7))]


def stream_input(data):
    b=data.batch;x=np.concatenate(b.sequences)
    if (b.starts[0]!=0 or len(x)!=data.receipt['native_samples'] or data.receipt['excluded_intervals']
            or any(a+len(v)!=e for a,v,e in zip(b.starts,b.sequences,b.starts[1:]))):
        raise ValueError('Every native query sample must be retained in order')
    ends=np.arange(40,len(x)+1,8,dtype=np.int64)
    windows=np.stack([x[e-40:e] for e in ends])
    refs=[r for r in data.receipt['cue_intervals'] if r['class_name'] in ACTIVE]
    return x,windows,ends,refs


def detect(x,ends,q,bank_id,member,config):
    d=ConfirmedClassBoutDetectorV1(bank_id=bank_id,class_names=CLASSES,rest_label='relax',sample_rate_hz=200.,**config)
    events=d.feed(x,0,recording_id=member,probabilities=q,window_end_indices=ends,bank_id=bank_id)
    discarded=d.discarded;censored=d.finish()
    return events,censored,discarded


def source_metric(records):
    refs=sum(len(r['references']) for r in records);events=sum(len(r['events']) for r in records)
    matched=sum(len(r['matches']) for r in records);correct=0;f1=[]
    for c in ACTIVE:
        truth=sum(v['class_name']==c for r in records for v in r['references'])
        pred=sum(e['estimated_class']==c for r in records for e in r['events'])
        tp=sum(r['references'][m['reference_index']]['class_name']==c==r['events'][m['event_index']]['estimated_class']
            for r in records for m in r['matches'])
        correct+=tp;f1.append(2*tp/(truth+pred) if truth+pred else 0.)
    return dict(references=refs,detections=events,matched=matched,unmatched_detections=events-matched,
        detection_f1=2*matched/(refs+events) if refs+events else 0.,
        stable_class_proxy_macro_f1=float(np.mean(f1)),stable_class_proxy_correct=correct)


def prepare():
    parent=json.loads((HERE/'ROAM_NATIVE_CONTINUOUS_V1_PROTOCOL.json').read_text(encoding='utf8'))
    source=dict(parent['source_sha256'])
    for path in (Path(__file__),HERE/'verify_roam_class_transition_v1.py',
        ROOT/'src/emgimu/feature_bank/class_transition_bouts_v1.py',ROOT/'src/emgimu/feature_bank/native_class_transition_stream_v1.py',
        ROOT/'tests/test_class_transition_bouts_v1.py',ROOT/'tests/test_native_class_transition_stream_v1.py',
        ROOT/'tests/test_roam_class_transition_study_v1.py',HERE/'verify_roam_native_continuous_v2.py'):
        source[path.relative_to(ROOT).as_posix()]=sha(path)
    artifacts={INPUT.relative_to(ROOT).as_posix():sha(INPUT)}
    for i in range(3):
        path=HERE/f'roam_source_fusion_v1/source/bank_fold{i}.pkl';artifacts[path.relative_to(ROOT).as_posix()]=sha(path)
    target=dict(parent['artifact_sha256'])
    for path in (HERE/'ROAM_NATIVE_CONTINUOUS_V1_RESULTS.json',HERE/'roam_native_continuous_v1/recordings.json'):
        target[path.relative_to(ROOT).as_posix()]=sha(path)
    data=source_data()
    assert sha(ARCHIVE)==ARCHIVE_SHA
    write(PROTOCOL,dict(schema='roam_class_transition_v1',source_sha256=source,source_artifact_sha256=artifacts,
        target_artifact_sha256=target,archive_sha256=ARCHIVE_SHA,native_source_recordings=[d.receipt for d in data.values()],
        source_users=list(range(1,19)),validation_users=list(range(19,24)),descriptive_users=list(range(24,29)),
        source_candidates=grid(),source_selection='Maximize equal-user detection F1 on source cue references; tie by equal-user stable-class proxy event F1, fewer unmatched detections, then original candidate order. These are training scores, not an unbiased source evaluation.',
        source_probability='Reuse three frozen source-user-out seven-provider banks, original source-OOF cue temperatures/population, one native200ms window per classifier readout every40ms. No representation/classifier/temperature refitting or target labels.',
        boundary='Confirm source class changes after1/3/5 windows; optional trailing1/5 probability mean; confidence0/.5/.7, otherwise Unknown. Boundary is center of first window in confirmed run. Active-to-active changes close/open a segment without Rest. Unknown, unobserved initial start, invalid1..30s duration and EOF are censored; no query truth enters inference.',
        target='One source-selected configuration only. Same20 complete unsupported/reaching recordings and100 active references as prior native rest detector, with fixed greedy IoU>=.5. Class-aware detector output is identical across current calibration budgets0/1/2; saved target profiles affect classification only.',
        costs='Detector and source_window arm require zero target calibration. window_full/joint_full require saved6 long plus0/3/6 current cues. Experimental computation of another arm does not charge its independent source-only control.',
        primary='At2 shots/class on validation19..23, new joint_full vs old neutral-detector joint_full: higher fixed-reference end-to-end success, higher detection recall, nonworse active event F1 and both active recalls, no more unmatched detections and at least3/5 user success wins. Conditional probability losses are descriptive across different matches, never paired detector-improvement criteria.',
        secondary='Within new fixed matched subsets, joint_full vs source_window: lower loss/Brier, nonworse end-to-end success/F1/both recalls, at least3/5 user loss wins. Independent of detector primary; no default promotion.',
        scope=SCOPE,default_promoted=False,completion_proven=False))


def check(target=False):
    p=json.loads(PROTOCOL.read_text(encoding='utf8'))
    required={**p['source_sha256'],**p['source_artifact_sha256'],PROTOCOL.relative_to(ROOT).as_posix():sha(PROTOCOL)}
    if target:
        required.update(p['target_artifact_sha256'])
        required.update({q.relative_to(ROOT).as_posix():sha(q) for q in (SOURCE,SOURCE_RECEIPT,OUT/'source/policy.json')})
    for rel,h in required.items():
        path=ROOT/rel
        if sha(path)!=h or subprocess.check_output(['git','show','HEAD:'+path.relative_to(REPOSITORY).as_posix()],cwd=REPOSITORY)!=path.read_bytes():
            raise ValueError('Commit exact protocol/source/policy before native work: '+rel)
    assert sha(ARCHIVE)==p['archive_sha256']
    return p


def source_run():
    if OUT.exists() or SOURCE.exists():raise FileExistsError('Source run started/completed; do not rerun')
    p=check();data=source_data();inputs=json.loads(INPUT.read_text(encoding='utf8'))
    assert json.loads(json.dumps([d.receipt for d in data.values()]))==p['native_source_recordings']
    folder=OUT/'source';folder.mkdir(parents=True);banks={};before={};arrays={};rows=[]
    for i in range(3):
        banks[i]=pickle.loads((HERE/f'roam_source_fusion_v1/source/bank_fold{i}.pkl').read_bytes());before[i]=pickle.dumps(banks[i])
    for (u,condition),data_item in data.items():
        fold=next(i for i,f in enumerate(inputs['source_folds']) if u in f['held_users']);bank=banks[fold]
        if any(f'/s{u}/' in t for t in bank.policy_.source_trials):raise ValueError('Source query user entered representation fit')
        x,windows,ends,refs=stream_input(data_item);member=data_item.receipt['member'];key=f'u{u}_{condition}'
        q=window_scores(bank,windows,ends,recording_id=member,user_id=f'ROAM_source_s{u}')
        arrays[key+'_q']=q;arrays[key+'_ends']=ends
        rows.append(dict(user=u,condition=condition,key=key,member=member,fold=fold,bank_id=bank.bank_id_,references=refs,native_samples=len(x)))
        if condition=='reaching' and u%3==0:print(f'Source native windows {u}/18 users',flush=True)
    candidates=[]
    for index,config in enumerate(p['source_candidates']):
        evaluated=[]
        for row in rows:
            x=np.concatenate(data[row['user'],row['condition']].batch.sequences);key=row['key']
            events,censored,discarded=detect(x,arrays[key+'_ends'],arrays[key+'_q'],row['bank_id'],row['member'],config)
            evaluated.append(dict(user=row['user'],references=row['references'],events=events,matches=matching(events,row['references'])))
        per_user=[dict(user=u,**source_metric([r for r in evaluated if r['user']==u])) for u in p['source_users']]
        candidates.append(dict(index=index,config=config,per_user=per_user,
            mean_detection_f1=float(np.mean([r['detection_f1'] for r in per_user])),
            mean_stable_class_proxy_macro_f1=float(np.mean([r['stable_class_proxy_macro_f1'] for r in per_user])),
            total_unmatched_detections=sum(r['unmatched_detections'] for r in per_user)))
    selected=min(candidates,key=lambda c:(-c['mean_detection_f1'],-c['mean_stable_class_proxy_macro_f1'],c['total_unmatched_detections'],c['index']))
    assert all(pickle.dumps(banks[i])==before[i] for i in banks)
    np.savez_compressed(folder/'window_scores.npz',**arrays)
    policy=dict(schema='native_class_transition_policy_v1',source_protocol_sha256=sha(PROTOCOL),inference_bank_id=inputs['inference_bank_id'],
        config=selected['config'],selected_candidate=selected['index'],sample_rate_hz=200.,classes=CLASSES,
        source_users=p['source_users'],source_query_recordings=[r['member'] for r in rows],
        source_probability_cache_sha256=sha(folder/'window_scores.npz'),training_scores_are_unbiased=False)
    write(folder/'policy.json',policy)
    write(SOURCE,dict(schema='roam_class_transition_v1_source',protocol_sha256=sha(PROTOCOL),source_recordings=rows,
        source_users=18,native_recordings=72,independent_active_references=360,candidates=candidates,
        selected_candidate=selected['index'],selected_config=selected['config'],existing_classifiers_refitted=False,
        target_arrays_read=False,training_scores_are_unbiased=False,
        artifacts_sha256={v.relative_to(ROOT).as_posix():sha(v) for v in folder.iterdir()},scope=SCOPE))
    print(json.dumps(dict(selected_config=selected['config'],source_training_detection_f1=selected['mean_detection_f1'])),flush=True)


def target_run():
    folder=OUT/'target'
    if folder.exists() or TARGET.exists():raise FileExistsError('Target run started/completed; do not rerun')
    p=check(target=True);policy=json.loads((OUT/'source/policy.json').read_text(encoding='utf8'))
    old=json.loads((HERE/'ROAM_NATIVE_JOINT_V1_RESULTS.json').read_text(encoding='utf8'))
    baseline=json.loads((HERE/'ROAM_NATIVE_CONTINUOUS_V1_RESULTS.json').read_text(encoding='utf8'))
    bank=pickle.loads((HERE/'roam_native_joint_v1/source_bank.pkl').read_bytes());w=build(bank);before=pickle.dumps(w)
    assert policy['inference_bank_id']==bank.bank_id_ and policy['source_protocol_sha256']==sha(PROTOCOL)
    folder.mkdir();records=[];cells=[]
    with zipfile.ZipFile(ARCHIVE) as z:
        for subject in old['subjects']:
            u=subject['user'];user=f'ROAM_s{u}';personal=w.load_profile(ROOT/subject['personal_path'],user_id=user)
            detected=[]
            for condition in ('unsupported','reaching'):
                member=f'data/ROAM_EMG/s{u}/s{u}_static_{condition}.csv';data=load_roam_cued_intervals(z,member)
                x,windows,ends,refs=stream_input(data);q=window_scores(bank,windows,ends,recording_id=member,user_id=user)
                events,censored,discarded=detect(x,ends,q,bank.bank_id_,member,policy['config'])
                batch=None;source_q=np.empty((0,3))
                if events:
                    batch=TemporalBoutBatchV1(tuple(e['emg'] for e in events),tuple(e['trial_id'] for e in events),(member,)*len(events),
                        tuple(e['start'] for e in events),200.,CHANNELS,PREPROCESSING,'estimated')
                    b,ids,offsets,_=native_windows(batch,40,8);read=bank.predict(b,ids,window_offsets=offsets,user_id=user)
                    source_q=read['probabilities'][[read['trial_ids'].index(t) for t in batch.trial_ids]]
                detected.append((member,data.receipt,refs,events,censored,discarded,batch,source_q,q,ends))
            for shots in (0,1,2):
                sp=subject['budgets'][str(shots)]['session_path']
                session=None if sp is None else w.load_profile(ROOT/sp,user_id=user,session_id='hanging',personal=personal)
                immutable=pickle.dumps((personal,session));block=[]
                for member,receipt,refs,events,censored,discarded,batch,source_q,q,ends in detected:
                    probabilities=dict(source_window=source_q,window_full=np.empty((0,3)),joint_full=np.empty((0,3)))
                    if batch is not None:
                        result=w.predict(batch,personal=personal,session=session,user_id=user,session_id='hanging')
                        probabilities.update(window_full=result['temporal']['arms']['base'],joint_full=result['probabilities'])
                    saved=[]
                    for i,e in enumerate(events):
                        entry={k:v for k,v in e.items() if k!='emg'}
                        entry.update(probabilities={a:probabilities[a][i].tolist() for a in ARMS},
                            labels={a:CLASSES[int(probabilities[a][i].argmax())] for a in ARMS})
                        saved.append(entry)
                    row=dict(user=u,shots=shots,member=member,member_sha256=receipt['member_sha256'],native_samples=receipt['native_samples'],
                        references=refs,events=saved,matches=matching(saved,refs),censored_end=censored,discarded_candidates=discarded,
                        personal_path=subject['personal_path'],session_path=sp,detector_target_calibration_trials=0)
                    records.append(row);block.append(row)
                assert pickle.dumps((personal,session))==immutable
                cells.extend(dict(user=u,shots=shots,arm=a,metrics=metrics(block,a),
                    target_calibration_trials=0 if a=='source_window' else 6+3*shots) for a in ARMS)
            if u%2==0:print(f'Target native transitions {u-18}/10 users',flush=True)
    assert pickle.dumps(w)==before
    aggregates=[]
    for phase,users in (('validation',range(19,24)),('descriptive_final',range(24,29)),('all',range(19,29))):
        for shots in (0,1,2):
            part=[r for r in records if r['user'] in users and r['shots']==shots]
            aggregates.extend(dict(phase=phase,shots=shots,arm=a,metrics=metrics(part,a)) for a in ARMS)
    guard,secondary,wins,loss_wins=guards(aggregates,cells,baseline)
    write(folder/'recordings.json',records)
    write(TARGET,dict(schema='roam_class_transition_v1_target',protocol_sha256=sha(PROTOCOL),source_result_sha256=sha(SOURCE),
        source_acceptance_sha256=sha(SOURCE_RECEIPT),policy_sha256=sha(OUT/'source/policy.json'),selected_config=policy['config'],
        independent_active_references=100,recording_budget_blocks=60,cells=cells,aggregates=aggregates,
        detector_primary_guards=guard,detector_primary_pass=all(guard.values()),validation_user_success_wins=wins,
        within_detector_joint_guards=secondary,within_detector_joint_pass=all(secondary.values()),validation_user_loss_wins=loss_wins,
        artifacts_sha256={v.relative_to(ROOT).as_posix():sha(v) for v in folder.iterdir()},
        source_policy_committed_before_target_inference=True,target_fitted_or_selected=False,
        default_promoted=False,physical_validation_proven=False,completion_proven=False,scope=SCOPE))
    print(json.dumps(dict(detector_primary_pass=all(guard.values()),within_detector_joint_pass=all(secondary.values()),selected_config=policy['config'])),flush=True)


def guards(aggregates,cells,baseline):
    lookup={(r['phase'],r['shots'],r['arm']):r['metrics'] for r in aggregates}
    previous={(r['phase'],r['shots'],r['arm']):r['metrics'] for r in baseline['aggregates']}
    by_user={(r['user'],r['shots'],r['arm']):r['metrics'] for r in cells}
    old_user={(r['user'],r['shots'],r['arm']):r['metrics'] for r in baseline['cells']}
    full=lookup['validation',2,'joint_full'];old=previous['validation',2,'joint_full'];source=lookup['validation',2,'source_window']
    wins=sum(by_user[u,2,'joint_full']['correct']>old_user[u,2,'joint_full']['correct'] for u in range(19,24))
    loss_wins=sum(by_user[u,2,'joint_full']['conditional_log_loss'] is not None and
        by_user[u,2,'joint_full']['conditional_log_loss']<by_user[u,2,'source_window']['conditional_log_loss'] for u in range(19,24))
    primary=dict(higher_end_to_end_success=full['correct']>old['correct'],higher_detection_recall=full['matched']>old['matched'],
        nonworse_active_f1=full['active_event_macro_f1']>=old['active_event_macro_f1'],
        nonworse_both_active_recalls=all(full['active_recall'][c]>=old['active_recall'][c] for c in ACTIVE),
        no_more_unmatched_detections=full['unmatched_detections']<=old['unmatched_detections'],at_least3_user_success_wins=wins>=3)
    secondary=dict(lower_log_loss=full['conditional_log_loss'] is not None and full['conditional_log_loss']<source['conditional_log_loss'],
        lower_brier=full['conditional_brier'] is not None and full['conditional_brier']<source['conditional_brier'],
        nonworse_end_to_end_success=full['correct']>=source['correct'],nonworse_active_f1=full['active_event_macro_f1']>=source['active_event_macro_f1'],
        nonworse_both_active_recalls=all(full['active_recall'][c]>=source['active_recall'][c] for c in ACTIVE),at_least3_user_loss_wins=loss_wins>=3)
    return primary,secondary,wins,loss_wins


if __name__=='__main__':
    parser=argparse.ArgumentParser();m=parser.add_mutually_exclusive_group(required=True)
    m.add_argument('--prepare',action='store_true');m.add_argument('--source',action='store_true');m.add_argument('--target',action='store_true');args=parser.parse_args()
    prepare() if args.prepare else source_run() if args.source else target_run()
