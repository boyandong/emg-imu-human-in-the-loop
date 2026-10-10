"""Source-only precision-biased transition selection with fixed recall retention.

New versioned policy experiment; it reuses immutable verified source-window
probabilities. No source classifier retraining or repetition of the prior run.
Target users were previously inspected; this is a retrospective diagnostic.
"""
import argparse
import itertools
import json
import pickle
import subprocess
import zipfile
from pathlib import Path
import numpy as np
from emgimu.datasets.roam_cued_intervals_v1 import CHANNELS,CLASSES,PREPROCESSING,load_roam_cued_intervals
from emgimu.feature_bank.native_class_transition_stream_v1 import window_scores
from emgimu.feature_bank.native_bout_window_adapter_v2 import native_windows
from emgimu.feature_bank.personal_temporal_bouts_v1 import TemporalBoutBatchV1
from benchmarks.new_bank_v3.roam_native_joint_v1 import ROOT,HERE,REPOSITORY,ARCHIVE,ARCHIVE_SHA,sha,write,build
from benchmarks.new_bank_v3.roam_class_transition_v1 import detect,stream_input,source_metric
from benchmarks.new_bank_v3.roam_native_continuous_v1 import matching,metrics,ARMS,ACTIVE

PROTOCOL=HERE/'ROAM_PRECISION_TRANSITION_V2_PROTOCOL.json'
SOURCE=HERE/'ROAM_PRECISION_TRANSITION_V2_SOURCE_RESULTS.json'
TARGET=HERE/'ROAM_PRECISION_TRANSITION_V2_TARGET_RESULTS.json'
OUT=HERE/'roam_precision_transition_v2'
PREVIOUS_SOURCE=HERE/'ROAM_CLASS_TRANSITION_V1_SOURCE_RESULTS.json'
CACHE=HERE/'roam_class_transition_v1/source/window_scores.npz'
SOURCE_RECEIPT=ROOT/'feature_bank/ROAM_PRECISION_TRANSITION_V2_SOURCE_ACCEPTANCE.json'
TARGET_RECEIPT=ROOT/'feature_bank/ROAM_PRECISION_TRANSITION_V2_ACCEPTANCE.json'
SCOPE=('New source-only precision-biased class-transition policy on previously inspected native8/200Hz ROAM. '
       'Reuse verified frozen source per-window probabilities, without source refits or native source inference. '
       'Source18 users/72 recordings/360 active cue references select36 precommitted settings. '
       'Source policy selection is training, not unbiased evaluation. Target20 recordings/100 active cue references '
       'remain retrospective, not physiological boundaries, independent prospective validation, pinch,250Hz '
       'own-device, physical ring/F6/F9 or a universal/default benefit.')


def grid():
    return [dict(confirmations=n,smoothing_windows=s,confidence=c)
        for n,s,c in itertools.product((5,10,15,20),(1,5,10),(0.,.5,.7))]


def summarize(index,config,users):
    return dict(index=index,config=config,per_user=users,
        mean_detection_fbeta_half=float(np.mean([1.25*u['matched']/(.25*u['references']+u['detections']) for u in users])),
        mean_detection_recall=float(np.mean([u['matched']/u['references'] for u in users])),
        total_unmatched_detections=sum(u['unmatched_detections'] for u in users))


def select(candidates,baseline):
    # Precision emphasis is fixed before this grid is evaluated; retain >=90%
    # of the old source mean recall and no more old source extra detections.
    eligible=[c for c in candidates if c['mean_detection_recall']>=.9*baseline['mean_detection_recall']
        and c['total_unmatched_detections']<=baseline['total_unmatched_detections']]
    if not eligible:raise ValueError('The declared baseline must remain eligible')
    return min(eligible,key=lambda c:(-c['mean_detection_fbeta_half'],-c['mean_detection_recall'],c['total_unmatched_detections'],c['index']))


def prepare():
    parent=json.loads((HERE/'ROAM_CLASS_TRANSITION_V1_PROTOCOL.json').read_text(encoding='utf8'))
    source=dict(parent['source_sha256'])
    for path in (Path(__file__),HERE/'verify_roam_precision_transition_v2.py',ROOT/'tests/test_roam_precision_transition_v2.py'):
        source[path.relative_to(ROOT).as_posix()]=sha(path)
    artifacts=dict(parent['source_artifact_sha256'])
    for path in (PREVIOUS_SOURCE,CACHE,ROOT/'feature_bank/ROAM_CLASS_TRANSITION_V1_SOURCE_ACCEPTANCE.json'):
        artifacts[path.relative_to(ROOT).as_posix()]=sha(path)
    target=dict(parent['target_artifact_sha256'])
    target_path=HERE/'ROAM_CLASS_TRANSITION_V1_TARGET_RESULTS.json'
    target[target_path.relative_to(ROOT).as_posix()]=sha(target_path)
    write(PROTOCOL,dict(schema='roam_precision_transition_v2',source_sha256=source,source_artifact_sha256=artifacts,
        target_artifact_sha256=target,archive_sha256=ARCHIVE_SHA,
        source_users=list(range(1,19)),validation_users=list(range(19,24)),descriptive_users=list(range(24,29)),
        source_candidates=grid(),source_selection='Among source candidates retaining >=90% old mean source detection recall and no more old source unmatched detections, maximize equal-user detection F0.5; tie by higher mean recall, fewer unmatched, original index. Old source configuration is included. Scores are policy training.',
        source_probability='Reuse exact SHA-bound144 cached arrays from the independently verified previous source phase. No native source-window computation or classifier refitting. Inert zero sample placeholders satisfy the unchanged probability-only FSM capture API; placeholder samples are never features, output artifacts or predictive claims.',
        boundary='Unchanged causal confirmed-class FSM at200Hz,200ms windows/40ms hop. Confirmation5/10/15/20, trailing mean1/5/10, confidence0/.5/.7. Labels and cue edges never enter inference. Unknown/unobserved start/1..30s/EOF censoring remains unchanged. Longer confirmation delays algorithmic availability; no physical latency claim.',
        target='Exactly one source-selected configuration after independent source verification and Git commit. Same20 whole recordings/100 active references and saved personal/session profiles. Three arms,0/1/2 current shots. No target parameter search or refitting.',
        primary='Validation19..23 at2 shots/class joint_full against previous class-transition V1: strictly fewer unmatched detections, nonworse correct/matched/active event F1/both active recalls and at least3 users with fewer unmatched and nonworse correct. Conditional losses across detectors are not paired.',
        secondary='Within identical new matches joint_full against zero-target-calibration source_window: lower loss/Brier, nonworse success/F1/both active recalls and at least3/5 user loss wins.',
        costs='Detector/source_window0 target cues; personal/window and joint6 long+3*shots current cues. No hidden reference-edge or calibration input to detector.',
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
            raise ValueError('Commit exact protocol/source/policy before new native work: '+rel)
    return p


def source_run():
    if OUT.exists() or SOURCE.exists():raise FileExistsError('New source policy phase started/completed; do not rerun')
    p=check();previous=json.loads(PREVIOUS_SOURCE.read_text(encoding='utf8'))
    folder=OUT/'source';folder.mkdir(parents=True);rows=previous['source_recordings']
    cache=np.load(CACHE,allow_pickle=False);candidates=[]
    old=previous['candidates'][previous['selected_candidate']]
    baseline=summarize(-1,old['config'],old['per_user'])
    for index,config in enumerate(p['source_candidates']):
        evaluated=[]
        for row in rows:
            key=row['key'];placeholders=np.zeros((row['native_samples'],8),np.float32)
            events,_,_=detect(placeholders,cache[key+'_ends'],cache[key+'_q'],row['bank_id'],row['member'],config)
            evaluated.append(dict(user=row['user'],references=row['references'],events=events,matches=matching(events,row['references'])))
        users=[dict(user=u,**source_metric([r for r in evaluated if r['user']==u])) for u in p['source_users']]
        candidates.append(summarize(index,config,users))
        if (index+1)%6==0:print(f'Source-only cached policy grid {index+1}/{len(p["source_candidates"])}',flush=True)
    selected=select(candidates,baseline)
    policy=dict(schema='native_class_transition_policy_v1',source_protocol_sha256=sha(PROTOCOL),
        inference_bank_id=json.loads((HERE/'ROAM_SOURCE_FUSION_V1_SOURCE_INPUT.json').read_text())['inference_bank_id'],
        config=selected['config'],selected_candidate=selected['index'],sample_rate_hz=200.,classes=CLASSES,rest_label='relax',
        source_users=p['source_users'],source_query_recordings=[r['member'] for r in rows],
        source_probability_cache_sha256=sha(CACHE),training_scores_are_unbiased=False)
    write(folder/'policy.json',policy)
    write(SOURCE,dict(schema='roam_precision_transition_v2_source',protocol_sha256=sha(PROTOCOL),
        source_recordings=rows,source_users=18,native_recordings=72,independent_active_references=360,
        candidates=candidates,baseline=baseline,selected_candidate=selected['index'],selected_config=selected['config'],
        existing_classifiers_refitted=False,native_source_inference_repeated=False,target_arrays_read=False,training_scores_are_unbiased=False,
        artifacts_sha256={v.relative_to(ROOT).as_posix():sha(v) for v in folder.iterdir()},scope=SCOPE))
    print(json.dumps(dict(selected_config=selected['config'],source_training_fbeta_half=selected['mean_detection_fbeta_half'])),flush=True)


def target_run():
    folder=OUT/'target'
    if folder.exists() or TARGET.exists():raise FileExistsError('Target run started/completed; do not rerun')
    p=check(target=True);assert sha(ARCHIVE)==p['archive_sha256'];policy=json.loads((OUT/'source/policy.json').read_text(encoding='utf8'))
    old=json.loads((HERE/'ROAM_NATIVE_JOINT_V1_RESULTS.json').read_text(encoding='utf8'))
    baseline=json.loads((HERE/'ROAM_CLASS_TRANSITION_V1_TARGET_RESULTS.json').read_text(encoding='utf8'))
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
            if u%2==0:print(f'Target precision policy {u-18}/10 users',flush=True)
    assert pickle.dumps(w)==before
    aggregates=[]
    for phase,users in (('validation',range(19,24)),('descriptive_final',range(24,29)),('all',range(19,29))):
        for shots in (0,1,2):
            part=[r for r in records if r['user'] in users and r['shots']==shots]
            aggregates.extend(dict(phase=phase,shots=shots,arm=a,metrics=metrics(part,a)) for a in ARMS)
    guard,secondary,wins,loss_wins=guards(aggregates,cells,baseline)
    write(folder/'recordings.json',records)
    write(TARGET,dict(schema='roam_precision_transition_v2_target',protocol_sha256=sha(PROTOCOL),source_result_sha256=sha(SOURCE),
        source_acceptance_sha256=sha(SOURCE_RECEIPT),policy_sha256=sha(OUT/'source/policy.json'),selected_config=policy['config'],
        independent_active_references=100,recording_budget_blocks=60,cells=cells,aggregates=aggregates,
        detector_primary_guards=guard,detector_primary_pass=all(guard.values()),validation_user_false_event_wins=wins,
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
    wins=sum(by_user[u,2,'joint_full']['unmatched_detections']<old_user[u,2,'joint_full']['unmatched_detections']
        and by_user[u,2,'joint_full']['correct']>=old_user[u,2,'joint_full']['correct'] for u in range(19,24))
    loss_wins=sum(by_user[u,2,'joint_full']['conditional_log_loss'] is not None and
        by_user[u,2,'joint_full']['conditional_log_loss']<by_user[u,2,'source_window']['conditional_log_loss'] for u in range(19,24))
    primary=dict(fewer_unmatched_detections=full['unmatched_detections']<old['unmatched_detections'],
        nonworse_end_to_end_success=full['correct']>=old['correct'],nonworse_detection_recall=full['matched']>=old['matched'],
        nonworse_active_f1=full['active_event_macro_f1']>=old['active_event_macro_f1'],
        nonworse_both_active_recalls=all(full['active_recall'][c]>=old['active_recall'][c] for c in ACTIVE),at_least3_user_false_event_wins=wins>=3)
    secondary=dict(lower_log_loss=full['conditional_log_loss'] is not None and full['conditional_log_loss']<source['conditional_log_loss'],
        lower_brier=full['conditional_brier'] is not None and full['conditional_brier']<source['conditional_brier'],
        nonworse_end_to_end_success=full['correct']>=source['correct'],nonworse_active_f1=full['active_event_macro_f1']>=source['active_event_macro_f1'],
        nonworse_both_active_recalls=all(full['active_recall'][c]>=source['active_recall'][c] for c in ACTIVE),at_least3_user_loss_wins=loss_wins>=3)
    return primary,secondary,wins,loss_wins


if __name__=='__main__':
    parser=argparse.ArgumentParser();m=parser.add_mutually_exclusive_group(required=True)
    m.add_argument('--prepare',action='store_true');m.add_argument('--source',action='store_true');m.add_argument('--target',action='store_true');args=parser.parse_args()
    prepare() if args.prepare else source_run() if args.source else target_run()
