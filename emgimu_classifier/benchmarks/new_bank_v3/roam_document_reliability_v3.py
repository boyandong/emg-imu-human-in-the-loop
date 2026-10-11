"""Frozen native document D/E source-user CV and cached target composition."""
import argparse
import itertools
import json
import pickle
import subprocess
from pathlib import Path
import numpy as np
from emgimu.feature_bank.native_document_reliability_v3 import (
    NativeDocumentReliabilityV3,calibration_features,hierarchical_weights,policy_digest)
from emgimu.feature_bank.native_bout_window_adapter_v2 import native_windows
from benchmarks.new_bank_v3.roam_native_joint_v1 import ROOT,HERE,REPOSITORY,ARCHIVE,ARCHIVE_SHA,sha,write,build,merge,CLASSES,GROUPS
from benchmarks.new_bank_v3.roam_source_fusion_v1 import source_data,score

PROTOCOL=HERE/'ROAM_DOCUMENT_RELIABILITY_V3_PROTOCOL.json'
SOURCE=HERE/'ROAM_DOCUMENT_RELIABILITY_V3_SOURCE_RESULTS.json'
TARGET=HERE/'ROAM_DOCUMENT_RELIABILITY_V3_TARGET_RESULTS.json'
OUT=HERE/'roam_document_reliability_v3'
PREVIOUS=HERE/'ROAM_SOURCE_FUSION_V1_SOURCE_RESULTS.json'
INPUT=HERE/'ROAM_SOURCE_FUSION_V1_SOURCE_INPUT.json'
SOURCE_RECEIPT=ROOT/'feature_bank/ROAM_DOCUMENT_RELIABILITY_V3_SOURCE_ACCEPTANCE.json'
TARGET_RECEIPT=ROOT/'feature_bank/ROAM_DOCUMENT_RELIABILITY_V3_ACCEPTANCE.json'
ARMS=('source_window','legacy_reliability','legacy_window','legacy_joint','document_reliability','document_window','document_joint')
SCOPE=('Versioned native8/200Hz document-exact hierarchical D/E reliability with16-setting source-user CV. '
       'Three inherited source-user-out frozen classifiers and source OOF temperatures/population are reused. '
       'Global source OOF temperature/prior fitting and hyperparameter selection are policy training, not an unbiased '
       'source generalization estimate. Target composition reuses frozen180 cue-query provider/path arrays and saved '
       'calibrations. No target query feature inference, source classifier retraining, target tuning, independent '
       'prospective, automatic/physiological boundary, pinch,250Hz device, physical F6/F9 or default claim.')


def grid():return [dict(n0=n,temperature=t) for n,t in itertools.product((1.,4.,16.,64.),(.25,.5,1.,2.))]


def put_calibration(arrays,prefix,calibration):
    first=next(iter(calibration.values()));ids=tuple(first[2]);labels=tuple(first[1].tolist())
    for g,(x,y,axis) in calibration.items():
        assert tuple(axis)==ids and tuple(y)==labels;arrays[prefix+g]=x
    return dict(prefix=prefix,trial_ids=ids,labels=labels)


def get_calibration(cache,meta):
    return {g:(cache[meta['prefix']+g],np.asarray(meta['labels']),tuple(meta['trial_ids'])) for g in GROUPS}


def candidate(cache,blocks,config,population,policy_id):
    rows=[]
    for b in blocks:
        long=get_calibration(cache,b['long']);current=None if b['current'] is None else get_calibration(cache,b['current'])
        weights,_,_=hierarchical_weights(CLASSES,GROUPS,population,config['n0'],config['temperature'],long,current,source_policy_id=policy_id)
        q=sum(weights[i]*cache[b['query_prefix']+'raw_'+g] for i,g in enumerate(GROUPS));q/=q.sum(1,keepdims=True)
        rows.append(dict(user=b['user'],shots=b['shots'],**score(b['query_labels'],q)))
    users=[dict(user=u,log_loss=float(np.mean([r['log_loss'] for r in rows if r['user']==u])),
        brier=float(np.mean([r['brier'] for r in rows if r['user']==u]))) for u in range(1,19)]
    return dict(config=config,per_user=users,mean_log_loss=float(np.mean([u['log_loss'] for u in users])),
        mean_brier=float(np.mean([u['brier'] for u in users])))


def prepare():
    parent=json.loads((HERE/'ROAM_PRECISION_TRANSITION_V2_PROTOCOL.json').read_text())
    sources=dict(parent['source_sha256'])
    for path in (Path(__file__),HERE/'verify_roam_document_reliability_v3.py',ROOT/'src/emgimu/feature_bank/native_document_reliability_v3.py',
                 ROOT/'tests/test_native_document_reliability_v3.py',ROOT/'tests/test_roam_document_reliability_study_v3.py'):
        sources[path.relative_to(ROOT).as_posix()]=sha(path)
    previous=json.loads(PREVIOUS.read_text());artifacts=dict(previous['artifacts_sha256'])
    for path in (PREVIOUS,INPUT):artifacts[path.relative_to(ROOT).as_posix()]=sha(path)
    target=json.loads((HERE/'ROAM_NATIVE_JOINT_V1_RESULTS.json').read_text())
    target_artifacts=dict(target['artifacts_sha256'])
    for path in (HERE/'ROAM_NATIVE_JOINT_V1_RESULTS.json',ROOT/'feature_bank/ROAM_NATIVE_JOINT_V1_ACCEPTANCE.json'):
        target_artifacts[path.relative_to(ROOT).as_posix()]=sha(path)
    write(PROTOCOL,dict(schema='roam_document_reliability_v3',source_sha256=sources,source_artifact_sha256=artifacts,
        target_artifact_sha256=target_artifacts,archive_sha256=ARCHIVE_SHA,source_users=list(range(1,19)),
        validation_users=list(range(19,24)),descriptive_users=list(range(24,29)),source_candidates=grid(),
        reliability='Exact R=B/(W+1e-10), log(R+1e-10), softmax(logR/tau), alpha=n0/(n0+distinct trials). Long6 shrinks to source population; current0/3/6 shrinks to new long prior. Source-standardized mean features, equal independent trials, no query fitting.',
        selection='Minimize equal-source-user, equal-budget0/1/2 mean logloss of raw seven-provider document reliability on324 recording-disjoint source query cues/972 budget rows. Tie by Brier, original grid index.16 candidates n0=1/4/16/64,tau=.25/.5/1/2. No target selection.',
        target='Commit independently verified source policy before reading target cached readouts. Reuse180 previously inspected three-class oracle cues at0/1/2 current shots; native calibration feature readout only. Preserve source-zero and legacy reliability/window/joint controls. Apply new D/E weights to source providers, existing F7 heads/F8 routing and fixed.75/.125/.125 window/DTW/signature.',
        primary='At2 shots/class on validation19..23, document_joint against BOTH legacy_joint and source_window: lower loss/Brier, nonworse F1/all-class recall, at least3/5 user loss wins per control. Window/reliability comparisons separately descriptive. No default promotion.',
        costs='source_window0 target cues; all personal/session arms6 long+0/3/6 current cues, shared once. Current0 is not zero total registration.',
        scope=SCOPE,default_promoted=False,completion_proven=False))


def check(target=False):
    p=json.loads(PROTOCOL.read_text());files={**p['source_sha256'],**p['source_artifact_sha256'],PROTOCOL.relative_to(ROOT).as_posix():sha(PROTOCOL)}
    if target:
        files.update(p['target_artifact_sha256']);files.update({v.relative_to(ROOT).as_posix():sha(v) for v in (SOURCE,SOURCE_RECEIPT,OUT/'source/policy.json')})
    for rel,h in files.items():
        path=ROOT/rel
        if sha(path)!=h or subprocess.check_output(['git','show','HEAD:'+path.relative_to(REPOSITORY).as_posix()],cwd=REPOSITORY)!=path.read_bytes():
            raise ValueError('Commit exact source/protocol/policy before native phase: '+rel)
    return p


def source_run():
    if SOURCE.exists() or OUT.exists():raise FileExistsError('New source policy phase already started; do not rerun')
    p=check();assert sha(ARCHIVE)==p['archive_sha256'];previous=json.loads(PREVIOUS.read_text());inputs=json.loads(INPUT.read_text())
    folder=OUT/'source';folder.mkdir(parents=True);data=source_data();cache={};blocks=[]
    banks={i:pickle.loads((HERE/f'roam_source_fusion_v1/source/bank_fold{i}.pkl').read_bytes()) for i in range(3)}
    before={i:pickle.dumps(b) for i,b in banks.items()}
    for u in p['source_users']:
        user=f'ROAM_source_s{u}';fold=next(i for i,f in enumerate(inputs['source_folds']) if u in f['held_users'])
        bank=banks[fold];w=build(bank);query=merge([data[u,c] for c in ('unsupported','reaching')])
        first=next(b for b in previous['blocks'] if b['user']==u and b['shots']==0)
        personal=w.load_profile(ROOT/first['personal_path'],user_id=user)
        long=put_calibration(cache,f'u{u}_long_',calibration_features(bank,personal,user))
        qb,qi,qo,_=native_windows(query.batch,40,8);raw=bank.predict_providers(qb,qi,window_offsets=qo,user_id=user)
        order=[raw['trial_ids'].index(t) for t in query.batch.trial_ids]
        for g,q in raw['probabilities'].items():cache[f'u{u}_raw_'+g]=q[order]
        for shots in (0,1,2):
            old=next(b for b in previous['blocks'] if b['user']==u and b['shots']==shots)
            session=None if old['session_path'] is None else w.load_profile(ROOT/old['session_path'],user_id=user,session_id='hanging',personal=personal)
            current=None if session is None else put_calibration(cache,f'u{u}_current{shots}_',calibration_features(bank,session,user))
            assert not set(query.batch.recording_ids)&(set(personal.calibration.recording_ids)|(set() if session is None else set(session.calibration.recording_ids)))
            blocks.append(dict(user=u,fold=fold,shots=shots,long=long,current=current,query_prefix=f'u{u}_',
                query_ids=query.batch.trial_ids,query_labels=[query.labels[t] for t in query.batch.trial_ids],
                query_recording_ids=query.batch.recording_ids,personal_path=old['personal_path'],session_path=old['session_path']))
        if u%3==0:print(f'Native source calibration/query feature collection {u}/18 users',flush=True)
    assert all(pickle.dumps(banks[i])==before[i] for i in banks)
    candidates=[dict(index=i,**candidate(cache,blocks,c,inputs['source_population'],sha(PROTOCOL))) for i,c in enumerate(p['source_candidates'])]
    selected=min(candidates,key=lambda c:(c['mean_log_loss'],c['mean_brier'],c['index']))
    np.savez_compressed(folder/'features_probabilities.npz',**cache)
    policy=dict(schema='native_document_reliability_policy_v3',source_protocol_sha256=sha(PROTOCOL),
        inference_bank_id=inputs['inference_bank_id'],providers=GROUPS,classes=CLASSES,sensor_contract=(200.,40,8),
        channel_ids=query.batch.channel_ids,preprocessing_id=query.batch.preprocessing_id,population=inputs['source_population'],
        **selected['config'],source_query_recording_ids=[d.receipt['member'] for d in data.values()],
        source_cache_sha256=sha(folder/'features_probabilities.npz'),selected_candidate=selected['index'])
    policy['policy_id']=policy_digest(policy);write(folder/'policy.json',policy)
    write(SOURCE,dict(schema='roam_document_reliability_v3_source',protocol_sha256=sha(PROTOCOL),blocks=blocks,candidates=candidates,
        selected_config=selected['config'],selected_candidate=selected['index'],source_query_trials=324,source_budget_rows=972,
        existing_classifiers_refitted=False,target_arrays_read=False,source_scores_are_unbiased=False,
        artifacts_sha256={v.relative_to(ROOT).as_posix():sha(v) for v in folder.iterdir()},scope=SCOPE))
    print(json.dumps(dict(selected_config=selected['config'],source_training_logloss=selected['mean_log_loss'])),flush=True)


def target_run():
    folder=OUT/'target'
    if folder.exists() or TARGET.exists():raise FileExistsError('Target composition phase already started; do not rerun')
    p=check(target=True);old=json.loads((HERE/'ROAM_NATIVE_JOINT_V1_RESULTS.json').read_text())
    bank=pickle.loads((HERE/'roam_native_joint_v1/source_bank.pkl').read_bytes());w=build(bank);before=pickle.dumps(w)
    adapter=NativeDocumentReliabilityV3(w,OUT/'source/policy.json',expected_sha256=sha(OUT/'source/policy.json'))
    inherited=np.load(HERE/'roam_native_joint_v1/readouts.npz',allow_pickle=False)
    folder.mkdir();arrays={};blocks=[];cells=[]
    for subject in old['subjects']:
        u=subject['user'];user=f'ROAM_s{u}';personal=w.load_profile(ROOT/subject['personal_path'],user_id=user)
        raw={g:inherited[f'u{u}_source_'+g] for g in GROUPS}
        for shots in (0,1,2):
            path=subject['budgets'][str(shots)]['session_path']
            session=None if path is None else w.load_profile(ROOT/path,user_id=user,session_id='hanging',personal=personal)
            state=adapter.prepare_state(personal=personal,session=session,user_id=user,session_id='hanging')
            prefix=f'u{u}_s{shots}_';decision={g:inherited[prefix+'decision_'+g] for g in GROUPS}
            result=adapter.compose(state,raw,decision,inherited[prefix+'DTW_blended'],inherited[prefix+'signature_blended'],
                trial_ids=subject['query_ids'],recording_ids=subject['query_recordings'],user_id=user,session_id='hanging',
                provider_trial_ids={g:subject['query_ids'] for g in GROUPS},provider_classes={g:CLASSES for g in GROUPS})
            values={name:inherited[prefix+key] for name,key in dict(source_window='population',legacy_reliability='window_reliability',legacy_window='window_full',legacy_joint='joint_full').items()}
            values.update({name:result[name] for name in ARMS if name.startswith('document_')})
            for arm,q in values.items():
                arrays[prefix+arm]=q;cells.append(dict(user=u,shots=shots,arm=arm,target_calibration_trials=0 if arm=='source_window' else 6+3*shots,**score(subject['query_labels'],q)))
            blocks.append(dict(user=u,shots=shots,query_ids=subject['query_ids'],query_labels=subject['query_labels'],
                query_recording_ids=subject['query_recordings'],personal_path=subject['personal_path'],session_path=path,
                weights=result['document_weights'],routed_weights=result['routed_weights'],state_id=state.state_id,
                long_calibration_trials=state.long_trials,current_calibration_trials=state.current_trials))
        if u%2==0:print(f'Cached target composition {u-18}/10 users',flush=True)
    assert pickle.dumps(w)==before
    aggregates=[]
    for phase,users in (('validation',range(19,24)),('descriptive_final',range(24,29)),('all',range(19,29))):
        y=[v for b in blocks if b['user'] in users and b['shots']==0 for v in b['query_labels']]
        for shots in (0,1,2):
            for arm in ARMS:
                q=np.concatenate([arrays[f'u{u}_s{shots}_'+arm] for u in users])
                aggregates.append(dict(phase=phase,shots=shots,arm=arm,**score(y,q)))
    guard,wins=guards(aggregates,cells)
    np.savez_compressed(folder/'readouts.npz',**arrays)
    write(TARGET,dict(schema='roam_document_reliability_v3_target',protocol_sha256=sha(PROTOCOL),source_result_sha256=sha(SOURCE),
        source_acceptance_sha256=sha(SOURCE_RECEIPT),policy_sha256=sha(OUT/'source/policy.json'),blocks=blocks,cells=cells,aggregates=aggregates,
        primary_guards=guard,primary_pass=all(all(v.values()) for v in guard.values()),validation_user_loss_wins=wins,
        target_query_feature_inference_repeated=False,target_fitted_or_selected=False,source_policy_committed_before_target=True,
        artifacts_sha256={v.relative_to(ROOT).as_posix():sha(v) for v in folder.iterdir()},default_promoted=False,completion_proven=False,scope=SCOPE))
    print(json.dumps(dict(primary_guards=guard,validation_user_loss_wins=wins)),flush=True)


def guards(aggregates,cells):
    ag={(r['phase'],r['shots'],r['arm']):r for r in aggregates};cu={(r['user'],r['shots'],r['arm']):r for r in cells}
    full=ag['validation',2,'document_joint'];guards={};wins={}
    for arm in ('legacy_joint','source_window'):
        base=ag['validation',2,arm];wins[arm]=sum(cu[u,2,'document_joint']['log_loss']<cu[u,2,arm]['log_loss'] for u in range(19,24))
        guards[arm]=dict(lower_log_loss=full['log_loss']<base['log_loss'],lower_brier=full['brier']<base['brier'],
            nonworse_macro_f1=full['macro_f1']>=base['macro_f1'],nonworse_all_recalls=all(full['recall'][c]>=base['recall'][c] for c in CLASSES),at_least3_user_loss_wins=wins[arm]>=3)
    return guards,wins


if __name__=='__main__':
    parser=argparse.ArgumentParser();m=parser.add_mutually_exclusive_group(required=True)
    m.add_argument('--prepare',action='store_true');m.add_argument('--source',action='store_true');m.add_argument('--target',action='store_true');args=parser.parse_args()
    prepare() if args.prepare else source_run() if args.source else target_run()
