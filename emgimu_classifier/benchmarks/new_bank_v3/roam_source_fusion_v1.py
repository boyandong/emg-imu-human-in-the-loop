"""Source-only population mixture; freeze source policy before target readouts.

Reuses frozen three OOF source classifiers and target predictions. New native
work only collects disjoint source-user posture readouts and fits five weights.
"""
import argparse
import csv
import json
import pickle
import subprocess
import zipfile
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score,confusion_matrix
from benchmarks.new_bank_v3.roam_native_joint_v1 import (
    ROOT,HERE,REPOSITORY,ARCHIVE,ARCHIVE_SHA,SEQUENCE,sha,write,merge,select,build,CLASSES,GROUPS)
from emgimu.datasets.roam_cued_intervals_v1 import load_roam_cued_intervals
from emgimu.feature_bank.native_bout_window_adapter_v2 import native_windows
from emgimu.feature_bank.frozen_emg_provider_bank_v1 import FrozenEmgProviderBankV1
from emgimu.feature_bank.source_probability_fusion_v1 import SourceProbabilityFusionV1

PROTOCOL=HERE/'ROAM_SOURCE_FUSION_V1_PROTOCOL.json'
INPUT=HERE/'ROAM_SOURCE_FUSION_V1_SOURCE_INPUT.json'
SOURCE=HERE/'ROAM_SOURCE_FUSION_V1_SOURCE_RESULTS.json'
TARGET=HERE/'ROAM_SOURCE_FUSION_V1_TARGET_RESULTS.json'
OUT=HERE/'roam_source_fusion_v1'
OLD=HERE/'ROAM_NATIVE_JOINT_V1_RESULTS.json'
PROVIDERS=('source_window','reliability_window','F7F8_window','DTW','signature')
OLD_NAMES=('population','window_reliability','window_full','DTW_blended','signature_blended')
SCOPE=('Retrospective native8/200Hz three-class ROAM source-only convex population fusion. '
    'Frozen representations/classifiers and original target probabilities are reused, not refitted. '
    'Source OOF users1..18 supply972 budget rows on324 independent posture queries; weights are committed '
    'before application to180 previously inspected19..28 queries. Source temperature/prior used all source '
    'OOF labels; source policy training loss is not an unbiased source evaluation. Postures are not days '
    'or redonnings; cue boundaries are oracle, pinch/physical ring/F6/ADC-F9/device efficacy unavailable.')


def source_data():
    data={}
    with zipfile.ZipFile(ARCHIVE) as archive:
        for u in range(1,19):
            for condition in ('resting','hanging','unsupported','reaching'):
                member=f'data/ROAM_EMG/s{u}/s{u}_static_{condition}.csv'
                d=load_roam_cued_intervals(archive,member)
                if tuple(d.labels.values())!=SEQUENCE or d.receipt['excluded_intervals']:
                    raise ValueError('Source native cue coverage differs')
                data[u,condition]=d
    return data


def prepare():
    if sha(ARCHIVE)!=ARCHIVE_SHA:raise ValueError('Native archive changed')
    old=json.loads(OLD.read_text(encoding='utf8'));data=source_data()
    source_input=dict(schema='roam_source_fusion_v1_source_input',source_origin_sha256=sha(OLD),
        source_temperatures=old['source_temperatures'],source_population=old['source_population'],
        source_folds=[{k:f[k] for k in ('held_users','fit_ids','path','sha256')} for f in old['source_oof']],
        inference_bank_id=old['source_bank_id'],forbidden_target_query_ids=[t for s in old['subjects'] for t in s['query_ids']])
    write(INPUT,source_input)
    sources=[ROOT/n for n in json.loads((HERE/'ROAM_NATIVE_JOINT_V1_PROTOCOL.json').read_text(encoding='utf8'))['source_sha256']]
    sources += [ROOT/'src/emgimu/feature_bank/source_probability_fusion_v1.py',Path(__file__),
        HERE/'verify_roam_source_fusion_v1.py',ROOT/'tests/test_source_probability_fusion_v1.py',
        ROOT/'tests/test_roam_source_fusion_study_v1.py']
    sources=sorted(set(sources))
    source_artifacts={INPUT.relative_to(ROOT).as_posix():sha(INPUT)}
    source_artifacts.update({f['path']:f['sha256'] for f in source_input['source_folds']})
    write(PROTOCOL,dict(schema='roam_source_fusion_v1',archive_sha256=ARCHIVE_SHA,
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sources},source_artifact_sha256=source_artifacts,
        target_artifact_sha256={OLD.relative_to(ROOT).as_posix():sha(OLD),
            'benchmarks/new_bank_v3/roam_native_joint_v1/readouts.npz':sha(HERE/'roam_native_joint_v1/readouts.npz')},
        native_source_recordings=[d.receipt for d in data.values()],source_users=list(range(1,19)),
        providers=PROVIDERS,classes=CLASSES,current_shots=[0,1,2],regularization=1e-6,
        source_fit='Reuse each source user-out fold representation/scaler/classifier. Build Rest-long6 and separate hanging-current0/3/6 profiles, then predict full unsupported/reaching cues. Five population weights minimize equal-user source probability logloss plus fixed1e-6 L2 penalty on the simplex. Source OOF temperatures/prior are inherited source policy training parameters, not an unbiased source estimate.',
        target_apply='Commit learned source manifest and source result before reading/applying frozen target probability arrays. No target model/profile/temperature/weight fitting. All five providers evaluated; shared registration charged6+3*shots even if a learned coefficient is zero. Source-only population/reliability controls keep their previously recorded costs.',
        omissions='Omit one of five providers, renormalize retained frozen coefficients, never refit. If remaining mass is zero, emit Unknown and uniform scoreable fallback; label metrics count Unknown as wrong.',
        primary='At2 current cues/class on previously inspected validation19..23, source-selected must beat BOTH fixed_joint and population: lower loss/Brier, nonworse F1/all-class recall and at least3/5 per-user loss wins for each. Final24..28 and pooled outcomes descriptive, no default promotion.',
        scope=SCOPE,default_promoted=False))


def committed(path):
    relative=path.relative_to(REPOSITORY).as_posix()
    raw=subprocess.run(['git','show','HEAD:'+relative],cwd=REPOSITORY,capture_output=True,check=True).stdout
    if raw!=path.read_bytes():raise ValueError('Commit frozen inputs/policy before native work: '+relative)


def check(p,*,target=False):
    for name,digest in {**p['source_sha256'],**p['source_artifact_sha256']}.items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen source input differs: '+name)
        committed(ROOT/name)
    committed(PROTOCOL)
    if sha(ARCHIVE)!=p['archive_sha256']:raise ValueError('Native archive changed')
    if target:
        for name,digest in p['target_artifact_sha256'].items():
            if sha(ROOT/name)!=digest:raise ValueError('Frozen target artifact differs')
        committed(SOURCE);committed(OUT/'source/policy.json')


def component_readout(w,query,personal,session,user):
    common=dict(personal=personal,session=session,user_id=user,session_id='hanging')
    r=w.predict(query.batch,**common);q=r['temporal']['arms']
    b,ids,offsets,_=native_windows(query.batch,40,8)
    raw=w.window.bank.predict(b,ids,window_offsets=offsets,user_id=user)
    reliability=w.window.predict(b,ids,window_offsets=offsets,personal=personal.window,
        session=None if session is None else session.window,user_id=user,session_id='hanging',
        observed_channel_ids=query.batch.channel_ids,preprocessing_id=query.batch.preprocessing_id,
        use_anchor=False,use_session_routing=False)
    order=[raw['trial_ids'].index(t) for t in query.batch.trial_ids]
    return dict(source_window=raw['probabilities'][order],reliability_window=reliability['probabilities'][order],
        F7F8_window=q['base'],DTW=q['DTW_blended'],signature=q['signature_blended'])


def source_run():
    if SOURCE.exists() or OUT.exists():raise FileExistsError('Source study already started/completed; do not rerun')
    p=json.loads(PROTOCOL.read_text(encoding='utf8'));check(p)
    inputs=json.loads(INPUT.read_text(encoding='utf8'));data=source_data()
    if json.loads(json.dumps([d.receipt for d in data.values()]))!=p['native_source_recordings']:
        raise ValueError('Source native receipt differs')
    folder=OUT/'source';folder.mkdir(parents=True);banks={};blocks=[];arrays={};artifacts={}
    for i,fold in enumerate(inputs['source_folds']):
        families,models=pickle.loads((ROOT/fold['path']).read_bytes())
        bank=FrozenEmgProviderBankV1(families,models,inputs['source_temperatures'],classes=CLASSES,class_names=CLASSES,
            source_trial_ids=fold['fit_ids'],source_policy_id=sha(PROTOCOL),sample_rate_hz=200.,window_samples=40,
            channels=8,population=inputs['source_population'],n0=4.,reliability_temperature=.5)
        path=folder/f'bank_fold{i}.pkl';path.write_bytes(pickle.dumps(bank))
        banks[i]=build(bank)
    print('1/3 Reusing3 frozen source classifiers; collecting18 source-user posture readouts',flush=True)
    all_trials=[];all_labels=[];all_scenarios=[];all_users=[];all_probabilities={g:[] for g in PROVIDERS}
    for u in range(1,19):
        i=next(i for i,f in enumerate(inputs['source_folds']) if u in f['held_users']);w=banks[i]
        user=f'ROAM_source_s{u}';query=merge([data[u,c] for c in ('unsupported','reaching')]);long=select(data[u,'resting'],2)
        forbidden=dict(forbidden_trial_ids=query.batch.trial_ids,forbidden_recording_ids=query.batch.recording_ids)
        personal=w.enroll(long.batch,long.labels,user_id=user,session_id='resting',**forbidden)
        pp=folder/f's{u}_personal.zip';w.save_profile(personal,pp)
        for shots in (0,1,2):
            session=None;sp=None
            if shots:
                current=select(data[u,'hanging'],shots)
                session=w.enroll(current.batch,current.labels,user_id=user,session_id='hanging',personal=personal,**forbidden)
                sp=folder/f's{u}_current{shots}.zip';w.save_profile(session,sp)
            components=component_readout(w,query,personal,session,user);key=f'u{u}_s{shots}_'
            for g,q in components.items():arrays[key+g]=q;all_probabilities[g].append(q)
            y=[query.labels[t] for t in query.batch.trial_ids]
            all_trials.extend(query.batch.trial_ids);all_labels.extend(y)
            all_scenarios.extend([f'current_shots{shots}']*len(y));all_users.extend([user]*len(y))
            blocks.append(dict(user=u,fold=i,shots=shots,query_ids=query.batch.trial_ids,query_labels=y,
                personal_path=pp.relative_to(ROOT).as_posix(),session_path=None if sp is None else sp.relative_to(ROOT).as_posix(),
                unique_calibration_trials=6+3*shots))
        print(f'  source user {u}/18 complete',flush=True)
    print('2/3 Fitting5 nonnegative population coefficients using source users only',flush=True)
    all_probabilities={g:np.concatenate(v) for g,v in all_probabilities.items()}
    policy,diagnostic=SourceProbabilityFusionV1.fit(all_probabilities,all_labels,classes=CLASSES,providers=PROVIDERS,
        trial_ids=all_trials,scenario_ids=all_scenarios,user_ids=all_users,source_protocol_id=sha(PROTOCOL),
        inference_bank_id=inputs['inference_bank_id'],forbidden_trial_ids=inputs['forbidden_target_query_ids'],
        regularization=p['regularization'])
    write(folder/'policy.json',policy.manifest());np.savez_compressed(folder/'readouts.npz',**arrays)
    artifacts={path.relative_to(ROOT).as_posix():sha(path) for path in folder.iterdir()}
    write(SOURCE,dict(schema='roam_source_fusion_v1_source',protocol_sha256=sha(PROTOCOL),policy_id=policy.policy_id,
        policy_weights=dict(zip(policy.providers,policy.weights)),diagnostic=diagnostic,blocks=blocks,
        artifacts_sha256=artifacts,source_native_recordings=72,source_native_intervals=648,
        source_query_trials=324,source_budget_rows=972,target_probability_arrays_read=False,
        existing_source_classifiers_refitted=False,scope=SCOPE))
    print('3/3 Saved source policy; commit it before applying any target probabilities',flush=True)
    print(json.dumps(dict(weights=dict(zip(policy.providers,policy.weights)),stationarity_gap=diagnostic['simplex_stationarity_gap'])),flush=True)


def score(labels,q,predicted=None):
    y=np.asarray(labels);yp=np.asarray(CLASSES)[q.argmax(1)] if predicted is None else np.asarray(predicted)
    cm=confusion_matrix(y,yp,labels=(*CLASSES,'Unknown'));truth=np.array([CLASSES.index(c) for c in y])
    return dict(trials=len(y),macro_f1=float(f1_score(y,yp,labels=CLASSES,average='macro',zero_division=0)),
        accuracy=float(np.mean(y==yp)),log_loss=float(-np.log(np.maximum(q[np.arange(len(y)),truth],1e-15)).mean()),
        brier=float(np.mean((q-np.eye(3)[truth])**2)),unknown_trials=int(np.sum(yp=='Unknown')),
        recall={c:float(cm[i,i]/cm[i].sum()) for i,c in enumerate(CLASSES)},confusion_with_Unknown=cm.tolist())


def target_run():
    folder=OUT/'target'
    if TARGET.exists() or folder.exists():raise FileExistsError('Target application already started/completed; do not rerun')
    p=json.loads(PROTOCOL.read_text(encoding='utf8'));check(p,target=True)
    source=json.loads(SOURCE.read_text(encoding='utf8'));policy=SourceProbabilityFusionV1.from_manifest(json.loads((OUT/'source/policy.json').read_text(encoding='utf8')))
    old=json.loads(OLD.read_text(encoding='utf8'));native=np.load(HERE/'roam_native_joint_v1/readouts.npz',allow_pickle=False)
    if policy.policy_id!=source['policy_id'] or policy.source_protocol_id!=sha(PROTOCOL):raise ValueError('Selected source policy differs')
    folder.mkdir();arrays={};rows=[];cells=[]
    for subject in old['subjects']:
        u=subject['user'];ids=tuple(subject['query_ids']);labels=subject['query_labels']
        for shots in (0,1,2):
            key=f'u{u}_s{shots}_';components={g:native[key+n] for g,n in zip(PROVIDERS,OLD_NAMES)}
            common=dict(evaluation_trial_ids=ids,inference_bank_id=old['source_bank_id'])
            def predict(names):
                return policy.predict({g:components[g] for g in names},provider_trial_ids={g:ids for g in names},
                    provider_classes={g:CLASSES for g in names},**common)
            selected=predict(PROVIDERS);arms={'source_selected':selected}
            arms.update({'minus_'+g:predict(tuple(n for n in PROVIDERS if n!=g)) for g in PROVIDERS})
            for name,old_name in [('fixed_joint','joint_full'),('window_full','window_full'),('population','population'),('reliability','window_reliability')]:
                q=native[key+old_name];arms[name]=dict(probabilities=q,labels=tuple(np.asarray(CLASSES)[q.argmax(1)]),rejected=np.zeros(len(q),bool))
            q=np.mean(list(components.values()),axis=0)
            arms['uniform_components']=dict(probabilities=q,labels=tuple(np.asarray(CLASSES)[q.argmax(1)]),rejected=np.zeros(len(q),bool))
            for arm,r in arms.items():
                q=r['probabilities'];arrays[key+arm]=q;arrays[key+arm+'_rejected']=r['rejected']
                cost=0 if arm=='population' else 6+3*shots
                cells.append(dict(user=u,phase='validation' if u<=23 else 'descriptive_final',shots=shots,arm=arm,
                    unique_calibration_trials=cost,**score(labels,q,r['labels'])))
                for t,c,yp,prob in zip(ids,labels,r['labels'],q):
                    rows.append(dict(user=u,shots=shots,arm=arm,trial_id=t,true_label=c,predicted=yp,
                        **{'p_'+c:float(prob[j]) for j,c in enumerate(CLASSES)}))
    aggregates=[]
    for phase,users in [('validation',range(19,24)),('descriptive_final',range(24,29)),('descriptive_all',range(19,29))]:
        labels=[c for s in old['subjects'] if s['user'] in users for c in s['query_labels']]
        for shots in (0,1,2):
            for arm in arms:
                q=np.concatenate([arrays[f'u{u}_s{shots}_{arm}'] for u in users])
                rejected=np.concatenate([arrays[f'u{u}_s{shots}_{arm}_rejected'] for u in users])
                yp=np.asarray(CLASSES,object)[q.argmax(1)];yp[rejected]='Unknown'
                aggregates.append(dict(phase=phase,shots=shots,arm=arm,**score(labels,q,yp)))
    selected=next(c for c in aggregates if (c['phase'],c['shots'],c['arm'])==('validation',2,'source_selected'))
    guards={};wins={}
    for comparison in ('fixed_joint','population'):
        base=next(c for c in aggregates if (c['phase'],c['shots'],c['arm'])==('validation',2,comparison))
        wins[comparison]=sum(next(c for c in cells if (c['user'],c['shots'],c['arm'])==(u,2,'source_selected'))['log_loss']<
            next(c for c in cells if (c['user'],c['shots'],c['arm'])==(u,2,comparison))['log_loss'] for u in range(19,24))
        guards[comparison]=dict(lower_log_loss=selected['log_loss']<base['log_loss'],lower_brier=selected['brier']<base['brier'],
            nonworse_macro_f1=selected['macro_f1']>=base['macro_f1'],
            nonworse_all_class_recall=all(selected['recall'][c]>=base['recall'][c] for c in CLASSES),at_least3_user_loss_wins=wins[comparison]>=3)
    np.savez_compressed(folder/'readouts.npz',**arrays)
    with (folder/'predictions.csv').open('x',encoding='utf8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(TARGET,dict(schema='roam_source_fusion_v1_target',protocol_sha256=sha(PROTOCOL),source_result_sha256=sha(SOURCE),
        policy_sha256=sha(OUT/'source/policy.json'),policy_id=policy.policy_id,weights=dict(zip(policy.providers,policy.weights)),
        cells=cells,aggregates=aggregates,primary_guards=guards,user_loss_wins=wins,
        primary_pass=all(v for g in guards.values() for v in g.values()),independent_query_trials=180,
        predictions=len(rows),artifacts_sha256={f.relative_to(ROOT).as_posix():sha(f) for f in folder.iterdir()},
        target_models_profiles_or_policy_refitted=False,source_policy_committed_before_target_apply=True,
        scope=SCOPE,default_promoted=False,physical_validation_proven=False,completion_proven=False))
    print(json.dumps(dict(target_cells=len(cells),predictions=len(rows),primary_guards=guards),ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',action='store_true');group.add_argument('--source',action='store_true');group.add_argument('--target',action='store_true')
    args=parser.parse_args()
    if args.prepare:prepare()
    elif args.source:source_run()
    else:target_run()
