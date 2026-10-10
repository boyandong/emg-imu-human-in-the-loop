"""Matched raw/normalized source training and personal/current-session evaluation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
from scipy.special import softmax
from benchmarks.song_real8_study import load_session
from benchmarks.song_real8.song_personal_session_workflow_v1 import (aggregate,select,score,windows,fit_source,read,
    CLASSES,CHANNELS,PREPROCESSING)
from benchmarks.new_bank_v3.emg_window_bank_v1 import GROUPS
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.calibration import DocumentPersonalNormalizerV2
from emgimu.feature_bank.force_nested_oof import fit_temperature,temperature_probability
from emgimu.feature_bank.document_reliability_v2 import DocumentReliabilityWeightsV2
from emgimu.feature_bank.frozen_emg_provider_bank_v1 import FrozenEmgProviderBankV1
from emgimu.feature_bank.personal_session_workflow_v1 import PersonalSessionWorkflowV1
from emgimu.feature_bank.matched_normalized_bank_v1 import MatchedNormalizedProviderBankV1,MatchedNormalizedWorkflowV1

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PROTOCOL=HERE/'SONG_MATCHED_NORMALIZATION_V1_PROTOCOL.json'
RESULT=HERE/'SONG_MATCHED_NORMALIZATION_V1_RESULTS.json'
OUT=HERE/'song_matched_normalization_v1'
MODES=('raw','normalized')
CONFIG=dict(channel_ids=CHANNELS,preprocessing_id=PREPROCESSING,rest_label='neutral',
            quality_options={'line_frequency_hz':50,'pre_highpass_available':False})


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def subset(item,trials):
    mask=np.isin(item['trial'],trials)
    return dict(batch=item['batch'].take(np.flatnonzero(mask)),trial=item['trial'][mask],hand=item['hand'][mask])


def normalize(item,normalizer):return dict(item,batch=normalizer.transform(item['batch']))


def norm(item):return DocumentPersonalNormalizerV2(rest_label='neutral').fit(item['batch'],item['hand'])


def join(items):
    return dict(batch=FeatureBatch(np.concatenate([v['batch'].emg for v in items]),250.),
                trial=np.concatenate([v['trial'] for v in items]),hand=np.concatenate([v['hand'] for v in items]))


def trial_axis(item):
    _,y,ids=aggregate(item['batch'].emg.mean(1),item)
    return ids,y


def prepare():
    if PROTOCOL.exists():raise FileExistsError('Frozen protocol already exists')
    previous=json.loads((HERE/'SONG_PERSONAL_SESSION_V1_PROTOCOL.json').read_text(encoding='utf8'))
    files=list(previous['source_sha256'])+[
        'src/emgimu/feature_bank/matched_normalized_bank_v1.py','tests/test_matched_normalized_bank_v1.py',
        'benchmarks/song_real8/song_matched_normalization_v1.py','benchmarks/song_real8/SONG_PERSONAL_SESSION_V1_PROTOCOL.json']
    p=dict(schema='song_matched_normalization_v1',source_folder=previous['source_folder'],
        hdf5_sha256=previous['hdf5_sha256'],source_sessions=['S01','S02'],personal_session='S03',current_session='S04',
        classes=CLASSES,groups=GROUPS,current_shots_per_class=[0,1,2,5],long_term_shots_per_class=5,
        source_normalization_shots_per_class=5,selection_seed=previous['selection_seed'],source_seed=20261010,
        source_fit='Both branches exclude the same5/class source calibration reservation from family/scaler/classifier fits. Raw fits raw causal windows; normalized fits document Rest-median and active Q95+1e-10 transformed windows. Within each leave-source-recording-out fold, every family/scaler/classifier is refitted; source training normalization uses its own reserved20 trials only.',
        source_validation='Each held source recording reserves the same20 trials. At0 current shots use the training-recording long-term normalizer; at1/2/5 use only the nested held-recording calibration subset. Fit source OOF probability temperatures and population weights on the equal-mass0/1/2/5 validation scenarios. Choose n0=[4,12,24], tau=[.5,1,2] by source-only mean loss after personal-then-current reliability updates. Source OOF data are reused for calibration and policy selection; not nested unbiased source-performance evidence.',
        target_split='Same S03 long-term20 and S04 reserved20/remaining124 trial IDs as the earlier personal-session experiment. Previously inspected, same-person/day developmental data. At0 current shots normalized models still require20 long-term trials. No evaluation-label probability fitting, normalization or tuning.',
        primary='Predeclared five-shot normalized_session versus matched raw_session: lower logloss and class-mean Brier, nonworse macro-F1 and nonworse recall for every class. Report all other budgets, source-population/personal/F0 controls and all six provider removals in both domains regardless of signs. Passing is developmental evidence only, never automatic GUI/default promotion.',
        scope='Six declared EMG window providers with matched normalization training, not full document F0-F9. Separate source checkpoints and profile identities; source/current calibration trials excluded from evaluation. No raw-ADC quality or F8 routing is inferred from normalized classifier inputs. No GUI normalized-candidate integration, live efficacy, physical re-donning, cross-day/general-user proof or mechanical-force claim.',
        source_sha256={name:sha(ROOT/name) for name in files},default_promoted=False)
    with PROTOCOL.open('x',encoding='utf8',newline='\n') as f:json.dump(p,f,indent=2);f.write('\n')
    print('Frozen matched normalization protocol; native data not loaded',flush=True)


def policy_selection(folds,temperatures,population):
    for f in folds:
        for shots in (0,1,2,5):
            f['eval'][shots]['q']={g:temperature_probability(q,temperatures[g]) for g,q in f['eval'][shots]['q'].items()}
    candidates=[]
    for n0 in (4.,12.,24.):
        for tau in (.5,1.,2.):
            losses=[]
            for f in folds:
                personal=DocumentReliabilityWeightsV2(CLASSES,GROUPS,tuple(population),n0,tau,'matched_source_oof')
                long=personal.calculate({g:(z,f['long_y'],f['long_ids']) for g,z in f['long_features'].items()},
                    forbidden_trial_ids=f['fit_ids'])['final']
                for shots in (0,1,2,5):
                    weights=long
                    if shots:
                        c=f['cal'][shots]
                        weights=DocumentReliabilityWeightsV2(CLASSES,GROUPS,tuple(long),n0,tau,'matched_source_current').calculate(
                            {g:(z,c['y'],c['ids']) for g,z in c['features'].items()},
                            forbidden_trial_ids=tuple(f['fit_ids'])+tuple(f['long_ids'])+tuple(f['eval'][shots]['ids']))['final']
                    v=f['eval'][shots];q=sum(float(weights[i])*v['q'][g] for i,g in enumerate(GROUPS))
                    losses.append(score(v['y'],q)['log_loss'])
            candidates.append(dict(n0=n0,temperature=tau,mean_source_loss=float(np.mean(losses))))
    return min(candidates,key=lambda x:(x['mean_source_loss'],x['n0'],x['temperature'])),candidates


def run():
    if RESULT.exists() or OUT.exists():raise FileExistsError('Existing normalized experiment will not be overwritten')
    p=json.loads(PROTOCOL.read_text(encoding='utf8'))
    for name,digest in p['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen source changed: '+name)
    def load(s):
        item=load_session(Path(p['source_folder'])/f'2026-09-18_{s}',s,filter_mode='causal')
        if item['audit']['sha256']!=p['hdf5_sha256'][s]:raise ValueError('Native recording changed')
        item['batch']=FeatureBatch(item['batch'].emg,250.)
        return item
    print('1/4 Matched source reservations and leave-recording-out fits',flush=True)
    source={s:load(s) for s in p['source_sessions']};partitions={};normalizers={}
    for s,item in source.items():
        ids,y=trial_axis(item);cal,ev=select(ids,y,5,p['selection_seed'])
        partitions[s]=dict(cal=ids[cal],fit=ids[ev]);normalizers[s]=norm(subset(item,ids[cal]))
    folds={mode:[] for mode in MODES};temperatures={};populations={};policies={};candidate_tables={}
    for held in p['source_sessions']:
        train=next(s for s in p['source_sessions'] if s!=held)
        traincal=subset(source[train],partitions[train]['cal']);trainfit=subset(source[train],partitions[train]['fit'])
        heldids,heldy=trial_axis(source[held]);heldeval=subset(source[held],partitions[held]['fit'])
        for mode in MODES:
            training=trainfit if mode=='raw' else normalize(trainfit,normalizers[train])
            fitted,models,_=fit_source(training)
            long_item=traincal if mode=='raw' else normalize(traincal,normalizers[train])
            long_features,_,ly,lids=read(fitted,models,long_item)
            fold=dict(session=held,fit_ids=partitions[train]['fit'],long_ids=lids,long_y=ly,
                      long_features=long_features,eval={},cal={})
            for shots in (0,1,2,5):
                chosen=None;reference=normalizers[train]
                if shots:
                    ci,_=select(heldids,heldy,shots,p['selection_seed']);chosen=subset(source[held],heldids[ci]);reference=norm(chosen)
                    readcal=chosen if mode=='raw' else normalize(chosen,reference)
                    z,_,cy,cids=read(fitted,models,readcal);fold['cal'][shots]=dict(features=z,y=cy,ids=cids)
                validation=heldeval if mode=='raw' else normalize(heldeval,reference)
                _,q,y,ids=read(fitted,models,validation);fold['eval'][shots]=dict(q=q,y=y,ids=ids)
            folds[mode].append(fold)
            print(f'Source fold {train}->{held}/{mode}: all4 budgets cached',flush=True)
    for mode in MODES:
        frames=[f['eval'][n] for f in folds[mode] for n in (0,1,2,5)]
        y=np.concatenate([v['y'] for v in frames]);yi=np.array([CLASSES.index(c) for c in y])
        temperatures[mode]={g:fit_temperature(np.concatenate([v['q'][g] for v in frames]),yi) for g in GROUPS}
        pop=softmax(-np.array([score(y,temperature_probability(np.concatenate([v['q'][g] for v in frames]),temperatures[mode][g]))['log_loss'] for g in GROUPS]))
        populations[mode]=tuple(pop)
        policies[mode],candidate_tables[mode]=policy_selection(folds[mode],temperatures[mode],pop)
    print('2/4 Fit and freeze paired source checkpoints',flush=True)
    OUT.mkdir();workflows={};source_metadata={};source_all=tuple(np.unique(np.concatenate([v['trial'] for v in source.values()])))
    source_fit=tuple(sorted(t for d in partitions.values() for t in d['fit']));before={}
    for mode in MODES:
        items=[subset(source[s],partitions[s]['fit']) for s in p['source_sessions']]
        if mode=='normalized':items=[normalize(v,normalizers[s]) for s,v in zip(p['source_sessions'],items)]
        fitted,models,dimensions=fit_source(join(items));selected=policies[mode]
        kwargs=dict(classes=CLASSES,class_names=CLASSES,source_trial_ids=source_all,source_policy_id=sha(PROTOCOL)+':'+mode,
            sample_rate_hz=250.,window_samples=50,channels=8,population=populations[mode],
            n0=selected['n0'],reliability_temperature=selected['temperature'])
        if mode=='normalized':
            bank=MatchedNormalizedProviderBankV1(fitted,models,temperatures[mode],source_model_fit_trials=source_fit,
                source_normalization_trials={s:d['cal'].tolist() for s,d in partitions.items()},**kwargs)
            workflow=MatchedNormalizedWorkflowV1(bank,**CONFIG)
        else:
            bank=FrozenEmgProviderBankV1(fitted,models,temperatures[mode],**kwargs);workflow=PersonalSessionWorkflowV1(bank,**CONFIG)
        packed=pickle.dumps(bank);path=OUT/(mode+'_source_bank.pkl');path.write_bytes(packed)
        before[mode]=packed;workflows[mode]=workflow
        source_metadata[mode]=dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),bank_id=bank.bank_id_,
            contract_id=workflow.contract_id,dimensions=dimensions,temperatures=temperatures[mode],population=populations[mode],
            selected_policy=selected,policy_candidates=candidate_tables[mode])
    print('3/4 S03 personal profiles and same124 S04 trials at every budget',flush=True)
    long,current=load('S03'),load('S04');lids,ly=trial_axis(long);cids,cy=trial_axis(current)
    li,_=select(lids,ly,5,p['selection_seed']);reserved,ev=select(cids,cy,5,p['selection_seed'])
    evaluation_ids=cids[ev];ey=cy[ev];eb,ei,eo=windows(current,evaluation_ids)
    personal={};personal_before={};artifacts={};profiles={};roundtrips=[]
    kwargs=dict(user_id='Song',observed_channel_ids=CHANNELS,preprocessing_id=PREPROCESSING)
    lb,lt,lo=windows(long,lids[li]);labels=dict(zip(lids[li],ly[li]))
    for mode,w in workflows.items():
        profile=w.enroll_user(lb,lt,labels,window_offsets=lo,session_id='S03',forbidden_evaluation_trials=evaluation_ids,**kwargs)
        path=OUT/(mode+'_personal_S03.zip');w.save_profile(profile,path);personal[mode]=w.load_profile(path,user_id='Song')
        personal_before[mode]=pickle.dumps(personal[mode]);profiles[mode]=dict(personal=path.relative_to(ROOT).as_posix(),sessions={})
        artifacts[path.relative_to(ROOT).as_posix()]=sha(path)
    cells=[];rows=[];arrays=dict(trial_ids=evaluation_ids,labels=ey,class_names=np.array(CLASSES));states={}
    for shots in (0,1,2,5):
        cb=ci=co=clabels=None
        if shots:
            selected,_=select(cids,cy,shots,p['selection_seed']);cb,ci,co=windows(current,cids[selected]);clabels=dict(zip(cids[selected],cy[selected]))
        for mode,w in workflows.items():
            state=None
            if shots:
                state=w.calibrate_session(cb,ci,clabels,window_offsets=co,personal=personal[mode],session_id='S04',
                    forbidden_evaluation_trials=evaluation_ids,**kwargs)
                path=OUT/f'{mode}_session_S04_{shots}shot.zip';w.save_profile(state,path)
                state=w.load_profile(path,user_id='Song',session_id='S04',personal=personal[mode])
                profiles[mode]['sessions'][str(shots)]=path.relative_to(ROOT).as_posix();artifacts[path.relative_to(ROOT).as_posix()]=sha(path)
            pk=dict(window_offsets=eo,personal=personal[mode],session_id='S04',**kwargs)
            full=w.predict(eb,ei,session=state,**pk);long_only=w.predict(eb,ei,**pk)
            transformed=eb if mode=='raw' else w.classifier_input(eb,session=state,**pk)
            readout=w.bank.predict_providers(transformed,ei,window_offsets=eo,user_id='Song')
            providers=readout['probabilities'];population=w.bank.predict(transformed,ei,window_offsets=eo,user_id='Song')['probabilities']
            arms={'F0':providers['F0'],'population':population,'personal':long_only['probabilities'],'session':full['probabilities']}
            for g in GROUPS:arms['session_minus_'+g]=w.predict(eb,ei,session=state,available=tuple(k for k in GROUPS if k!=g),**pk)['probabilities']
            for arm,q in arms.items():
                total_long=0 if mode=='raw' and arm in ('F0','population') else 20
                total_current=4*shots if arm.startswith('session') or (mode=='normalized' and arm in ('F0','population')) else 0
                cells.append(dict(mode=mode,shots=shots,arm=arm,long_term_calibration_trials=total_long,
                    current_calibration_trials=total_current,**score(ey,q)))
                for t,y,prob in zip(evaluation_ids,ey,q):rows.append(dict(mode=mode,shots=shots,arm=arm,trial_id=t,label=y,
                    **{f'p_{c}':float(prob[i]) for i,c in enumerate(CLASSES)}))
            for g,q in providers.items():arrays[f'{mode}_{shots}_{g}']=q
            states[f'{mode}_{shots}']=dict(weights=list(full['weights']),personal_profile_id=personal[mode].profile_id,
                session_profile_id=None if state is None else state.profile_id,calibration_ids=[] if state is None else list(state.calibration_trials))
            if mode=='normalized':
                arrays[f'normalized_emg_{shots}']=transformed.emg
                reference=personal[mode].normalizer if state is None else state.normalizer
                arrays[f'center_{shots}']=reference.center_;arrays[f'scale_{shots}']=reference.scale_
            if shots:
                # A fresh loaded profile pair, not only the in-memory fitting state.
                restored=w.load_profile(ROOT/profiles[mode]['personal'],user_id='Song')
                reloaded=w.load_profile(ROOT/profiles[mode]['sessions'][str(shots)],user_id='Song',session_id='S04',personal=restored)
                replay=w.predict(eb,ei,session=reloaded,**dict(pk,personal=restored))
                error=float(np.max(abs(full['probabilities']-replay['probabilities'])))
                if error>1e-12:raise ValueError('Persisted matched model/profile changed prediction')
                roundtrips.append(dict(mode=mode,shots=shots,maximum_probability_error=error))
            if pickle.dumps(w.bank)!=before[mode] or pickle.dumps(personal[mode])!=personal_before[mode]:raise ValueError('Source/personal state mutated')
        print(f'{shots}shot/class: both domains, all20 matched cells complete',flush=True)
    primary={(c['mode'],c['shots'],c['arm']):c for c in cells};raw=primary[('raw',5,'session')];normalized=primary[('normalized',5,'session')]
    guards=dict(lower_log_loss=normalized['log_loss']<raw['log_loss'],lower_brier=normalized['brier']<raw['brier'],
        nonworse_macro_f1=normalized['macro_f1']>=raw['macro_f1'],
        nonworse_all_class_recall=all(normalized['recall'][c]>=raw['recall'][c] for c in CLASSES))
    table=OUT/'predictions.csv'
    with table.open('w',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    arrays_path=OUT/'readouts.npz';np.savez_compressed(arrays_path,**arrays)
    for path in (table,arrays_path,*[ROOT/v['path'] for v in source_metadata.values()]):artifacts[path.relative_to(ROOT).as_posix()]=sha(path)
    source_oof={mode:[dict(session=f['session'],fit_ids=f['fit_ids'].tolist(),long_calibration_ids=f['long_ids'].tolist(),
        evaluation_ids=f['eval'][0]['ids'].tolist(),current_calibration_ids={str(n):f['cal'][n]['ids'].tolist() for n in (1,2,5)},
        labels=f['eval'][0]['y'].tolist(),probabilities={str(n):{g:q.tolist() for g,q in f['eval'][n]['q'].items()} for n in (0,1,2,5)}) for f in folds[mode]] for mode in MODES}
    result=dict(schema=p['schema'],protocol_sha256=sha(PROTOCOL),workflow_config=CONFIG,source=source_metadata,
        source_all_ids=source_all,source_model_fit_ids=source_fit,source_normalization_ids={s:d['cal'].tolist() for s,d in partitions.items()},
        source_normalizers={s:dict(center=n.center_.tolist(),scale=n.scale_.tolist()) for s,n in normalizers.items()},
        source_oof=source_oof,personal_calibration_ids=lids[li].tolist(),reserved_current_ids=cids[reserved].tolist(),
        evaluation_ids=evaluation_ids.tolist(),evaluation_labels=ey.tolist(),profiles=profiles,states=states,cells=cells,
        primary_guards=guards,primary_pass=all(guards.values()),roundtrips=roundtrips,artifact_sha256=artifacts,
        source_state_immutable=True,personal_state_immutable=True,default_promoted=False,physical_validation_proven=False,
        completion_proven=False,scope=p['scope'])
    RESULT.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8',newline='\n')
    print('4/4 Saved all80 cells, signed primary guards and independent source/profile packages',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    prepare() if args.prepare else run()
