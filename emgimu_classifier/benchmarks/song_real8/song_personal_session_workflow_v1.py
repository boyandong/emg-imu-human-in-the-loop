"""Source-only six-provider fits and same-user recorded-session lifecycle replay."""
import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score,recall_score
from benchmarks.song_real8_study import load_session
from benchmarks.new_bank_v3.emg_window_bank_v1 import factories,GROUPS
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.force_nested_oof import fit_temperature,temperature_probability
from emgimu.feature_bank.document_reliability_v2 import DocumentReliabilityWeightsV2
from emgimu.feature_bank.frozen_emg_provider_bank_v1 import FrozenEmgProviderBankV1
from emgimu.feature_bank.personal_session_workflow_v1 import PersonalSessionWorkflowV1

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PROTOCOL=HERE/'SONG_PERSONAL_SESSION_V1_PROTOCOL.json'
RESULT=HERE/'SONG_PERSONAL_SESSION_V1_RESULTS.json'
OUT=HERE/'song_personal_session_v1'
CLASSES=('fist','index_pinch','neutral','open_hand')
CHANNELS=tuple(f'CH{i+1}' for i in range(8))
PREPROCESSING='song250_causal_hp40_order4_notch50_100_Q30_zero_session_initial_v1'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(values,item):
    ids=np.unique(item['trial']);y=[];features=[]
    for t in ids:
        mask=item['trial']==t;labels=np.unique(item['hand'][mask])
        if len(labels)!=1:raise ValueError('One label per native trial required')
        features.append(values[mask].mean(0));y.append(labels.item())
    return np.stack(features),np.asarray(y),ids


def select(ids,labels,shots,seed):
    ids=np.asarray(ids);y=np.asarray(labels);cal=[]
    if (ids.ndim!=1 or y.shape!=ids.shape or len(set(ids))!=len(ids) or set(y)!=set(CLASSES)
            or isinstance(shots,bool) or not isinstance(shots,(int,np.integer)) or shots<1):
        raise ValueError('Unique aligned native trials, all four classes and positive integer shots required')
    for c in CLASSES:
        candidates=sorted(np.flatnonzero(y==c),key=lambda i:hashlib.sha256(f'{seed}|{ids[i]}'.encode()).digest())
        if len(candidates)<=shots:raise ValueError('Independent calibration and evaluation trials required')
        cal.extend(candidates[:shots])
    cal=np.array(sorted(cal),dtype=int);evaluation=np.flatnonzero(~np.isin(np.arange(len(ids)),cal))
    return cal,evaluation


def score(y,q):
    classes=np.asarray(CLASSES);y=np.asarray(y);q=np.asarray(q)
    if not len(y) or q.shape!=(len(y),4) or not np.isfinite(q).all() or np.any(q<0) or not np.allclose(q.sum(1),1.,rtol=0,atol=1e-12):
        raise ValueError('Aligned finite four-class probabilities required')
    truth=np.array([CLASSES.index(c) for c in y]);prediction=classes[q.argmax(1)]
    return {'trials':len(y),'macro_f1':float(f1_score(y,prediction,labels=classes,average='macro',zero_division=0)),
        'accuracy':float(np.mean(y==prediction)),'log_loss':float(-np.log(np.maximum(q[np.arange(len(y)),truth],1e-15)).mean()),
        'brier':float(np.mean((q-np.eye(4)[truth])**2)),
        'per_class_f1':dict(zip(CLASSES,map(float,f1_score(y,prediction,labels=classes,average=None,zero_division=0)))),
        'recall':dict(zip(CLASSES,map(float,recall_score(y,prediction,labels=classes,average=None,zero_division=0))))}


def fit_source(item):
    fitted=factories();fitted['F0'][0].rest_label='neutral';models={};features={};q={};metadata={}
    for name,fs in fitted.items():
        columns=[]
        for family in fs:
            family.fit(item['batch'],item['hand']);x,y,ids=aggregate(family.transform(item['batch']),item);columns.append(x)
        x=np.concatenate(columns,axis=1);scaler=StandardScaler().fit(x)
        model=LogisticRegression(C=1,class_weight='balanced',max_iter=2000,random_state=20261010).fit(scaler.transform(x),y)
        if tuple(model.classes_)!=CLASSES or model.n_iter_.max()>=2000:raise ValueError('Source convergence/class contract failed')
        models[name]=(scaler,model);features[name]=scaler.transform(x)
        metadata[name]={'dimension':x.shape[1],'iterations':model.n_iter_.tolist()}
    return fitted,models,metadata


def read(fitted,models,item):
    features={};q={}
    for name,fs in fitted.items():
        columns=[]
        for family in fs:
            x,y,ids=aggregate(family.transform(item['batch']),item);columns.append(x)
        scaler,model=models[name];z=scaler.transform(np.concatenate(columns,axis=1));features[name]=z;q[name]=model.predict_proba(z)
    return features,q,y,ids


def windows(item,trials):
    mask=np.isin(item['trial'],trials);ids=item['trial'][mask];offsets=np.zeros(len(ids),dtype=int)
    for t in np.unique(ids):offsets[ids==t]=np.arange(np.sum(ids==t))
    return item['batch'].take(np.flatnonzero(mask)),ids,offsets


def prepare():
    if PROTOCOL.exists():raise FileExistsError('Protocol already frozen')
    old=json.loads((HERE/'SONG_F0_STREAM_V1_PROTOCOL.json').read_text(encoding='utf8'))
    paths=['benchmarks/song_real8/song_personal_session_workflow_v1.py','benchmarks/song_real8_study.py',
        'benchmarks/new_bank_v3/emg_window_bank_v1.py','src/emgimu/feature_bank/personal_session_workflow_v1.py',
        'src/emgimu/feature_bank/frozen_emg_provider_bank_v1.py','src/emgimu/feature_bank/available_bank_fusion_v1.py',
        'src/emgimu/feature_bank/calibration.py','src/emgimu/feature_bank/document_session_v3.py',
        'src/emgimu/feature_bank/session_shift_summary.py','src/emgimu/feature_bank/document_quality_v3.py',
        'src/emgimu/feature_bank/activation_profile.py','src/emgimu/feature_bank/document_reliability_v2.py',
        'src/emgimu/feature_bank/new_bank_v2.py','src/emgimu/feature_bank/new_bank_v1.py',
        'src/emgimu/feature_bank/document_scale_v3.py','src/emgimu/feature_bank/spec_spatial_v3.py',
        'src/emgimu/feature_bank/document_ces_v3.py','src/emgimu/feature_bank/document_spectral_v3.py',
        'src/emgimu/feature_bank/document_temporal_v3.py','src/emgimu/feature_bank/families.py',
        'src/emgimu/feature_bank/quality_observability.py','src/emgimu/feature_bank/affine_spd_anchor.py',
        'src/emgimu/feature_bank/force_nested_oof.py','src/emgimu/feature_bank/core.py',
        'tests/test_personal_session_workflow_v1.py','tests/test_song_personal_session_protocol_v1.py']
    p={'schema':'song_personal_session_v1','source_folder':old['source_folder'],
        'hdf5_sha256':json.loads((HERE/'F9_DOCUMENT_V3_PROTOCOL.json').read_text(encoding='utf8'))['expected_session_sha256'],
        'source_sessions':['S01','S02'],'personal_session':'S03','current_session':'S04','groups':list(GROUPS),
        'classes':list(CLASSES),'long_term_shots_per_class':5,'current_shots_per_class':[0,1,2,5],
        'selection_seed':'20261010_song_personal_session_v1','source_seed':20261010,
        'source_probability_calibration':'Leave-one-recorded-source-session-out; every representation/scaler/model refit. Temperature fit on source OOF only. These are two recordings of one person/day, not independent training users.',
        'source_policy':'Population=softmax(-source OOF calibrated family logloss). Choose n0 in[4,12,24], reliability temperature in[.5,1,2] by mean source-only evaluation logloss across both source-session OOF folds and nested1/2/5-shot calibration. Calibration5/class is reserved from each source fold at all budgets; no target tuning. Source OOF temperatures/prior are reused inside this source policy search; not nested unbiased source-performance evidence.',
        'target_split':'Reserve5 native formal trials/class in each of S03/S04 by SHA256(seed|trial_id). S03 reserved20 form long-term personal profile; S04 nested4/8/20 form session calibration. Identical S04 remaining evaluation trials for all budgets, including0. All previously inspected recordings; retrospective developmental evidence.',
        'outputs':'Shared raw-feature source classifiers, separate long-term and current rest/scale/activation profiles, F7 standardized-feature coordinates, F8 descriptor, F9 observations and checksum-verified user/session packages. Only calibration-based reliability changes classifier fusion. No normalized classifier, trained F7 classifier, quality rejection, DTW/full-bout or anatomical F6 is silently added.',
        'scope':'Six declared EMG window groups and all six leave-provider-out fusion readouts, not the entire document F0-F9 bank. Single operator-identified Song user/day, cued stable trial boundaries, S01-S03 readiness failures, prior S04 inspection. Source/control arms use0 target trials; personal uses20 S03 trials; session/removals use20 S03 plus0/4/8/20 S04. F7/F8/F9 outputs are descriptive context, not validated routing/gating. No cross-day, physical re-donning, whole-action, live accuracy or low-total-burden proof.',
        'source_sha256':{v:sha(ROOT/v) for v in paths},'default_promoted':False}
    PROTOCOL.write_text(json.dumps(p,indent=2)+'\n',encoding='utf8',newline='\n')
    print('Frozen native Song personal/session lifecycle protocol; no target signals loaded',flush=True)


def run():
    if RESULT.exists() or OUT.exists():raise FileExistsError('Existing workflow results will not be overwritten')
    p=json.loads(PROTOCOL.read_text(encoding='utf8'))
    for name,digest in p['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen workflow implementation changed: '+name)
    def load(session):
        item=load_session(Path(p['source_folder'])/f'2026-09-18_{session}',session,filter_mode='causal')
        if item['audit']['sha256']!=p['hdf5_sha256'][session]:raise ValueError('Native recording changed')
        item['batch']=FeatureBatch(item['batch'].emg,250.) # IMU is not an input to this experiment.
        return item
    sources={s:load(s) for s in p['source_sessions']};folds=[]
    print('1/4 Source-only session OOF fits and policy selection',flush=True)
    for held in p['source_sessions']:
        train=sources[next(s for s in p['source_sessions'] if s!=held)]
        fitted,models,_=fit_source(train);features,q,y,ids=read(fitted,models,sources[held])
        if set(ids)&set(train['trial']):raise ValueError('OOF source trial overlap')
        folds.append({'session':held,'features':features,'q':q,'labels':y,'ids':ids,'fit_ids':np.unique(train['trial'])})
    temperatures={k:fit_temperature(np.concatenate([f['q'][k] for f in folds]),
        np.concatenate([np.array([CLASSES.index(c) for c in f['labels']]) for f in folds])) for k in GROUPS}
    for f in folds:f['q']={k:temperature_probability(v,temperatures[k]) for k,v in f['q'].items()}
    population=softmax(-np.array([score(np.concatenate([f['labels'] for f in folds]),np.concatenate([f['q'][k] for f in folds]))['log_loss'] for k in GROUPS]))
    candidates=[]
    for n0 in (4.,12.,24.):
        for tau in (.5,1.,2.):
            losses=[]
            for f in folds:
                reserved,evaluation=select(f['ids'],f['labels'],5,p['selection_seed'])
                for shots in (1,2,5):
                    selected,_=select(f['ids'],f['labels'],shots,p['selection_seed']);ids=f['ids'][selected];y=f['labels'][selected]
                    policy=DocumentReliabilityWeightsV2(CLASSES,GROUPS,tuple(population),n0,tau,'source_oof_only')
                    state=policy.calculate({k:(f['features'][k][selected],y,ids) for k in GROUPS},forbidden_trial_ids=f['fit_ids'])
                    fused=sum(w*f['q'][k][evaluation] for k,w in zip(GROUPS,state['final']))
                    losses.append(score(f['labels'][evaluation],fused)['log_loss'])
            candidates.append({'n0':n0,'temperature':tau,'mean_source_loss':float(np.mean(losses))})
    selected=min(candidates,key=lambda r:(r['mean_source_loss'],r['n0'],r['temperature']))
    joined={'batch':FeatureBatch(np.concatenate([v['batch'].emg for v in sources.values()]),250.),
        'trial':np.concatenate([v['trial'] for v in sources.values()]),'hand':np.concatenate([v['hand'] for v in sources.values()])}
    fitted,models,dimensions=fit_source(joined)
    bank=FrozenEmgProviderBankV1(fitted,models,temperatures,classes=CLASSES,class_names=CLASSES,
        source_trial_ids=np.unique(joined['trial']),source_policy_id=sha(PROTOCOL),sample_rate_hz=250.,window_samples=50,channels=8,
        population=tuple(population),n0=selected['n0'],reliability_temperature=selected['temperature'])
    before=pickle.dumps(bank);workflow=PersonalSessionWorkflowV1(bank,channel_ids=CHANNELS,preprocessing_id=PREPROCESSING,
        rest_label='neutral',quality_options={'line_frequency_hz':50,'pre_highpass_available':False})
    OUT.mkdir();package=OUT/'source_bank.pkl';package.write_bytes(before)
    print('2/4 S03 long-term personal profile and fixed S04 reservation',flush=True)
    long=load('S03');current=load('S04')
    _,ly,lids=aggregate(long['batch'].emg.mean(1),long);li,_=select(lids,ly,5,p['selection_seed'])
    _,cy,cids=aggregate(current['batch'].emg.mean(1),current);reserved,evaluation=select(cids,cy,5,p['selection_seed'])
    evaluation_ids=cids[evaluation];eb,ei,eo=windows(current,evaluation_ids);ey=cy[evaluation]
    kwargs=dict(user_id='Song',observed_channel_ids=CHANNELS,preprocessing_id=PREPROCESSING)
    b,ids,offsets=windows(long,lids[li]);labels=dict(zip(lids[li],ly[li]))
    personal=workflow.enroll_user(b,ids,labels,window_offsets=offsets,session_id='S03',forbidden_evaluation_trials=evaluation_ids,**kwargs)
    personal_path=OUT/'personal_S03.zip';workflow.save_profile(personal,personal_path)
    personal=workflow.load_profile(personal_path,user_id='Song');personal_before=pickle.dumps(personal)
    readout=bank.predict_providers(eb,ei,window_offsets=eo,user_id='Song');q=readout['probabilities']
    if tuple(evaluation_ids)!=readout['trial_ids']:raise ValueError('Native evaluation trial axis differs')
    rows=[];cells=[];arrays={};roundtrips=[]
    print('3/4 Nested session calibration, persistence and six provider removals',flush=True)
    for shots in p['current_shots_per_class']:
        session=None;path=None;calibration_ids=[]
        if shots:
            ci,_=select(cids,cy,shots,p['selection_seed']);calibration_ids=cids[ci].tolist()
            b,ids,offsets=windows(current,cids[ci]);labels=dict(zip(cids[ci],cy[ci]))
            session=workflow.calibrate_session(b,ids,labels,window_offsets=offsets,personal=personal,session_id='S04',
                forbidden_evaluation_trials=evaluation_ids,**kwargs)
            path=OUT/f'session_S04_{shots}shot.zip';workflow.save_profile(session,path)
            restored=workflow.load_profile(path,user_id='Song',session_id='S04',personal=personal)
        original=workflow.predict(eb,ei,window_offsets=eo,personal=personal,session=session,session_id='S04',**kwargs)
        if shots:
            replay=workflow.predict(eb,ei,window_offsets=eo,personal=personal,session=restored,session_id='S04',**kwargs)
            error=float(np.max(abs(original['probabilities']-replay['probabilities'])))
            if error>1e-12:raise ValueError('Session save/load changed prediction')
            roundtrips.append({'shots':shots,'maximum_probability_error':error,'profile_sha256':sha(path)})
        population_q=bank.predict(eb,ei,window_offsets=eo,user_id='Song')['probabilities']
        personal_q=workflow.predict(eb,ei,window_offsets=eo,personal=personal,session_id='S04',**kwargs)['probabilities']
        arms={'F0':q['F0'],'population':population_q,'uniform':np.mean(list(q.values()),axis=0),
            'personal':personal_q,'session':original['probabilities']}
        for omitted in GROUPS:
            arms['session_minus_'+omitted]=workflow.predict(eb,ei,window_offsets=eo,personal=personal,session=session,
                session_id='S04',available=tuple(g for g in GROUPS if g!=omitted),**kwargs)['probabilities']
        for arm,probability in arms.items():
            cells.append({'shots':shots,'arm':arm,'long_term_calibration_trials':0 if arm in ('F0','population','uniform') else 20,
                'current_calibration_trials':4*shots if arm.startswith('session') else 0,**score(ey,probability)})
            for t,y,prob in zip(evaluation_ids,ey,probability):rows.append({'shots':shots,'arm':arm,'trial_id':t,'label':y,**{f'p_{c}':float(prob[i]) for i,c in enumerate(CLASSES)}})
        arrays[f'shots{shots}_quality']=original['quality_observations']
        for name,modes in original['anchor_coordinates'].items():
            for mode,x in modes.items():arrays[f'shots{shots}_{name}_{mode}']=x
        arrays[f'shots{shots}_normalized_emg']=workflow.normalized_view(eb,personal=personal,session=session,session_id='S04',**kwargs).emg
        if pickle.dumps(personal)!=personal_before or pickle.dumps(bank)!=before:raise ValueError('Session/evaluation mutated source or long-term profile')
        print(f'Current{shots}shot/class: {len(evaluation_ids)} fixed evaluation trials, all11 arms complete',flush=True)
    table=OUT/'predictions.csv'
    with table.open('w',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    array_path=OUT/'context_readouts.npz';np.savez_compressed(array_path,**arrays)
    result={'schema':p['schema'],'protocol_sha256':sha(PROTOCOL),'source_bank_path':package.relative_to(ROOT).as_posix(),
        'source_bank_sha256':sha(package),'source_bank_id':bank.bank_id_,'source_trial_ids':np.unique(joined['trial']).tolist(),
        'source_dimensions':dimensions,'temperatures':temperatures,'population':population.tolist(),'source_policy_candidates':candidates,
        'selected_source_policy':selected,'source_oof':[{'session':f['session'],'fit_ids':f['fit_ids'].tolist(),
            'validation_ids':f['ids'].tolist(),'labels':f['labels'].tolist(),'calibrated_probabilities':{k:v.tolist() for k,v in f['q'].items()}} for f in folds],
        'personal_calibration_ids':list(personal.calibration_trials),'personal_profile_path':personal_path.relative_to(ROOT).as_posix(),
        'personal_profile_sha256':sha(personal_path),'reserved_current_ids':cids[reserved].tolist(),
        'evaluation_ids':evaluation_ids.tolist(),'evaluation_labels':ey.tolist(),'cells':cells,'session_roundtrips':roundtrips,
        'prediction_path':table.relative_to(ROOT).as_posix(),'prediction_sha256':sha(table),
        'context_path':array_path.relative_to(ROOT).as_posix(),'context_sha256':sha(array_path),
        'source_state_immutable':True,'personal_state_immutable':True,'scope':p['scope'],
        'default_promoted':False,'physical_validation_proven':False,'completion_proven':False}
    RESULT.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8',newline='\n')
    print('4/4 Saved source-frozen native session workflow, context readouts and all signed results',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    prepare() if args.prepare else run()
