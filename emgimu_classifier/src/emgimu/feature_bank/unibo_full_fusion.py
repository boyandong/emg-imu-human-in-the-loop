"""Native four-channel day/posture fusion with source-day probability calibration."""
from pathlib import Path
import argparse
import csv
import json
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from emgimu.datasets.unibo_physiology import load_chronological_raw_windows, chronological_fold, _as_feature_windows
from emgimu.datasets.unibo_baseline import hierarchical_segment_weights
from .core import FeatureBatch
from .screening import FAMILY_FACTORIES,SEED
from .unibo_study import _metrics
from .force_nested_oof import fit_temperature,temperature_probability
from .calibration import late_fusion

IDS=tuple(n for n in FAMILY_FACTORIES if n!='F3_Ring')


def batch(data):
    return FeatureBatch(data.emg,200.)


def weights(data):
    return hierarchical_segment_weights(_as_feature_windows(data,np.empty((len(data),0),dtype=np.float32)))


def classifier(features,labels,w):
    scaler=StandardScaler().fit(features,sample_weight=w)
    model=LogisticRegression(C=1,class_weight='balanced',max_iter=1000,random_state=SEED).fit(scaler.transform(features),labels,sample_weight=w)
    return scaler,model


def variants(raw,calibrated,qmean,qmin):
    w=np.ones(len(IDS))/len(IDS)
    quality={n:np.ones(len(qmean)) if n=='F9_Quality' else qmin if n in ('F0','F2b_CSP') else qmean for n in IDS}
    specs={'full':(calibrated,quality),'uniform_calibrated':(calibrated,None),'uniform_raw':(raw,None),
        'baseline_F0':({'F0':calibrated['F0']},None),'baseline_F0_raw':({'F0':raw['F0']},None),
        **{f'without_{n}':({k:p for k,p in calibrated.items() if k!=n},quality) for n in IDS}}
    return {n:late_fusion(p,IDS,w,q) for n,(p,q) in specs.items()}


def run(dataset,source_run,output):
    if output.exists():raise FileExistsError(output)
    print('[1/3] loading source days and separate source-day calibration',flush=True)
    raw=load_chronological_raw_windows(dataset,(1,2,3,4,5))
    train,_=chronological_fold(raw,(1,2,3,4,5),5,6)
    inner_train,cal=chronological_fold(raw,(1,2,3,4),5,6)
    if set(inner_train.trial_id)&set(cal.trial_id):raise AssertionError('source-day leakage')
    old_families,_=pickle.loads((source_run/'fitted_states.pkl').read_bytes())
    old_split=json.loads((source_run/'split_trial_ids.json').read_text())
    if set(old_split['train'])!=set(train.trial_id):raise AssertionError('different reused family source trials')
    full_states={};inner_states={};temperatures={};source_predictions={};dimensions={}
    tw,cw=weights(train),weights(cal);iw=weights(inner_train)
    for i,n in enumerate(IDS):
        print(f'[2/3] provider {i+1}/{len(IDS)} {n}: fit days 1-4; temperature day 5; final classifier days 1-5',flush=True)
        f=FAMILY_FACTORIES[n]();a=f.fit_transform(batch(inner_train),inner_train.labels)
        s,m=classifier(a,inner_train.labels,iw);before=pickle.dumps((f,s,m))
        cp=m.predict_proba(s.transform(f.transform(batch(cal))))
        if pickle.dumps((f,s,m))!=before:raise AssertionError('source calibration mutated fitted family')
        inner_states[n]=(f,s,m);source_predictions[n]=cp;temperatures[n]=fit_temperature(cp,cal.labels)
        f=old_families[n];before=pickle.dumps(f);a=f.transform(batch(train))
        if pickle.dumps(f)!=before:raise AssertionError('reused family changed')
        s,m=classifier(a,train.labels,tw);full_states[n]=(f,s,m);dimensions[n]=a.shape[1]
    print('[3/3] saving source-fit package; target days never opened',flush=True);output.mkdir(parents=True)
    with (output/'source_fitted_states.pkl').open('wb') as h:pickle.dump((full_states,inner_states),h)
    np.savez_compressed(output/'source_calibration_predictions.npz',labels=cal.labels,trials=cal.trial_id,
        subjects=cal.subject_id,days=cal.session_id,posture=cal.posture,**source_predictions)
    (output/'source_split_trial_ids.json').write_text(json.dumps({'inner_train':sorted(set(inner_train.trial_id)),
        'calibration':sorted(set(cal.trial_id)),'full_train':sorted(set(train.trial_id))},indent=2))
    (output/'run_manifest.json').write_text(json.dumps({'source_run':source_run.name,'source_days':[1,2,3,4,5],
        'inner_train_days':[1,2,3,4],'probability_calibration_day':5,'temperatures':temperatures,'families':IDS,
        'dimensions':dimensions,'sample_rate_hz':200,'native_channels':4,'target_calibration_trials':0,
        'ring_or_oracle_posture_used':False,'source_calibration_weights':'source hierarchical window weights in classifier; temperature unweighted windows',
        'temperature_bounds':[.25,4],'scope':'source-day heldout probability calibration; not full nested OOF'},indent=2))
    print(json.dumps({'status':'ok','source_windows':len(train),'source_calibration_windows':len(cal)}))


def evaluate(dataset,package,output,phase):
    if output.exists():raise FileExistsError(output)
    manifest=json.loads((package/'run_manifest.json').read_text());states,_=pickle.loads((package/'source_fitted_states.pkl').read_bytes())
    before=pickle.dumps(states);days=(6,) if phase=='validation' else (7,8)
    print(f'[1/2] loading target days {days} and transforming fixed source providers',flush=True)
    target=load_chronological_raw_windows(dataset,days);raw={};calibrated={}
    for n,(f,s,m) in states.items():
        x=f.transform(batch(target));raw[n]=m.predict_proba(s.transform(x))
        calibrated[n]=temperature_probability(raw[n],manifest['temperatures'][n])
        if n=='F9_Quality':qmean=np.clip(x[:,-3],0,1);qmin=np.clip(x[:,-2],0,1)
    probabilities=variants(raw,calibrated,qmean,qmin);rows=[];w=weights(target)
    cells=[('ALL','ALL',np.ones(len(target),bool))]
    cells.extend(('ALL',f'posture_{p}',target.posture==p) for p in range(1,5))
    cells.extend((u,'ALL',target.subject_id==u) for u in sorted(set(target.subject_id)))
    cells.extend(('ALL',str(d),target.session_id==d) for d in sorted(set(target.session_id)))
    cells.extend((u,f'{d}_posture_{p}',(target.subject_id==u)&(target.session_id==d)&(target.posture==p))
        for u in sorted(set(target.subject_id)) for d in sorted(set(target.session_id)) for p in range(1,5))
    for method,p in probabilities.items():
        for u,condition,mask in cells:
            if not mask.any():continue
            rows.append({'dataset':'unibo_inail','phase':phase,'subject':u,'condition':condition,'shots_per_class':0,
                'method':method,'feature_bank':'|'.join(IDS),**_metrics(target.labels[mask],p[mask],w[mask])})
    if pickle.dumps(states)!=before:raise AssertionError('target changed source states')
    print('[2/2] saving full-bank target predictions and cell metrics',flush=True);output.mkdir(parents=True)
    with (output/'ablation_full_bank.csv').open('w',newline='',encoding='utf-8') as h:
        writer=csv.DictWriter(h,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    np.savez_compressed(output/'heldout_predictions.npz',labels=target.labels,trials=target.trial_id,subjects=target.subject_id,
        days=target.session_id,posture=target.posture,**probabilities)
    (output/'run_manifest.json').write_text(json.dumps({**manifest,'source_package':str(package),'phase':phase,
        'target_days':days,'classifier_or_family_fit':False,'population_weights':'uniform',
        'quality':'F0/CSP min, robust signal providers mean, F9 1'},indent=2))
    source_split=json.loads((package/'source_split_trial_ids.json').read_text())
    target_ids=sorted(set(target.trial_id))
    if set(source_split['full_train'])&set(target_ids):raise AssertionError('target trial leakage')
    (output/'split_trial_ids.json').write_text(json.dumps({'train':source_split['full_train'],'test':target_ids},indent=2))
    print(json.dumps({'status':'ok','rows':len(rows),'target_windows':len(target)}))


def audit_source(dataset,package):
    manifest=json.loads((package/'run_manifest.json').read_text())
    states,inner=pickle.loads((package/'source_fitted_states.pkl').read_bytes());before=pickle.dumps((states,inner))
    split=json.loads((package/'source_split_trial_ids.json').read_text())
    a,b=set(split['inner_train']),set(split['calibration'])
    if a&b or a|b!=set(split['full_train']):raise AssertionError('invalid source calibration partition')
    cal=load_chronological_raw_windows(dataset,(manifest['probability_calibration_day'],));error=0.
    with np.load(package/'source_calibration_predictions.npz',allow_pickle=False) as saved:
        for key,value in (('labels',cal.labels),('trials',cal.trial_id),('subjects',cal.subject_id),('days',cal.session_id),('posture',cal.posture)):
            np.testing.assert_array_equal(value,saved[key])
        for n,(f,s,m) in inner.items():
            p=m.predict_proba(s.transform(f.transform(batch(cal))))
            np.testing.assert_allclose(p,saved[n],atol=1e-8,rtol=1e-7)
            np.testing.assert_allclose(fit_temperature(p,cal.labels),manifest['temperatures'][n])
            error=max(error,float(np.abs(p-saved[n]).max()))
    if pickle.dumps((states,inner))!=before:raise AssertionError('source replay changed fitted states')
    result={'status':'ok','source_probability_arrays_checked':len(inner),'max_probability_error':error,
        'source_temperature_recomputed':True,'classifier_or_family_fit':False,'target_days_loaded':False}
    (package/'source_replay_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


def replay(dataset,package,output):
    manifest=json.loads((output/'run_manifest.json').read_text());states,_=pickle.loads((package/'source_fitted_states.pkl').read_bytes())
    before=pickle.dumps(states);target=load_chronological_raw_windows(dataset,manifest['target_days']);raw={};calibrated={}
    for n,(f,s,m) in states.items():
        x=f.transform(batch(target));raw[n]=m.predict_proba(s.transform(x));calibrated[n]=temperature_probability(raw[n],manifest['temperatures'][n])
        if n=='F9_Quality':qmean=np.clip(x[:,-3],0,1);qmin=np.clip(x[:,-2],0,1)
    ps=variants(raw,calibrated,qmean,qmin);error=0.
    with np.load(output/'heldout_predictions.npz',allow_pickle=False) as saved:
        for key,value in (('labels',target.labels),('trials',target.trial_id),('subjects',target.subject_id),('days',target.session_id),('posture',target.posture)):
            np.testing.assert_array_equal(value,saved[key])
        if set(saved.files)-{'labels','trials','subjects','days','posture'}!=set(ps):raise AssertionError('missing variant coverage')
        for n,p in ps.items():
            np.testing.assert_allclose(p,saved[n],rtol=1e-7,atol=1e-8);error=max(error,float(np.abs(p-saved[n]).max()))
    if pickle.dumps(states)!=before:raise AssertionError('replay changed source states')
    result={'status':'ok','prediction_arrays_checked':len(ps),'max_probability_error':error,'classifier_or_family_fit':False}
    (output/'replay_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',type=Path);p.add_argument('package',type=Path)
    p.add_argument('--source-run',type=Path);p.add_argument('--output',type=Path);p.add_argument('--phase',choices=('validation','final'),default='validation');p.add_argument('--replay',action='store_true')
    p.add_argument('--audit-source',action='store_true')
    a=p.parse_args()
    if a.source_run:run(a.dataset,a.source_run,a.package)
    elif a.audit_source:audit_source(a.dataset,a.package)
    elif a.replay:replay(a.dataset,a.package,a.output)
    elif a.output:evaluate(a.dataset,a.package,a.output,a.phase)
    else:p.error('--output is required for target evaluation')
