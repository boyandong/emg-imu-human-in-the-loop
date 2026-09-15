"""Before-wearing trial OOF calibration and after-wearing full-bank ablations."""
from pathlib import Path
import argparse
import csv
import json
import pickle
import numpy as np
from emgimu.datasets.electrode_shift import load_electrode_shift_windows, PATH_RE
from .force_nested_oof import SubjectWindows, predict, fit_temperature, temperature_probability
from .force_full_fusion import aggregate
from .screening import FAMILY_FACTORIES
from .electrode_shift_study import _metrics
from .calibration import late_fusion

IDS=tuple(FAMILY_FACTORIES)


def load(archive,user,domains):
    d=load_electrode_shift_windows(archive,subjects=(user,),domains=domains)
    return SubjectWindows(d.batch,d.labels,d.subjects,d.trials)


def variants(raw,calibrated,qmean,qmin):
    weights=np.ones(len(IDS))/len(IDS)
    quality={n:np.ones(len(qmean)) if n=='F9_Quality' else qmin if n in ('F0','F2b_CSP') else qmean for n in IDS}
    specs={'full':(calibrated,quality),'uniform_calibrated':(calibrated,None),'uniform_raw':(raw,None),
        'baseline_F0':({'F0':calibrated['F0']},None),'baseline_F0_raw':({'F0':raw['F0']},None),
        **{f'without_{n}':({k:p for k,p in calibrated.items() if k!=n},quality) for n in IDS}}
    return {n:late_fusion(p,IDS,weights,q) for n,(p,q) in specs.items()}


def run(archive,output,phase):
    if output.exists():raise FileExistsError(output)
    users=(15,16,17) if phase=='validation' else (18,19,20)
    states={};saved={};splits={};temperatures={};rows=[];collected={}
    for i,user in enumerate(users):
        print(f'[{i+1}/3] subject {user}: before-wearing trial OOF and frozen after-domain fusion',flush=True)
        train=load(archive,user,('training',));target=load(archive,user,('trial_1','trial_2','trial_3','trial_4'))
        _,sy,_,st=aggregate(np.zeros((len(train.labels),1)),train)
        oof={n:np.zeros((len(st),5)) for n in IDS};folds=[]
        reps=np.asarray([int(PATH_RE.fullmatch(t)['rep']) for t in train.trials])
        for rep in sorted(set(reps)):
            a=train.take(np.flatnonzero(reps!=rep));b=train.take(np.flatnonzero(reps==rep))
            folds.append({'repetition':int(rep),'train':sorted(set(a.trials)),'validation':sorted(set(b.trials))})
            for n in IDS:
                p,y,_,trials,state=predict(a,b,n,subject_disjoint=False)
                index=np.asarray([np.flatnonzero(st==t).item() for t in trials]);np.testing.assert_array_equal(sy[index],y)
                oof[n][index]=p;states[(user,'oof',int(rep),n)]=state
        raw={};calibrated={}
        for n in IDS:
            p,y,_,trials,state=predict(train,target,n,subject_disjoint=False)
            states[(user,n)]=state;temp=fit_temperature(oof[n],sy);temperatures[f'{user}_{n}']=temp
            raw[n]=p;calibrated[n]=temperature_probability(p,temp);saved[f'{user}_oof_{n}']=oof[n]
            if n=='F9_Quality':
                features,_,_,_=aggregate(state[0].transform(target.batch),target)
                qmean=np.clip(features[:,-3],0,1);qmin=np.clip(features[:,-2],0,1)
        saved.update({f'{user}_source_labels':sy,f'{user}_source_trials':st,f'{user}_labels':y,f'{user}_trials':trials})
        domains=np.asarray([PATH_RE.fullmatch(t)['domain'] for t in trials])
        for method,p in variants(raw,calibrated,qmean,qmin).items():
            saved[f'{user}_prediction_{method}']=p
            collected.setdefault(method,[]).append((user,y,p,domains))
            for domain in ('ALL',*sorted(set(domains))):
                mask=np.ones(len(y),bool) if domain=='ALL' else domains==domain
                rows.append({'dataset':'libemg_electrode_shift','phase':phase,'subject':user,'condition':domain,
                    'shots_per_class':0,'method':method,'feature_bank':'|'.join(IDS),**_metrics(y[mask],p[mask])})
        splits[str(user)]={'train':st.tolist(),'test':trials.tolist(),'oof':folds}
    for method,values in collected.items():
        y=np.concatenate([v[1] for v in values]);p=np.concatenate([v[2] for v in values]);domains=np.concatenate([v[3] for v in values])
        for domain in ('ALL',*sorted(set(domains))):
            mask=np.ones(len(y),bool) if domain=='ALL' else domains==domain
            rows.append({'dataset':'libemg_electrode_shift','phase':phase,'subject':'ALL','condition':domain,
                'shots_per_class':0,'method':method,'feature_bank':'|'.join(IDS),**_metrics(y[mask],p[mask])})
    output.mkdir(parents=True)
    with (output/'ablation_full_bank.csv').open('w',newline='',encoding='utf-8') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (output/'fitted_states.pkl').open('wb') as h:pickle.dump(states,h)
    np.savez_compressed(output/'heldout_predictions.npz',**saved)
    (output/'split_trial_ids.json').write_text(json.dumps(splits,indent=2))
    (output/'run_manifest.json').write_text(json.dumps({'phase':phase,'families':IDS,'temperatures':temperatures,
        'probability_fit':'before-wearing five repetition-held-out trial folds; each family/scaler/classifier refit within fold',
        'target_calibration_trials':0,'population_weights':'uniform','quality':'F0/CSP min; other signal providers mean; F9 1',
        'F6_F7_F8':'unavailable or intentionally absent in strict zero-target-calibration protocol',
        'scope':'subject-specific native five-class wearing benchmark; fixed nine-provider package, no final-user selection'},indent=2))
    print(json.dumps({'status':'ok','rows':len(rows)}))


def replay(archive,output):
    states=pickle.loads((output/'fitted_states.pkl').read_bytes());before=pickle.dumps(states)
    splits=json.loads((output/'split_trial_ids.json').read_text());manifest=json.loads((output/'run_manifest.json').read_text())
    checked=0;error=0.
    with np.load(output/'heldout_predictions.npz',allow_pickle=False) as saved:
        for user_key,split in splits.items():
            user=int(user_key);train=load(archive,user,('training',));target=load(archive,user,('trial_1','trial_2','trial_3','trial_4'))
            if set(split['train'])&set(split['test']):raise AssertionError('target leakage')
            raw={};calibrated={}
            for n in IDS:
                family,scaler,model=states[(user,n)];b,y,_,trials=aggregate(family.transform(target.batch),target)
                np.testing.assert_array_equal(saved[f'{user}_labels'],y);np.testing.assert_array_equal(saved[f'{user}_trials'],trials)
                raw[n]=model.predict_proba(scaler.transform(b))
                sy=saved[f'{user}_source_labels'];st=saved[f'{user}_source_trials'];oof=np.zeros((len(st),5))
                for fold in split['oof']:
                    if set(fold['train'])&set(fold['validation']) or set(fold['train'])|set(fold['validation'])!=set(split['train']):
                        raise AssertionError('invalid source OOF partition')
                    d=train.take(np.flatnonzero(np.isin(train.trials,fold['validation'])))
                    f,s,m=states[(user,'oof',fold['repetition'],n)];x,iy,_,it=aggregate(f.transform(d.batch),d)
                    index=np.asarray([np.flatnonzero(st==t).item() for t in it]);np.testing.assert_array_equal(sy[index],iy)
                    oof[index]=m.predict_proba(s.transform(x))
                np.testing.assert_allclose(oof,saved[f'{user}_oof_{n}'],rtol=1e-7,atol=1e-8);checked+=1
                temp=fit_temperature(oof,sy);np.testing.assert_allclose(temp,manifest['temperatures'][f'{user}_{n}'])
                calibrated[n]=temperature_probability(raw[n],temp)
                if n=='F9_Quality':qmean=np.clip(b[:,-3],0,1);qmin=np.clip(b[:,-2],0,1)
            for method,p in variants(raw,calibrated,qmean,qmin).items():
                expected=saved[f'{user}_prediction_{method}'];np.testing.assert_allclose(p,expected,rtol=1e-7,atol=1e-8)
                error=max(error,float(np.abs(p-expected).max()));checked+=1
    if pickle.dumps(states)!=before:raise AssertionError('replay changed fitted state')
    result={'status':'ok','probability_arrays_checked':checked,'max_target_probability_error':error,
        'classifier_or_family_fit':False,'source_oof_temperatures_recomputed':True}
    (output/'replay_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--phase',choices=('validation','final'),default='validation');p.add_argument('--replay',action='store_true')
    a=p.parse_args();replay(a.archive,a.output) if a.replay else run(a.archive,a.output,a.phase)
