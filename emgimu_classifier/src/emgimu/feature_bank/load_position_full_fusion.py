"""EMG-only load/position full fusion with source-condition OOF calibration."""
from pathlib import Path
import argparse
import csv
import json
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from emgimu.datasets.emg_fmg import load_emg_fmg_windows
from .screening import FAMILY_FACTORIES,SEED
from .emg_fmg_study import _metrics,_trial_probabilities
from .force_nested_oof import fit_temperature,temperature_probability
from .wearing_full_fusion import IDS,variants


def predict(train,target,name):
    if set(train.trials)&set(target.trials):raise AssertionError('trial leakage')
    family=FAMILY_FACTORIES[name]();a=family.fit_transform(train.batch,train.labels)
    scaler=StandardScaler().fit(a,sample_weight=train.sample_weight)
    model=LogisticRegression(C=1,class_weight='balanced',max_iter=1000,random_state=SEED).fit(scaler.transform(a),train.labels,sample_weight=train.sample_weight)
    state=(family,scaler,model);before=pickle.dumps(state)
    x=family.transform(target.batch);p=model.predict_proba(scaler.transform(x));p,y,loads,positions=_trial_probabilities(p,target)
    if pickle.dumps(state)!=before:raise AssertionError('target changed fitted state')
    return p,y,loads,positions,state,x


def run(archive,output,phase):
    if output.exists():raise FileExistsError(output)
    users=(1,2,3) if phase=='validation' else (4,5,6)
    states={};saved={};temperatures={};splits=[];rows=[]
    for i,user in enumerate(users):
        print(f'[{i+1}/3] subject {user}: streaming EMG-only trials',flush=True)
        data=load_emg_fmg_windows(archive,subjects=(user,),loads=(0,250,500,750,1000),positions=range(1,9))
        for scenario,mask in (('load_shift',data.loads==0),('position_shift',data.positions==1)):
            train=data.take(np.flatnonzero(mask));target=data.take(np.flatnonzero(~mask))
            st=np.unique(train.trials);sy=np.asarray([np.unique(train.labels[train.trials==t]).item() for t in st])
            oof={n:np.zeros((len(st),4)) for n in IDS};folds=[]
            groups=((1,2),(3,4),(5,6),(7,8)) if scenario=='load_shift' else ((0,),(250,),(500,),(750,),(1000,))
            factor=train.positions if scenario=='load_shift' else train.loads
            print(f'  {scenario}: {len(groups)} source-condition folds; nine-provider target comparison',flush=True)
            for fold,values in enumerate(groups):
                held=np.isin(factor,values);a=train.take(np.flatnonzero(~held));b=train.take(np.flatnonzero(held))
                folds.append({'fold':fold,'train':sorted(set(a.trials)),'validation':sorted(set(b.trials))})
                for n in IDS:
                    p,y,_,_,state,_=predict(a,b,n);it=np.unique(b.trials)
                    index=np.asarray([np.flatnonzero(st==t).item() for t in it]);np.testing.assert_array_equal(sy[index],y)
                    oof[n][index]=p;states[(user,scenario,'oof',fold,n)]=state
            raw={};calibrated={};prefix=f'{user}_{scenario}'
            for n in IDS:
                p,y,loads,positions,state,x=predict(train,target,n);states[(user,scenario,n)]=state
                temp=fit_temperature(oof[n],sy);temperatures[f'{prefix}_{n}']=temp
                raw[n]=p;calibrated[n]=temperature_probability(p,temp);saved[f'{prefix}_oof_{n}']=oof[n]
                if n=='F9_Quality':
                    q,_,_,_=_trial_probabilities(x,target);qmean=np.clip(q[:,-3],0,1);qmin=np.clip(q[:,-2],0,1)
            saved.update({f'{prefix}_source_labels':sy,f'{prefix}_source_trials':st,f'{prefix}_labels':y,f'{prefix}_trials':np.unique(target.trials)})
            condition=loads if scenario=='load_shift' else positions
            for method,p in variants(raw,calibrated,qmean,qmin).items():
                saved[f'{prefix}_prediction_{method}']=p
                for value in ('ALL',*sorted(set(condition))):
                    m=np.ones(len(y),bool) if value=='ALL' else condition==value
                    rows.append({'dataset':'emg_fmg','phase':phase,'subject':user,'scenario':scenario,
                        'condition':str(value),'shots_per_class':0,'method':method,'feature_bank':'|'.join(IDS),**_metrics(y[m],p[m])})
            splits.append({'subject':user,'scenario':scenario,'train':st.tolist(),'test':np.unique(target.trials).tolist(),'source_oof':folds})
    aggregate=[]
    for scenario,condition,method in sorted({(r['scenario'],r['condition'],r['method']) for r in rows}):
        selected=[r for r in rows if (r['scenario'],r['condition'],r['method'])==(scenario,condition,method)]
        pc=[json.loads(r['per_class_f1_json']) for r in selected]
        aggregate.append({**selected[0],'subject':'ALL',**{k:float(np.mean([r[k] for r in selected])) for k in ('macro_f1','accuracy','log_loss','brier','ece')},
            'per_class_f1_json':json.dumps({k:float(np.mean([p[k] for p in pc])) for k in pc[0]})})
    output.mkdir(parents=True)
    with (output/'ablation_full_bank.csv').open('w',newline='',encoding='utf-8') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows+aggregate)
    with (output/'fitted_states.pkl').open('wb') as h:pickle.dump(states,h)
    np.savez_compressed(output/'heldout_predictions.npz',**saved)
    (output/'scenario_split_trial_ids.json').write_text(json.dumps(splits,indent=2))
    (output/'run_manifest.json').write_text(json.dumps({'phase':phase,'families':IDS,'temperatures':temperatures,
        'dimensions':{n:len(states[(users[0],'load_shift',n)][0].feature_names) for n in IDS},
        'source_oof':'load: position pairs at 0g; position: individual loads at position1; every family/scaler/classifier refit per fold',
        'temperature_unit':'source held-condition trial-mean probabilities','target_calibration_trials':0,
        'emg_channels':'9-16 only; FMG never used','sample_rate_hz':2000,'aggregation':'mean per-user metrics',
        'trial_crop':'central 9 seconds; eight sparse windows as original adapter','quality':'F0/CSP min, robust providers mean, F9 1',
        'scope':'native external load/limb-position protocol, not voluntary contraction level'},indent=2))
    print(json.dumps({'status':'ok','rows':len(rows+aggregate)}))


def replay(archive,output):
    states=pickle.loads((output/'fitted_states.pkl').read_bytes());before=pickle.dumps(states)
    splits=json.loads((output/'scenario_split_trial_ids.json').read_text());manifest=json.loads((output/'run_manifest.json').read_text())
    checked=0;error=0.
    with np.load(output/'heldout_predictions.npz',allow_pickle=False) as saved:
        for user in sorted({s['subject'] for s in splits}):
            data=load_emg_fmg_windows(archive,subjects=(user,),loads=(0,250,500,750,1000),positions=range(1,9))
            for split in (s for s in splits if s['subject']==user):
                scenario=split['scenario'];prefix=f'{user}_{scenario}'
                if set(split['train'])&set(split['test']):raise AssertionError('target leakage')
                target=data.take(np.flatnonzero(np.isin(data.trials,split['test'])));raw={};calibrated={}
                st=saved[f'{prefix}_source_trials'];sy=saved[f'{prefix}_source_labels']
                for n in IDS:
                    oof=np.zeros((len(st),4))
                    for fold in split['source_oof']:
                        if set(fold['train'])&set(fold['validation']) or set(fold['train'])|set(fold['validation'])!=set(st):raise AssertionError('source fold leakage')
                        d=data.take(np.flatnonzero(np.isin(data.trials,fold['validation'])))
                        f,s,m=states[(user,scenario,'oof',fold['fold'],n)]
                        p,y,_,_=_trial_probabilities(m.predict_proba(s.transform(f.transform(d.batch))),d)
                        index=np.asarray([np.flatnonzero(st==t).item() for t in np.unique(d.trials)]);np.testing.assert_array_equal(sy[index],y);oof[index]=p
                    np.testing.assert_allclose(oof,saved[f'{prefix}_oof_{n}'],rtol=1e-7,atol=1e-8);checked+=1
                    temp=fit_temperature(oof,sy);np.testing.assert_allclose(temp,manifest['temperatures'][f'{prefix}_{n}'])
                    f,s,m=states[(user,scenario,n)];x=f.transform(target.batch)
                    p,y,_,_=_trial_probabilities(m.predict_proba(s.transform(x)),target)
                    np.testing.assert_array_equal(y,saved[f'{prefix}_labels']);np.testing.assert_array_equal(np.unique(target.trials),saved[f'{prefix}_trials'])
                    raw[n]=p;calibrated[n]=temperature_probability(p,temp)
                    if n=='F9_Quality':q,_,_,_=_trial_probabilities(x,target);qmean=np.clip(q[:,-3],0,1);qmin=np.clip(q[:,-2],0,1)
                for method,p in variants(raw,calibrated,qmean,qmin).items():
                    expected=saved[f'{prefix}_prediction_{method}'];np.testing.assert_allclose(p,expected,rtol=1e-7,atol=1e-8)
                    error=max(error,float(np.abs(p-expected).max()));checked+=1
    if pickle.dumps(states)!=before:raise AssertionError('replay changed fitted state')
    result={'status':'ok','probability_arrays_checked':checked,'max_target_probability_error':error,
        'source_oof_temperatures_recomputed':True,'classifier_or_family_fit':False}
    (output/'replay_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--phase',choices=('validation','final'),default='validation');p.add_argument('--replay',action='store_true')
    a=p.parse_args();replay(a.archive,a.output) if a.replay else run(a.archive,a.output,a.phase)
