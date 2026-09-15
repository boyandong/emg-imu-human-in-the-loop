"""Personal amplitude normalization with source-only family/classifier fitting."""
from pathlib import Path
import argparse
import csv
import json
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from emgimu.datasets.epn612 import load_epn612_windows
from .core import FeatureBatch
from .calibration import PersonalNormalizer
from .epn_study import aggregate_trials, _metrics
from .screening import FAMILY_FACTORIES, SEED


def run(archive:Path,output:Path,phase:str)->None:
    if output.exists():raise FileExistsError(output)
    users=(16,17,18) if phase=='validation' else (19,20,21)
    print(f'[1/3] loading EPN {phase}',flush=True)
    train=load_epn612_windows(archive,users=range(1,16));target=load_epn612_windows(archive,users=users)
    population=PersonalNormalizer().fit(train.batch,train.labels)
    normalized=np.empty_like(train.batch.emg);normalizers={}
    for user in np.unique(train.users):
        idx=np.flatnonzero(train.users==user)
        normalizer=PersonalNormalizer().fit(train.batch.take(idx),train.labels[idx])
        normalized[idx]=normalizer.transform(train.batch.take(idx)).emg;normalizers[int(user)]=normalizer
    normalized_batch=FeatureBatch(normalized,train.batch.sample_rate_hz,train.batch.imu)
    models={}
    for method,batch in (('raw',train.batch),('personal_normalization',normalized_batch)):
        families=[FAMILY_FACTORIES[name]() for name in ('F0','F3_Ring')]
        x,y,_,source_trials,_=aggregate_trials(np.concatenate([f.fit_transform(batch,train.labels) for f in families],1),train)
        scaler=StandardScaler().fit(x)
        model=LogisticRegression(C=1,class_weight='balanced',max_iter=1000,random_state=SEED).fit(scaler.transform(x),y)
        models[method]=(families,scaler,model)
    rows=[];splits=[];predictions={};personal_states={}
    print('[2/3] comparing raw/normalized predictions on identical remaining trials',flush=True)
    for user in users:
        idx=np.flatnonzero(target.users==user);data=target.take(idx)
        for shots in (0,1,2,5):
            rng=np.random.default_rng(SEED+user);chosen=[]
            for label in range(6):
                trials=np.unique(data.trials[data.labels==label]);chosen.extend(rng.permutation(trials)[:shots])
            cal=np.flatnonzero(np.isin(data.trials,chosen))
            normalizer=population if shots==0 else PersonalNormalizer().fit(data.batch.take(cal),data.labels[cal])
            personal_states[(user,shots)]=normalizer
            for method,(families,scaler,model) in models.items():
                batch=data.batch if method=='raw' else normalizer.transform(data.batch)
                x,y,_,trials,w=aggregate_trials(np.concatenate([f.transform(batch) for f in families],1),data)
                ev=~np.isin(trials,chosen)
                if set(trials[ev])&set(chosen):raise AssertionError('trial leakage')
                p=model.predict_proba(scaler.transform(x[ev]))
                predictions[f'{user}_{shots}_{method}']=p
                rows.append({'dataset':'epn612','phase':phase,'subject':user,'condition':'cross_user',
                    'shots_per_class':shots,'feature_bank':'F0+F3_Ring','method':method,**_metrics(y[ev],p,w[ev])})
            splits.append({'user':user,'shots':shots,'calibration':list(chosen),'evaluation':trials[ev].tolist()})
    for shots in (0,1,2,5):
        for method in models:
            selected=[r for r in rows if r['shots_per_class']==shots and r['method']==method]
            rows.append({**selected[0],'subject':'ALL',**{k:float(np.mean([r[k] for r in selected])) for k in ('macro_f1','accuracy','log_loss','brier','ece')},'per_class_f1_json':'mean per-user aggregation'})
    print('[3/3] saving normalization states and trial evidence',flush=True);output.mkdir(parents=True)
    with (output/'calibration_curve.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/'split_trial_ids.json').write_text(json.dumps(splits,indent=2),encoding='utf-8')
    (output/'run_manifest.json').write_text(json.dumps({'phase':phase,'seed':SEED,'source_trials':source_trials.tolist(),
        'source_normalizer_fit':'source-user training trials only','target_normalizer_fit':'explicit calibration trials only; zero-shot source population',
        'center':'median rest','scale':'95th percentile absolute active magnitude','classifier_C':1},indent=2),encoding='utf-8')
    with (output/'fitted_states.pkl').open('wb') as handle:pickle.dump((models,normalizers,population,personal_states),handle)
    np.savez_compressed(output/'heldout_predictions.npz',**predictions)
    print(json.dumps({'status':'ok','rows':len(rows)}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('archive',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=('validation','final'),required=True);args=parser.parse_args();run(args.archive,args.output,args.phase)
