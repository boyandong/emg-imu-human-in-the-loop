"""Trial-level DTW of ordered RMS envelopes; templates never use evaluation trials."""
from pathlib import Path
import argparse
import csv
import json
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from emgimu.datasets.semg_manus import load_semg_manus_windows
from .core import FeatureBatch
from .temporal import TemporalTemplateFamily
from .manus_study import GESTURES, _aggregate, _metrics
from .screening import FAMILY_FACTORIES, SEED


def trial_paths(data, trials):
    paths=[]
    for trial in trials:
        windows=data.batch.emg[data.trials==trial]
        paths.append(np.sqrt(np.mean(windows.astype(float)**2,axis=1)))
    if len({len(p) for p in paths})!=1:raise ValueError('variable window counts require explicit time alignment')
    return FeatureBatch(np.stack(paths),1.0)


def run(archive:Path,output:Path,phase:str)->None:
    raise ValueError('Legacy MANUS sparse-window DTW protocol is ineligible: full contiguous native sequences are required. Prior artifacts are retrospective proxy results only.')
    if output.exists():raise FileExistsError(output)
    users=(3,4,5,6,7,8);session=2 if phase=='validation' else 3
    print('[1/3] loading MANUS and fitting frozen F0+SPD',flush=True)
    train=load_semg_manus_windows(archive,users=users,sessions=(1,),gestures=GESTURES)
    target=load_semg_manus_windows(archive,users=users,sessions=(session,),gestures=GESTURES)
    aa=[];bb=[];families=[]
    for name in ('F0','F2c_SPD'):
        family=FAMILY_FACTORIES[name]()
        a,ay,au,_,_,at=_aggregate(family.fit_transform(train.batch,train.labels),train)
        b,y,u,_,_,trials=_aggregate(family.transform(target.batch),target)
        aa.append(a);bb.append(b);families.append(family)
    a=np.concatenate(aa,1);b=np.concatenate(bb,1);scaler=StandardScaler().fit(a)
    model=LogisticRegression(C=1,class_weight='balanced',max_iter=1000,random_state=SEED).fit(scaler.transform(a),ay)
    baseline=model.predict_proba(scaler.transform(b));source_paths=trial_paths(train,at);paths=trial_paths(target,trials)
    rows=[];splits=[];states={};saved={}
    print('[2/3] comparing source/personal DTW and fixed fusion',flush=True)
    for user in users:
        source=np.flatnonzero(au==user)
        source_template=TemporalTemplateFamily().fit(source_paths.take(source),ay[source])
        source_dist=source_template.transform(source_paths.take(source))
        temperature=max(float(np.median(source_dist)),1e-6)
        for shots in (0,1,2):
            rng=np.random.default_rng(SEED+user);chosen=[]
            for label in range(len(GESTURES)):
                chosen.extend(rng.permutation(np.flatnonzero((u==user)&(y==label)))[:shots])
            cal=np.asarray(chosen,dtype=int);ev=np.flatnonzero((u==user)&~np.isin(np.arange(len(y)),cal))
            if set(trials[cal])&set(trials[ev]):raise AssertionError('trial leakage')
            template=source_template if shots==0 else TemporalTemplateFamily().fit(paths.take(cal),y[cal])
            distances=template.transform(paths.take(ev));logits=-distances.astype(np.float64)/temperature
            logits-=logits.max(1,keepdims=True);p=np.exp(logits);p/=p.sum(1,keepdims=True)
            alpha=shots/(shots+2)
            for method,prob in (('baseline',baseline[ev]),('dtw',p),('baseline_dtw_fusion',(1-alpha)*baseline[ev]+alpha*p)):
                rows.append({'dataset':'semg_manus','phase':phase,'subject':user,'condition':f'session_{session}',
                    'shots_per_class':shots,'feature_bank':'F0+SPD|DTW','method':method,**_metrics(y[ev],prob)})
                saved[f'{user}_{shots}_{method}']=prob
            states[(user,shots)]=(template,temperature)
            splits.append({'user':user,'shots':shots,'source':at[source].tolist(),'calibration':trials[cal].tolist(),'evaluation':trials[ev].tolist()})
    for shots in (0,1,2):
        for method in ('baseline','dtw','baseline_dtw_fusion'):
            subset=[r for r in rows if r['shots_per_class']==shots and r['method']==method]
            rows.append({**subset[0],'subject':'ALL',**{k:float(np.mean([r[k] for r in subset])) for k in ('macro_f1','accuracy','log_loss','brier','ece')},'per_class_f1_json':'mean per-user aggregation'})
    print('[3/3] saving templates and held-out evidence',flush=True);output.mkdir(parents=True)
    with (output/'calibration_curve.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/'split_trial_ids.json').write_text(json.dumps(splits,indent=2),encoding='utf-8')
    (output/'run_manifest.json').write_text(json.dumps({'phase':phase,'seed':SEED,'target_session':session,
        'path':'ordered RMS of eight evenly spaced 200-ms windows; not continuous full-trial trajectory',
        'DTW_band_fraction':.1,'temperature':'median source-user template distance','fusion_alpha':'shots/(shots+2)',
        'unsupported_5':'only three speed trials/class/session'},indent=2),encoding='utf-8')
    with (output/'fitted_states.pkl').open('wb') as handle:pickle.dump((families,scaler,model,states),handle)
    np.savez_compressed(output/'heldout_predictions.npz',**saved,target_trials=trials,target_labels=y)
    print(json.dumps({'status':'ok','rows':len(rows)}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('archive',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=('validation','final'),required=True);args=parser.parse_args();run(args.archive,args.output,args.phase)
