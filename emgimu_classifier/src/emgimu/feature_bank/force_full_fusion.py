"""Shared specialist bank under strict source-force-only calibration."""
from pathlib import Path
import argparse
import csv
import json
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from emgimu.datasets.libemg_force import load_libemg_force_windows
from .calibration import ReliabilityWeights, PersonalAnchor, late_fusion
from .calibration_study import CONDITIONS
from .screening import FAMILY_FACTORIES,SEED,metrics

IDS=('F0','F1_X1H','F2b_CSP','F3_Ring','F4_Spectral','F5_Temporal','F9_Quality')


def aggregate(x,data):
    trials=np.unique(data.trials);rows=[];labels=[];users=[]
    for trial in trials:
        mask=data.trials==trial
        rows.append(x[mask].mean(0));labels.append(np.unique(data.labels[mask]).item());users.append(np.unique(data.subjects[mask]).item())
    return np.stack(rows),np.asarray(labels),np.asarray(users),trials


def run(root:Path,output:Path,phase:str)->None:
    if output.exists():raise FileExistsError(output)
    users=(7,8) if phase=='validation' else (9,10)
    train=load_libemg_force_windows(root,subjects=range(1,7),conditions=('Ramp',))
    calibration=load_libemg_force_windows(root,subjects=users,conditions=('Ramp',))
    target=load_libemg_force_windows(root,subjects=users,conditions=CONDITIONS)
    print('[1/3] fitting independent providers on source-force trials',flush=True)
    cal_features={};target_prob={};target_features={};states={}
    for name in IDS:
        family=FAMILY_FACTORIES[name]()
        a,ay,_,source_trials=aggregate(family.fit_transform(train.batch,train.labels),train)
        c,cy,cu,cal_trials=aggregate(family.transform(calibration.batch),calibration)
        b,y,u,trials=aggregate(family.transform(target.batch),target)
        scaler=StandardScaler().fit(a)
        model=LogisticRegression(C=1,class_weight='balanced',max_iter=2000,random_state=SEED).fit(scaler.transform(a),ay)
        cal_features[name]=scaler.transform(c);target_features[name]=scaler.transform(b)
        target_prob[name]=model.predict_proba(target_features[name]);states[name]=(family,scaler,model)
    population=np.ones(len(IDS))/len(IDS)
    reliability=ReliabilityWeights(tuple(range(7)),IDS,population,n0=8)
    rows=[];splits=[];anchors={};saved={}
    print('[2/3] comparing source-only full fusion and all family removals',flush=True)
    for user in users:
        for shots in (0,1,2):
            rng=np.random.default_rng(SEED+user);chosen=[]
            for label in range(7):chosen.extend(rng.permutation(np.flatnonzero((cu==user)&(cy==label)))[:shots])
            cal=np.asarray(chosen,dtype=int);ev=np.flatnonzero(u==user)
            if set(cal_trials[cal])&set(trials[ev]):raise AssertionError('source/target-force leakage')
            w=population if shots==0 else reliability.personal({n:(cal_features[n][cal],cy[cal]) for n in IDS})
            personal={n:target_prob[n][ev] for n in IDS}
            if shots:
                for n in IDS:
                    anchor=PersonalAnchor().fit(cal_features[n][cal],cy[cal])
                    temperature=max(float(np.median(anchor.transform(cal_features[n][cal])[:,:7])),1e-10)
                    logits=-anchor.transform(target_features[n][ev])[:,:7].astype(float)/temperature
                    logits-=logits.max(1,keepdims=True);p=np.exp(logits);p/=p.sum(1,keepdims=True)
                    alpha=shots/(shots+2);personal[n]=(1-alpha)*personal[n]+alpha*p
                    anchors[(user,shots,n)]=(anchor,temperature)
            variants={'full':(personal,w),'without_F7_anchor':({n:target_prob[n][ev] for n in IDS},w),
                'uniform_population':({n:target_prob[n][ev] for n in IDS},population),
                'baseline_F0':({'F0':target_prob['F0'][ev]},population),
                **{f'without_{n}':({f:p for f,p in personal.items() if f!=n},w) for n in IDS}}
            for method,(providers,weights) in variants.items():
                p=late_fusion(providers,IDS,weights)
                rows.append({'dataset':'libemg_contraction_intensity','phase':phase,'protocol':'Force-ZeroShot',
                    'subject':user,'condition':'ALL','shots_per_class':shots,'feature_bank':'|'.join(IDS),'method':method,
                    **metrics(y[ev],p,np.ones(len(ev)))})
                saved[f'{user}_{shots}_{method}']=p
            splits.append({'user':user,'shots':shots,'calibration':cal_trials[cal].tolist(),'evaluation':trials[ev].tolist()})
    for shots in (0,1,2):
        for method in variants:
            selected=[r for r in rows if r['shots_per_class']==shots and r['method']==method]
            rows.append({**selected[0],'subject':'ALL',**{k:float(np.mean([r[k] for r in selected])) for k in ('macro_f1','accuracy','log_loss','brier','ece')},'per_class_f1_json':'mean per-user aggregation'})
    for user in (*users,'ALL'):
        rows.append({'dataset':'libemg_contraction_intensity','phase':phase,'protocol':'Force-ZeroShot',
            'subject':user,'condition':'ALL','shots_per_class':5,'feature_bank':'|'.join(IDS),'method':'unsupported',
            **{k:'' for k in ('macro_f1','accuracy','log_loss','brier','ece','per_class_f1_json')}})
    print('[3/3] saving unified source-force bank evidence',flush=True);output.mkdir(parents=True)
    for name in ('calibration_curve','ablation_full_bank'):
        with (output/f'{name}.csv').open('w',newline='',encoding='utf-8') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/'split_trial_ids.json').write_text(json.dumps(splits,indent=2),encoding='utf-8')
    (output/'run_manifest.json').write_text(json.dumps({'phase':phase,'seed':SEED,'source_trials':source_trials.tolist(),
        'families':IDS,'source_force_only':'Ramp','evaluation_force':CONDITIONS,'evaluation_unit':'trial means',
        'n0':8,'anchor_alpha':'shots/(shots+2)','temperature_fit':'calibration only','unsupported_5':'four Ramp trials/class',
        'F6_F8':'unavailable: no real IMU or repeated-session key'},indent=2),encoding='utf-8')
    with (output/'fitted_states.pkl').open('wb') as handle:pickle.dump((states,anchors),handle)
    np.savez_compressed(output/'heldout_predictions.npz',**saved,labels=y,trials=trials,users=u)
    print(json.dumps({'status':'ok','rows':len(rows)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase',choices=('validation','final'),required=True);a=p.parse_args();run(a.root,a.output,a.phase)
