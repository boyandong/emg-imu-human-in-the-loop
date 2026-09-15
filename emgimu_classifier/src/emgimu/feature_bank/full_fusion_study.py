"""Frozen multi-specialist personal-anchor fusion and leave-family-out audit."""
from pathlib import Path
import argparse
import csv
import json
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from emgimu.datasets.epn612 import load_epn612_windows
from .calibration import ReliabilityWeights, PersonalAnchor, late_fusion
from .epn_study import FACTORIES, aggregate_trials, _metrics
from .screening import SEED

# Specialists retained for different benchmark failures; not a subset search.
FAMILIES=('F0','F1_X1H','F2b_CSP','F3_Ring','F4_Spectral','F5_Temporal','F6_IMU','F9_Quality')


def run(archive:Path,output:Path,phase:str)->None:
    if output.exists():raise FileExistsError(output)
    users=(16,17,18) if phase=='validation' else (19,20,21)
    print(f'[1/3] loading frozen EPN {phase}',flush=True)
    train=load_epn612_windows(archive,users=range(1,16));target=load_epn612_windows(archive,users=users)
    features={};probabilities={};states={}
    for name in FAMILIES:
        family=FACTORIES[name]()
        a,ay,_,at,_=aggregate_trials(family.fit_transform(train.batch,train.labels),train)
        b,y,u,trials,w=aggregate_trials(family.transform(target.batch),target)
        scaler=StandardScaler().fit(a)
        model=LogisticRegression(C=1,class_weight='balanced',max_iter=1000,random_state=SEED).fit(scaler.transform(a),ay)
        features[name]=scaler.transform(b);probabilities[name]=model.predict_proba(features[name]);states[name]=(family,scaler,model)
    population=np.ones(len(FAMILIES))/len(FAMILIES)
    reliability=ReliabilityWeights(tuple(range(6)),FAMILIES,population,n0=8)
    rows=[];splits=[];anchors={};saved={}
    print('[2/3] comparing full bank, no-anchor and all provider removals',flush=True)
    for user in users:
        for shots in (0,1,2,5):
            rng=np.random.default_rng(SEED+user);chosen=[]
            for label in range(6):chosen.extend(rng.permutation(np.flatnonzero((u==user)&(y==label)))[:shots])
            cal=np.asarray(chosen,dtype=int);ev=np.flatnonzero((u==user)&~np.isin(np.arange(len(y)),cal))
            if set(trials[cal])&set(trials[ev]):raise AssertionError('calibration leakage')
            weights=population if shots==0 else reliability.personal({name:(features[name][cal],y[cal]) for name in FAMILIES})
            personalized={name:probabilities[name][ev] for name in FAMILIES}
            if shots:
                for name in FAMILIES:
                    anchor=PersonalAnchor().fit(features[name][cal],y[cal])
                    own=anchor.transform(features[name][cal])[:,:6]
                    temperature=max(float(np.median(own)),1e-10)
                    logits=-anchor.transform(features[name][ev])[:,:6].astype(float)/temperature
                    logits-=logits.max(1,keepdims=True);p=np.exp(logits);p/=p.sum(1,keepdims=True)
                    alpha=shots/(shots+2)
                    personalized[name]=(1-alpha)*personalized[name]+alpha*p
                    anchors[(user,shots,name)]=(anchor,temperature)
            variants={'full':(personalized,weights),
                'without_F7_anchor':({name:probabilities[name][ev] for name in FAMILIES},weights),
                'uniform_population':({name:probabilities[name][ev] for name in FAMILIES},population),
                **{f'without_{name}':({n:p for n,p in personalized.items() if n!=name},weights) for name in FAMILIES}}
            for method,(providers,weight) in variants.items():
                p=late_fusion(providers,FAMILIES,weight)
                rows.append({'dataset':'epn612','phase':phase,'subject':user,'condition':'cross_user',
                    'shots_per_class':shots,'feature_bank':'|'.join(FAMILIES),'method':method,
                    'removed_family':'NONE' if method=='full' else method.removeprefix('without_'),**_metrics(y[ev],p,np.ones(len(ev)))})
                saved[f'{user}_{shots}_{method}']=p
            splits.append({'user':user,'shots':shots,'calibration':trials[cal].tolist(),'evaluation':trials[ev].tolist()})
    for shots in (0,1,2,5):
        for method in variants:
            selected=[r for r in rows if r['shots_per_class']==shots and r['method']==method]
            rows.append({**selected[0],'subject':'ALL',**{k:float(np.mean([r[k] for r in selected])) for k in ('macro_f1','accuracy','log_loss','brier','ece')},'per_class_f1_json':'mean per-user aggregation'})
    print('[3/3] saving full-bank evidence and fitted states',flush=True);output.mkdir(parents=True)
    for name,values in (('calibration_curve',rows),('ablation_full_bank',[r for r in rows if r['method']!='uniform_population'])):
        with (output/f'{name}.csv').open('w',newline='',encoding='utf-8') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    (output/'split_trial_ids.json').write_text(json.dumps(splits,indent=2),encoding='utf-8')
    (output/'run_manifest.json').write_text(json.dumps({'phase':phase,'seed':SEED,'source_trials':at.tolist(),
        'families':FAMILIES,'population_weights':population.tolist(),'n0':8,'anchor_alpha':'shots/(shots+2)',
        'anchor_temperature':'calibration distance median only','F8':'N/A: no validated repeated-session key',
        'quality':'F9 classifier provider; no sample-level quality gating in this run',
        'rule_selection':'fixed before validation; no tuning on final users'},indent=2),encoding='utf-8')
    with (output/'fitted_states.pkl').open('wb') as handle:pickle.dump((states,anchors),handle)
    np.savez_compressed(output/'heldout_predictions.npz',**saved,labels=y,trials=trials,users=u)
    print(json.dumps({'status':'ok','rows':len(rows)}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('archive',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=('validation','final'),required=True);args=parser.parse_args();run(args.archive,args.output,args.phase)
