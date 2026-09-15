"""Frozen multi-specialist personal-anchor fusion and leave-family-out audit."""
from pathlib import Path
import argparse
import csv
import json
import pickle
import hashlib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from emgimu.datasets.epn612 import load_epn612_windows
from .calibration import ReliabilityWeights, PersonalAnchor, SessionSignature, late_fusion
from .epn_study import FACTORIES, aggregate_trials, _metrics
from .screening import SEED

# Specialists retained for different benchmark failures; not a subset search.
FAMILIES=('F0','F1_X1H','F2b_CSP','F3_Ring','F4_Spectral','F5_Temporal','F6_IMU','F9_Quality')


def run(archive:Path,output:Path,phase:str,dataset:str='epn612',source_run:Path|None=None,probability_oof:Path|None=None,reliability_policy:Path|None=None)->None:
    if output.exists():raise FileExistsError(output)
    aggregate=aggregate_trials; score=_metrics; condition='cross_user';budgets=(0,1,2,5)
    if dataset=='semg_manus':
        from emgimu.datasets.semg_manus import load_semg_manus_windows
        from .manus_study import GESTURES, _aggregate, _metrics as manus_metrics
        users=(3,4,5,6,7,8);session=2 if phase=='validation' else 3;condition=f'session_{session}';budgets=(0,1,2)
        train=load_semg_manus_windows(archive,users=users,sessions=(1,),gestures=GESTURES)
        target=load_semg_manus_windows(archive,users=users,sessions=(session,),gestures=GESTURES)
        def aggregate(x,data):
            x,y,u,_,_,trials=_aggregate(x,data)
            return x,y,u,trials,np.ones(len(y))
        def score(y,p,w):return manus_metrics(y,p)
    else:
        users=(16,17,18) if phase=='validation' else (19,20,21)
        train=load_epn612_windows(archive,users=range(1,16));target=load_epn612_windows(archive,users=users)
    print(f'[1/3] loaded frozen {dataset} {phase}',flush=True)
    previous=None;previous_before=None;probability_audit=None
    if source_run:
        previous,_=pickle.loads((source_run/'fitted_states.pkl').read_bytes());previous_before=pickle.dumps(previous)
        old_manifest=json.loads((source_run/'run_manifest.json').read_text())
        if old_manifest.get('dataset','epn612')!=dataset or set(old_manifest['source_trials'])!=set(train.trials):
            raise AssertionError('different dataset or source fitting trials')
    features={};probabilities={};states={};source_features={};signature_rows=[]
    for name in FAMILIES:
        family=previous[name][0] if previous else FACTORIES[name]()
        a,ay,au,at,_=aggregate(family.transform(train.batch) if previous else family.fit_transform(train.batch,train.labels),train)
        b,y,u,trials,w=aggregate(family.transform(target.batch),target)
        if name=='F9_Quality':quality_mean=np.clip(b[:,-3],0,1);quality_min=np.clip(b[:,-2],0,1)
        if previous:_,scaler,model=previous[name]
        else:
            scaler=StandardScaler().fit(a)
            model=LogisticRegression(C=1,class_weight='balanced',max_iter=1000,random_state=SEED).fit(scaler.transform(a),ay)
        features[name]=scaler.transform(b);probabilities[name]=model.predict_proba(features[name]);states[name]=(family,scaler,model)
        source_features[name]=scaler.transform(a)
    if previous and pickle.dumps(previous)!=previous_before:raise AssertionError('reused source state changed')
    if probability_oof:
        from .force_nested_oof import fit_temperature, temperature_probability
        oof_path=probability_oof/'oof_predictions.npz'
        oof_manifest=json.loads((probability_oof/'run_manifest.json').read_text())
        if oof_manifest.get('dataset')!=dataset:raise AssertionError('different OOF source dataset')
        source_users=sorted(int(v) for v in np.unique(au))
        with np.load(oof_path,allow_pickle=False) as oof:
            if set(oof['trials'])!=set(at) or set(oof['users'])!=set(source_users):
                raise AssertionError('OOF probability fitting must use source training trials/users only')
            label_by_trial=dict(zip(at,ay))
            if any(label_by_trial[t]!=label for t,label in zip(oof['trials'],oof['labels'])):raise AssertionError('OOF labels changed')
            temperatures={n:fit_temperature(oof[f'raw_{n}'],oof['labels']) for n in FAMILIES}
        probabilities={n:temperature_probability(p,temperatures[n]) for n,p in probabilities.items()}
        probability_audit={'source_run':probability_oof.name,'source_oof_sha256':hashlib.sha256(oof_path.read_bytes()).hexdigest(),
            'fit_users':source_users,'fit_trials':at.tolist(),'temperatures':temperatures,
            'fit_scope':'source training/session OOF probabilities and labels only; target calibration excluded'}
    population=np.ones(len(FAMILIES))/len(FAMILIES)
    n0=8.;reliability_temperature=1.;policy_audit=None
    if reliability_policy:
        from .reliability_selection import load_policy
        population,n0,reliability_temperature,policy_audit=load_policy(reliability_policy,dataset,at,FAMILIES,probability_oof)
    reliability=ReliabilityWeights(tuple(range(6)),FAMILIES,population,n0=n0,temperature=reliability_temperature)
    rows=[];splits=[];anchors={};saved={}
    print('[2/3] comparing full bank, no-anchor and all provider removals',flush=True)
    for user in users:
        for shots in budgets:
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
                ('population_only' if reliability_policy else 'uniform_population'):({name:probabilities[name][ev] for name in FAMILIES},population),
                **{f'without_{name}':({n:p for n,p in personalized.items() if n!=name},weights) for name in FAMILIES}}
            if dataset=='semg_manus':
                context=weights.copy()
                if shots:
                    for i,name in enumerate(FAMILIES):
                        signature=SessionSignature().fit_long_term(source_features[name][au==user],ay[au==user])
                        vector=signature.from_session_calibration(features[name][cal],y[cal])
                        context[i]*=float(np.clip((vector[6:12].mean()+1)/2,.05,1))
                        signature_rows.append({'user':user,'shots':shots,'family':name,'vector':vector.tolist()})
                    context/=context.sum()
                variants['with_F8_context']=(personalized,context)
                variants['with_F8_F9_quality']=(personalized,context)
                variants.update({
                    'combined_full':(personalized,context),
                    'combined_without_F8_context':(personalized,weights),
                    'combined_without_F9_quality':(personalized,context),
                    'combined_without_F7_anchor':({name:probabilities[name][ev] for name in FAMILIES},context),
                    **{f'combined_without_{name}':({n:p for n,p in personalized.items() if n!=name},context) for name in FAMILIES}})
            for method,(providers,weight) in variants.items():
                q=None
                if method=='with_F8_F9_quality' or (method.startswith('combined_') and method!='combined_without_F9_quality'):
                    q={name:(np.ones(len(ev)) if name in ('F6_IMU','F9_Quality') else quality_min[ev] if name in ('F0','F2b_CSP') else quality_mean[ev]) for name in FAMILIES}
                p=late_fusion(providers,FAMILIES,weight,q)
                rows.append({'dataset':dataset,'phase':phase,'subject':user,'condition':condition,
                    'shots_per_class':shots,'feature_bank':'|'.join(FAMILIES),'method':method,
                    'removed_family':'NONE' if method=='full' else method.removeprefix('without_'),**score(y[ev],p,np.ones(len(ev)))})
                saved[f'{user}_{shots}_{method}']=p
            splits.append({'user':user,'shots':shots,'calibration':trials[cal].tolist(),'evaluation':trials[ev].tolist()})
    for shots in budgets:
        for method in variants:
            selected=[r for r in rows if r['shots_per_class']==shots and r['method']==method]
            rows.append({**selected[0],'subject':'ALL',**{k:float(np.mean([r[k] for r in selected])) for k in ('macro_f1','accuracy','log_loss','brier','ece')},'per_class_f1_json':'mean per-user aggregation'})
    if dataset=='semg_manus':
        for user in (*users,'ALL'):
            rows.append({'dataset':dataset,'phase':phase,'subject':user,'condition':condition,
                'shots_per_class':5,'feature_bank':'|'.join(FAMILIES),'method':'unsupported',
                'removed_family':'N/A',**{k:'' for k in ('macro_f1','accuracy','log_loss','brier','ece','per_class_f1_json')}})
    print('[3/3] saving full-bank evidence and fitted states',flush=True);output.mkdir(parents=True)
    for name,values in (('calibration_curve',rows),('ablation_full_bank',[r for r in rows if r['method'] not in ('uniform_population','population_only')])):
        with (output/f'{name}.csv').open('w',newline='',encoding='utf-8') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    (output/'split_trial_ids.json').write_text(json.dumps(splits,indent=2),encoding='utf-8')
    (output/'session_signatures.json').write_text(json.dumps(signature_rows,indent=2),encoding='utf-8')
    if probability_audit:(output/'probability_calibration.json').write_text(json.dumps(probability_audit,indent=2),encoding='utf-8')
    (output/'run_manifest.json').write_text(json.dumps({'phase':phase,'seed':SEED,'source_trials':at.tolist(),
        'families':FAMILIES,'population_weights':population.tolist(),'n0':n0,'reliability_temperature':reliability_temperature,
        'reliability_policy':policy_audit,'anchor_alpha':'shots/(shots+2)',
        'anchor_temperature':'calibration distance median only','dataset':dataset,
        'F8':'calibration class cosine agreement' if dataset=='semg_manus' else 'N/A: no validated repeated-session key',
        'quality':'F0/CSP min quality, robust providers mean quality, IMU/context 1' if dataset=='semg_manus' else 'F9 classifier provider only',
        'unsupported_5':'three trials/class/session' if dataset=='semg_manus' else None,
        'rule_selection':'source-user OOF selection; no target tuning' if reliability_policy else 'fixed before validation; no tuning on final users',
        'classifier_or_family_fit':source_run is None,'reused_source_run':source_run.name if source_run else None,
        'probability_calibration':'source-user OOF temperature' if probability_oof else 'native logistic probability'},indent=2),encoding='utf-8')
    with (output/'fitted_states.pkl').open('wb') as handle:pickle.dump((states,anchors),handle)
    np.savez_compressed(output/'heldout_predictions.npz',**saved,labels=y,trials=trials,users=u)
    print(json.dumps({'status':'ok','rows':len(rows)}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('archive',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=('validation','final'),required=True)
    parser.add_argument('--dataset',choices=('epn612','semg_manus'),default='epn612')
    parser.add_argument('--source-run',type=Path);parser.add_argument('--probability-oof',type=Path)
    parser.add_argument('--reliability-policy',type=Path)
    args=parser.parse_args();run(args.archive,args.output,args.phase,args.dataset,args.source_run,args.probability_oof,args.reliability_policy)
