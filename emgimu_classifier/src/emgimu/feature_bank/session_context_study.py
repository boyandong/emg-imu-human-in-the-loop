"""Compare session-signature reliability fusion on disjoint MANUS trials."""
from pathlib import Path
import argparse
import csv
import json
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from emgimu.datasets.semg_manus import load_semg_manus_windows
from .calibration import SessionSignature, ReliabilityWeights, late_fusion
from .manus_study import GESTURES, _aggregate, _metrics
from .screening import FAMILY_FACTORIES, SEED


def run(archive:Path, output:Path, phase:str)->None:
    if output.exists():raise FileExistsError(output)
    users=(3,4,5,6,7,8);session=2 if phase=='validation' else 3
    ids=('F0','F2c_SPD')
    print(f'[1/3] loading MANUS source/session {session}',flush=True)
    train=load_semg_manus_windows(archive,users=users,sessions=(1,),gestures=GESTURES)
    target=load_semg_manus_windows(archive,users=users,sessions=(session,),gestures=GESTURES)
    source={};features={};probabilities={};states={}
    for name in ids:
        family=FAMILY_FACTORIES[name]()
        a,ay,au,_,_,at=_aggregate(family.fit_transform(train.batch,train.labels),train)
        b,y,u,_,_,trials=_aggregate(family.transform(target.batch),target)
        scaler=StandardScaler().fit(a);a=scaler.transform(a);b=scaler.transform(b)
        model=LogisticRegression(C=1,class_weight='balanced',max_iter=1000,random_state=SEED).fit(a,ay)
        source[name]=a;features[name]=b;probabilities[name]=model.predict_proba(b)
        states[name]=(family,scaler,model)
    rows=[];splits=[];signatures=[];predictions={}
    rel=ReliabilityWeights(tuple(range(len(GESTURES))),ids,np.asarray([.5,.5]),n0=8)
    print('[2/3] comparing calibration reliability with session-context reliability',flush=True)
    for user in users:
        for shots in (0,1,2):
            rng=np.random.default_rng(SEED+user);chosen=[]
            for label in range(len(GESTURES)):
                chosen.extend(rng.permutation(np.flatnonzero((u==user)&(y==label)))[:shots])
            cal=np.asarray(chosen,dtype=int);ev=np.flatnonzero((u==user)&~np.isin(np.arange(len(y)),cal))
            if set(trials[cal])&set(trials[ev]):raise AssertionError('trial leakage')
            personal=rel.population if shots==0 else rel.personal({name:(features[name][cal],y[cal]) for name in ids})
            context=personal.copy()
            if shots:
                agreement=[]
                for name in ids:
                    signature=SessionSignature().fit_long_term(source[name][au==user],ay[au==user])
                    before=signature.long_term_.copy()
                    vector=signature.from_session_calibration(features[name][cal],y[cal])
                    np.testing.assert_array_equal(before,signature.long_term_)
                    n=len(GESTURES)
                    # Cosine agreement is dimension-independent; no target variance is fitted.
                    agreement.append(float(np.clip((vector[n:2*n].mean()+1)/2,.05,1)))
                    signatures.append({'user':user,'shots_per_class':shots,'family':name,
                        'vector':vector.tolist(),'names':signature.feature_names})
                context*=agreement;context/=context.sum()
            for method,w in (('uniform',rel.population),('personal',personal),('personal_session_context',context)):
                p=late_fusion({name:probabilities[name][ev] for name in ids},ids,w)
                rows.append({'dataset':'semg_manus','phase':phase,'subject':user,'condition':f'session_{session}',
                    'shots_per_class':shots,'feature_bank':'F0|F2c_SPD','method':method,
                    'weights_json':json.dumps(w.tolist()),**_metrics(y[ev],p)})
                predictions[f'u{user}_k{shots}_{method}']=p
            splits.append({'user':user,'shots':shots,'calibration':trials[cal].tolist(),'evaluation':trials[ev].tolist()})
    for shots in (0,1,2):
        for method in ('uniform','personal','personal_session_context'):
            selected=[r for r in rows if r['shots_per_class']==shots and r['method']==method]
            rows.append({**selected[0],'subject':'ALL','weights_json':'per-user weights',
                **{key:float(np.mean([r[key] for r in selected])) for key in ('macro_f1','accuracy','log_loss','brier','ece')},
                'per_class_f1_json':'mean per-user aggregation'})
    print('[3/3] persisting predictions, session vectors and fitted states',flush=True)
    output.mkdir(parents=True)
    with (output/'calibration_curve.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for name,value in (('split_trial_ids',splits),('session_signatures',signatures),('run_manifest',{
        'seed':SEED,'phase':phase,'source_session':1,'target_session':session,'users':users,'families':ids,
        'source_trial_ids':at.tolist(), 'population_weights':[.5,.5],'n0':8,'session_context_rule':'personal weights times clipped (mean class cosine+1)/2',
        'unsupported_budget_5':'only three trials/class/session; leave one for evaluation'})):
        (output/f'{name}.json').write_text(json.dumps(value,indent=2),encoding='utf-8')
    with (output/'fitted_states.pkl').open('wb') as handle:pickle.dump(states,handle)
    np.savez_compressed(output/'heldout_predictions.npz',**predictions, target_labels=y,target_trials=trials,target_users=u)
    print(json.dumps({'status':'ok','rows':len(rows),'output':str(output)}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('archive',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=('validation','final'),required=True);args=parser.parse_args();run(args.archive,args.output,args.phase)
