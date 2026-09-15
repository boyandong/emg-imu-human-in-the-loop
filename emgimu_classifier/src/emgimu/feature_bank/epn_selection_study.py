"""Held-out EPN shortlist interactions and complete leave-family-out diagnostics."""
from pathlib import Path
from itertools import combinations
import argparse
import csv
import json
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from emgimu.datasets.epn612 import load_epn612_windows
from .epn_study import FACTORIES, aggregate_trials, _metrics
from .screening import SEED

FULL=('F0','F3_Ring','F2b_CSP','F6_IMU')


def run(archive:Path,output:Path)->None:
    if output.exists():raise FileExistsError(output)
    print('[1/3] loading fixed EPN train 1–15 / validation 16–18',flush=True)
    train=load_epn612_windows(archive,users=range(1,16));target=load_epn612_windows(archive,users=(16,17,18))
    source={};features={};states={};predictions={};classifier_states={}
    for name in FULL:
        family=FACTORIES[name]()
        a,ay,_,at,_=aggregate_trials(family.fit_transform(train.batch,train.labels),train)
        b,y,u,trials,w=aggregate_trials(family.transform(target.batch),target)
        source[name]=a;features[name]=b;states[name]=family
    specs={'baseline':('F0',),'full':FULL}
    specs.update({f'single_{n}':('F0',n) for n in FULL[1:]})
    specs.update({f'pair_{a}_{b}':('F0',a,b) for a,b in combinations(FULL[1:],2)})
    specs.update({f'leave_out_{n}':tuple(f for f in FULL if f!=n) for n in FULL})
    rows=[]
    print('[2/3] fitting shortlist, selected pairs and all four removals',flush=True)
    for name,members in specs.items():
        a=np.concatenate([source[n] for n in members],1);b=np.concatenate([features[n] for n in members],1)
        scaler=StandardScaler().fit(a)
        model=LogisticRegression(C=1,class_weight='balanced',max_iter=1000,random_state=SEED).fit(scaler.transform(a),ay)
        p=model.predict_proba(scaler.transform(b));predictions[name]=p;classifier_states[name]=(scaler,model)
        for user in ('ALL',16,17,18):
            selected=np.ones(len(y),dtype=bool) if user=='ALL' else u==user
            rows.append({'dataset':'epn612','subject':user,'condition':'cross_user','model':name,
                'feature_family':'+'.join(members),'calibration_budget':0,'feature_dimension':a.shape[1],**_metrics(y[selected],p[selected],w[selected])})
    ablations=[];increments=[];complementarity=[]
    for user in ('ALL',16,17,18):
        cell=[r for r in rows if r['subject']==user];reference=next(r for r in cell if r['model']=='full')
        base=next(r for r in cell if r['model']=='baseline')
        for r in cell:
            if r['model']=='full' or r['model'].startswith('leave_out_'):
                ablations.append({**r,'removed_family':'NONE' if r['model']=='full' else r['model'].removeprefix('leave_out_'),
                    'delta_macro_f1_vs_full':r['macro_f1']-reference['macro_f1']})
            if r['model'].startswith('single_'):
                increments.append({'dataset':'epn612','subject':user,'core_bank':'F0','added_family':r['model'].removeprefix('single_'),
                    'delta_logloss':base['log_loss']-r['log_loss'],'delta_macro_f1':r['macro_f1']-base['macro_f1'],
                    'delta_brier':base['brier']-r['brier'],'evaluation':'held-out users'})
    for a,b in combinations(predictions,2):
        ea=predictions[a].argmax(1)!=y;eb=predictions[b].argmax(1)!=y
        correlation=float(np.corrcoef(ea,eb)[0,1]) if ea.std()>0 and eb.std()>0 else 0.
        complementarity.append({'dataset':'epn612','family_a':a,'family_b':b,'error_correlation':correlation,
            'disagreement':float(np.mean(ea!=eb)),'a_correct_b_wrong':float(np.mean(~ea&eb)),
            'a_wrong_b_correct':float(np.mean(ea&~eb))})
    print('[3/3] saving held-out evidence and fitted states',flush=True);output.mkdir(parents=True)
    for name,values in (('feature_family_results',rows),('ablation_full_bank',ablations),('conditional_incremental',increments),('error_complementarity',complementarity)):
        with (output/f'{name}.csv').open('w',newline='',encoding='utf-8') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    (output/'run_manifest.json').write_text(json.dumps({'seed':SEED,'train_users':list(range(1,16)),
        'validation_users':[16,17,18],'families':FULL,'shortlist_rule':'top three single-family held-out validation F1 gains',
        'scope':'development analysis; no new final-set selection','models':specs},indent=2),encoding='utf-8')
    (output/'split_trial_ids.json').write_text(json.dumps({'train':at.tolist(),'validation':trials.tolist()},indent=2),encoding='utf-8')
    np.savez_compressed(output/'heldout_predictions.npz',**predictions,labels=y,trials=trials,users=u)
    with (output/'fitted_states.pkl').open('wb') as handle:pickle.dump((states,classifier_states),handle)
    print(json.dumps({'status':'ok','models':len(specs),'rows':len(rows)}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('archive',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.archive,args.output)
