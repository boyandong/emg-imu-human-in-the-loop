"""Preconfigured F0 reference for the frozen independent force bank."""
from pathlib import Path
import argparse
import csv
import json
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from emgimu.datasets.libemg_force import load_libemg_force_windows
from .calibration_study import CONDITIONS,FROZEN_FAMILIES
from .screening import SEED,metrics


def run(root:Path,frozen_run:Path,output:Path)->None:
    if output.exists():raise FileExistsError(output)
    families,bank_scaler,bank_model,_=pickle.loads((frozen_run/'fitted_states.pkl').read_bytes())
    train=load_libemg_force_windows(root,subjects=range(1,7),conditions=('Ramp',))
    target=load_libemg_force_windows(root,subjects=(9,10),conditions=CONDITIONS)
    print('[1/2] fitting prespecified source-only F0 reference',flush=True)
    a=families[0].transform(train.batch);b=families[0].transform(target.batch)
    scaler=StandardScaler().fit(a,sample_weight=train.sample_weight)
    model=LogisticRegression(C=1,class_weight='balanced',max_iter=2000,random_state=SEED).fit(scaler.transform(a),train.labels,sample_weight=train.sample_weight)
    predictions={'F0':model.predict_proba(scaler.transform(b))}
    bank_x=np.concatenate([f.transform(target.batch) for f in families],1)
    predictions['+'.join(FROZEN_FAMILIES)]=bank_model.predict_proba(bank_scaler.transform(bank_x))
    rows=[]
    for name,p in predictions.items():
        for subject in ('ALL',9,10):
            for condition in ('ALL',*CONDITIONS):
                selected=np.ones(len(target.labels),dtype=bool)
                if subject!='ALL':selected&=target.subjects==subject
                if condition!='ALL':selected&=target.conditions==condition
                rows.append({'dataset':'libemg_contraction_intensity','phase':'final','subject':subject,'condition':condition,
                    'feature_family':name,'calibration_budget':0,**metrics(target.labels[selected],p[selected],target.sample_weight[selected])})
    print('[2/2] saving fair independent baseline/bank comparison',flush=True);output.mkdir(parents=True)
    with (output/'feature_family_results.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (output/'fitted_reference.pkl').open('wb') as handle:pickle.dump((families[0],scaler,model),handle)
    np.savez_compressed(output/'heldout_predictions.npz',**predictions,labels=target.labels,trials=target.trials,weights=target.sample_weight)
    (output/'split_trial_ids.json').write_text(json.dumps({'train':sorted(set(train.trials.tolist())),'test':sorted(set(target.trials.tolist()))},indent=2),encoding='utf-8')
    (output/'run_manifest.json').write_text(json.dumps({'seed':SEED,'train_subjects':list(range(1,7)),'target_subjects':[9,10],
        'source_force':'Ramp','target_force':CONDITIONS,'classifier_C':1,'frozen_bank':frozen_run.name,
        'reference_rule':'same F0 family and source-fit logistic parameters as development; no test tuning'},indent=2),encoding='utf-8')
    print(json.dumps({'status':'ok','rows':len(rows)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('frozen_run',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.root,a.frozen_run,a.output)
