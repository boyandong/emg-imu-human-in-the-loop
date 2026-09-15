from pathlib import Path
import argparse
import json
import pickle
import numpy as np
from emgimu.datasets.emg_fmg import load_emg_fmg_windows
from .emg_fmg_study import _trial_probabilities


def replay(archive:Path,run_root:Path)->dict:
    families,models=pickle.loads((run_root/'fitted_states.pkl').read_bytes());before=pickle.dumps((families,models))
    splits=json.loads((run_root/'scenario_split_trial_ids.json').read_text(encoding='utf-8'))
    checked=0;error=0.
    with np.load(run_root/'heldout_predictions.npz',allow_pickle=False) as saved:
        for subject in sorted({s['subject'] for s in splits}):
            data=load_emg_fmg_windows(archive,subjects=(subject,),loads=(0,250,500,750,1000),positions=range(1,9))
            for split in (s for s in splits if s['subject']==subject):
                scenario=split['scenario'];evaluation=split.get('validation',split.get('test',[]))
                if set(split['train'])&set(evaluation):raise AssertionError('trial leakage')
                target=data.take(np.flatnonzero(np.isin(data.trials,evaluation)))
                parts={n:f.transform(target.batch) for (s,sc,n),f in families.items() if s==subject and sc==scenario}
                for (s,sc,n),(scaler,model,members) in models.items():
                    if s!=subject or sc!=scenario:continue
                    p=model.predict_proba(scaler.transform(np.concatenate([parts[m] for m in members],1)))
                    p,y,_,_=_trial_probabilities(p,target);key=f's{subject}_{scenario}_{n}'
                    np.testing.assert_array_equal(y,saved[f's{subject}_{scenario}_labels'])
                    np.testing.assert_array_equal(np.unique(target.trials),saved[f's{subject}_{scenario}_trials'])
                    np.testing.assert_allclose(p,saved[key],rtol=1e-7,atol=1e-8)
                    error=max(error,float(np.abs(p-saved[key]).max()));checked+=1
    if pickle.dumps((families,models))!=before:raise AssertionError('fitted state changed')
    result={'status':'ok','run_id':run_root.name,'models_checked':checked,'max_probability_error':error,'fit_performed':False}
    (run_root/'replay_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('run_root',type=Path)
    a=p.parse_args();print(json.dumps(replay(a.archive,a.run_root)))
