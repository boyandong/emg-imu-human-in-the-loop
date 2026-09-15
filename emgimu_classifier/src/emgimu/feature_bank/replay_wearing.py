from pathlib import Path
import argparse
import json
import pickle
import numpy as np
from emgimu.datasets.electrode_shift import load_electrode_shift_windows
from .electrode_shift_study import _aggregate


def replay(archive:Path,run_root:Path)->dict:
    families,models=pickle.loads((run_root/'fitted_states.pkl').read_bytes());before=pickle.dumps((families,models))
    splits=json.loads((run_root/'subject_split_trial_ids.json').read_text(encoding='utf-8'))
    checked=0;error=0.
    with np.load(run_root/'heldout_predictions.npz',allow_pickle=False) as saved:
        for subject_string,split in splits.items():
            subject=int(subject_string)
            target=load_electrode_shift_windows(archive,subjects=(subject,),domains=('trial_1','trial_2','trial_3','trial_4'))
            parts={}
            for (s,name),family in families.items():
                if s==subject:parts[name],y,_,trials=_aggregate(family.transform(target.batch),target)
            np.testing.assert_array_equal(trials,saved[f's{subject}_trials']);np.testing.assert_array_equal(y,saved[f's{subject}_labels'])
            evaluation=split.get('validation',split.get('test',[]))
            if set(split['train'])&set(evaluation):raise AssertionError('trial leakage')
            for (s,name),(scaler,model,members) in models.items():
                if s!=subject:continue
                p=model.predict_proba(scaler.transform(np.concatenate([parts[n] for n in members],1)))
                np.testing.assert_allclose(p,saved[f's{subject}_{name}'],rtol=1e-7,atol=1e-8)
                error=max(error,float(np.abs(p-saved[f's{subject}_{name}']).max()));checked+=1
    if before!=pickle.dumps((families,models)):raise AssertionError('fitted state changed')
    result={'status':'ok','run_id':run_root.name,'models_checked':checked,'max_probability_error':error,'fit_performed':False}
    (run_root/'replay_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('run_root',type=Path)
    a=p.parse_args();print(json.dumps(replay(a.archive,a.run_root)))
