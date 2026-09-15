"""Replay frozen UniBo predictors from saved state and target days only."""
from pathlib import Path
import argparse
import json
import pickle
import numpy as np
from emgimu.datasets.unibo_physiology import load_chronological_raw_windows
from .core import FeatureBatch


def replay(dataset:Path,run_root:Path)->dict:
    manifest=json.loads((run_root/'run_manifest.json').read_text(encoding='utf-8'))
    families,models=pickle.loads((run_root/'fitted_states.pkl').read_bytes())
    before=pickle.dumps((families,models))
    print('[1/2] loading target days and transforming frozen families',flush=True)
    target=load_chronological_raw_windows(dataset,manifest['target_days'])
    batch=FeatureBatch(target.emg,200.,posture=target.posture.astype(str))
    features={name:family.transform(batch) for name,family in families.items()}
    errors=[]
    print('[2/2] comparing all saved probability arrays and sample identifiers',flush=True)
    with np.load(run_root/'heldout_predictions.npz',allow_pickle=False) as saved:
        np.testing.assert_array_equal(target.trial_id,saved['trials'])
        np.testing.assert_array_equal(target.labels,saved['labels'])
        np.testing.assert_array_equal(target.subject_id,saved['subjects'])
        np.testing.assert_array_equal(target.session_id,saved['days'])
        np.testing.assert_array_equal(target.posture,saved['posture'])
        for name,(scaler,model,members) in models.items():
            x=np.concatenate([features[m] for m in members],1)
            p=model.predict_proba(scaler.transform(x))
            np.testing.assert_allclose(p,saved[name],rtol=1e-7,atol=1e-8,err_msg=name)
            errors.append(float(np.abs(p-saved[name]).max()))
        expected={k for k in saved.files if k not in ('trials','labels','subjects','days','posture')}
        if expected!=set(models):raise AssertionError('prediction coverage mismatch')
    if before!=pickle.dumps((families,models)):raise AssertionError('replay changed fitted state')
    result={'status':'ok','run_id':run_root.name,'models_checked':len(models),'windows_checked':len(target),
        'max_absolute_probability_error':max(errors),'fit_performed':False,'target_days':manifest['target_days']}
    (run_root/'replay_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',type=Path);p.add_argument('run_root',type=Path)
    a=p.parse_args();print(json.dumps(replay(a.dataset,a.run_root)))
