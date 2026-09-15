"""Frozen EPN Core comparisons on independent users, without model fitting."""
from pathlib import Path
import argparse
import hashlib
import json
import pickle
import numpy as np
from emgimu.datasets.epn612 import load_epn612_windows
from .epn_study import aggregate_trials
from .force_nested_oof import temperature_probability
from .core_incremental_oof import CORE, ADDED, SPECS, evaluate, write


def run(archive, source, output):
    if output.exists():
        raise FileExistsError(output)
    manifest=json.loads((source/'run_manifest.json').read_text())
    if manifest['dataset']!='epn612' or manifest['phase'] not in ('validation','final'):
        raise ValueError('Requires independent EPN validation or final run')
    phase=manifest['phase']
    users=(16,17,18) if phase=='validation' else (19,20,21)
    if any(f':user{user}:' in trial for user in users for trial in manifest['source_trials']):
        raise AssertionError('Target users present in fitting trials')
    target=load_epn612_windows(archive,users=users)
    states,_=pickle.loads((source/'fitted_states.pkl').read_bytes())
    before=pickle.dumps(states)
    temperature=json.loads((source/'probability_calibration.json').read_text())
    if set(temperature['fit_users'])!=set(range(1,16)) or set(temperature['fit_trials'])!=set(manifest['source_trials']):
        raise AssertionError('Probability calibration must use source users and trials only')
    providers={}
    for name in manifest['families']:
        family,scaler,model=states[name]
        x,y,u,trials,_=aggregate_trials(family.transform(target.batch),target)
        providers[name]=temperature_probability(model.predict_proba(scaler.transform(x)),temperature['temperatures'][name])
    if pickle.dumps(states)!=before:
        raise AssertionError('Frozen fitting states changed')
    with np.load(source/'heldout_predictions.npz',allow_pickle=False) as saved:
        for name,value in (('labels',y),('users',u),('trials',trials)):
            np.testing.assert_array_equal(saved[name],value)
        for user in users:
            mask=u==user
            expected=sum(manifest['population_weights'][i]*providers[n][mask] for i,n in enumerate(manifest['families']))
            key=f'{user}_0_population_only'
            if key not in saved: key=f'{user}_0_uniform_population'
            np.testing.assert_allclose(expected,saved[key],atol=1e-10,rtol=1e-9)
    probabilities={name:np.mean([providers[n] for n in members],axis=0) for name,members in SPECS.items()}
    values=evaluate(y,u,probabilities,providers,phase,'independent users; fixed source-prespecified uniform late fusion; pooled ALL')
    output.mkdir(parents=True)
    for name,rows in zip(('feature_family_results','conditional_incremental','error_complementarity','interaction_results'),values):
        write(output/f'{name}.csv',rows)
    np.savez_compressed(output/'core_heldout_predictions.npz',labels=y,users=u,trials=trials,
        **probabilities,**{f'provider_{n}':p for n,p in providers.items()})
    (output/'split_trial_ids.json').write_text(json.dumps({'train':manifest['source_trials'],'calibration':[],'evaluation':trials.tolist()},indent=2))
    evidence={'dataset':'epn612','phase':phase,'families':list(CORE),'specs':SPECS,
        'source_run':source.name,'source_users':list(range(1,16)),'target_users':list(users),
        'classifier_or_family_fit':False,'target_calibration':False,'combination_weights':'fixed uniform',
        'source_artifact_sha256':{n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in
            ('fitted_states.pkl','run_manifest.json','probability_calibration.json','heldout_predictions.npz')},
        'scope':'late-fusion predictive conditional proxy; no concatenated-feature retraining; ALL pooled, individual users also recorded'}
    (output/'run_manifest.json').write_text(json.dumps(evidence,indent=2))
    audit={'status':'ok','source_population_replays':3,'frozen_state_unchanged':True,'target_trials':len(y),'models':len(SPECS)}
    (output/'replay_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();run(a.archive,a.source,a.output)
