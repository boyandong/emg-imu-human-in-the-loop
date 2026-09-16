"""Replay saved EPN Anchor metrics and split identities without model fitting."""
import argparse, csv, hashlib, json, pickle
from pathlib import Path
import numpy as np
from emgimu.feature_bank.epn_study import _metrics


def same(actual, expected):
    for key in ('macro_f1','accuracy','log_loss','brier','ece'):
        if not np.isclose(float(actual[key]), float(expected[key]), rtol=0, atol=1e-12):
            raise AssertionError(f'{key} changed')
    if actual['per_class_f1_json'] != expected['per_class_f1_json']:
        raise AssertionError('per-class F1 changed')


def replay(run):
    rows=list(csv.DictReader((run/'calibration_curve.csv').open(encoding='utf-8')))
    splits=json.loads((run/'split_trial_ids.json').read_text(encoding='utf-8'))
    state_path=run/'fitted_source_state.pkl';original=state_path.read_bytes();state=pickle.loads(original)
    if set(state) != {'families','scaler','classifier'}:raise AssertionError('source state schema changed')
    checked=0
    with np.load(run/'heldout_predictions.npz',allow_pickle=False) as saved:
        for shots in (0,1,2,5):
            combined={key:[] for key in ('probability','labels','weights')}
            for subject in sorted({int(r['subject']) for r in rows if r['subject']!='ALL'}):
                key=f'user{subject}_shots{shots}'
                probability=saved[f'{key}_probability'];labels=saved[f'{key}_labels'];weights=saved[f'{key}_weights'];trials=saved[f'{key}_trials']
                if probability.shape!=(len(labels),6) or len(weights)!=len(labels) or len(trials)!=len(labels):raise AssertionError('saved shape mismatch')
                if not np.all(np.isfinite(probability)) or not np.allclose(probability.sum(1),1,rtol=0,atol=1e-12):raise AssertionError('invalid probability')
                split=splits['target_cases'][key]
                if set(split['calibration']) & set(split['evaluation']) or set(trials)!=set(split['evaluation']):raise AssertionError('split identity mismatch')
                expected=next(r for r in rows if r['subject']==str(subject) and r['shots_per_class']==str(shots))
                same(_metrics(labels,probability,weights),expected);checked+=1
                for name,value in (('probability',probability),('labels',labels),('weights',weights)):combined[name].append(value)
            actual=_metrics(np.concatenate(combined['labels']),np.concatenate(combined['probability']),np.concatenate(combined['weights']))
            expected=next(r for r in rows if r['subject']=='ALL' and r['shots_per_class']==str(shots))
            same(actual,expected);checked+=1
    if state_path.read_bytes()!=original:raise AssertionError('source state file changed')
    result={'status':'ok','phase':json.loads((run/'run_manifest.json').read_text())['phase'],'metric_rows_replayed':checked,
        'probability_arrays_checked':12,'source_state_file_unchanged':True,'source_state_sha256':hashlib.sha256(original).hexdigest(),
        'scope':'saved probabilities, labels, weights and split identities; no refit and no raw-data formula replay'}
    (run/'replay_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);replay(p.parse_args().run)
