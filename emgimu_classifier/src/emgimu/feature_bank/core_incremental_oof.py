"""Fixed multiview Core additions and two prespecified interactions from nested source OOF."""
from itertools import combinations
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np
from .epn_study import _metrics

CORE=('F0','F3_Ring','F2b_CSP','F6_IMU')
ADDED=('F1_X1H','F4_Spectral','F5_Temporal','F9_Quality')
SPECS={'Core':CORE,**{f'Core+{n}':CORE+(n,) for n in ADDED},
    'B0':('F0',),'B0+X1':('F0','F1_X1H'),'B0+Frequency':('F0','F4_Spectral'),
    'B0+CSP':('F0','F2b_CSP'),'B0+X1+Frequency':('F0','F1_X1H','F4_Spectral'),
    'B0+X1+CSP':('F0','F1_X1H','F2b_CSP')}


def write(path,rows):
    with path.open('w',newline='',encoding='utf-8') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def run(source,output):
    if output.exists():raise FileExistsError(output)
    manifest=json.loads((source/'run_manifest.json').read_text())
    if manifest['dataset']!='epn612':raise ValueError('Frozen Core is the validated EPN F0/ring/CSP/IMU bank')
    splits=json.loads((source/'split_trial_ids.json').read_text());expected=set()
    for split in splits:
        if set(split['train'])&set(split['validation']):raise AssertionError('Source outer leakage')
        for inner in split['inner']:
            if set(inner['train'])&set(inner['validation']) or (set(inner['train'])|set(inner['validation']))!=set(split['train']):
                raise AssertionError('Source inner leakage or coverage')
        if expected&set(split['validation']):raise AssertionError('Duplicate outer trial coverage')
        expected.update(split['validation'])
    path=source/'oof_predictions.npz'
    with np.load(path,allow_pickle=False) as saved:
        y=saved['labels'];u=saved['users'];trials=saved['trials']
        if set(u)!=set(manifest['source_users']):raise AssertionError('Source OOF user coverage mismatch')
        if set(trials)!=expected or len(set(trials))!=len(trials):raise AssertionError('OOF trial mismatch')
        for split in splits:
            index=np.isin(trials,split['validation'])
            np.testing.assert_array_equal(saved['folds'][index],np.full(index.sum(),split['fold']))
        probabilities={name:np.mean([saved[f'calibrated_{n}'] for n in members],axis=0) for name,members in SPECS.items()}
        individual={n:saved[f'calibrated_{n}'].copy() for n in ADDED}
    rows=[];increments=[];interactions=[];pairs=[]
    for user in ('ALL',*sorted(set(u))):
        mask=np.ones(len(y),bool) if user=='ALL' else u==user
        scores={n:_metrics(y[mask],p[mask],np.ones(mask.sum())) for n,p in probabilities.items()}
        shared={'dataset':'epn612','phase':'source_nested_oof','subject':int(user) if user!='ALL' else user,
            'condition':'cross_user','calibration_budget':0,'evaluation':'source-user nested OOF; fixed uniform late fusion'}
        for name,score in scores.items():rows.append({**shared,'feature_family':'|'.join(SPECS[name]),'model':name,**score})
        for n in ADDED:
            base=scores['Core'];added=scores[f'Core+{n}']
            increments.append({**shared,'core_bank':'|'.join(CORE),'added_family':n,
                'delta_logloss':base['log_loss']-added['log_loss'],'delta_macro_f1':added['macro_f1']-base['macro_f1'],
                'delta_brier':base['brier']-added['brier']})
            a=probabilities['Core'][mask].argmax(1);b=individual[n][mask].argmax(1)
            ca=a==y[mask];cb=b==y[mask]
            pairs.append({**shared,'family_a':'Core','family_b':n,
                'error_correlation':float(np.corrcoef(~ca,~cb)[0,1]) if np.std(ca) and np.std(cb) else '',
                'disagreement_rate':float(np.mean(a!=b)),'a_correct_b_wrong':float(np.mean(ca&~cb)),
                'a_wrong_b_correct':float(np.mean(~ca&cb))})
        for family,single,pair in (('F4_Spectral','B0+Frequency','B0+X1+Frequency'),('F2b_CSP','B0+CSP','B0+X1+CSP')):
            base=scores['B0'];first=scores['B0+X1'];second=scores[single];joint=scores[pair]
            interactions.append({**shared,'family_a':'F1_X1H','family_b':family,
                'S_negative_logloss':-joint['log_loss']+first['log_loss']+second['log_loss']-base['log_loss'],
                'S_macro_f1':joint['macro_f1']-first['macro_f1']-second['macro_f1']+base['macro_f1'],
                'S_negative_brier':-joint['brier']+first['brier']+second['brier']-base['brier']})
    output.mkdir(parents=True)
    for name,values in (('feature_family_results',rows),('conditional_incremental',increments),('error_complementarity',pairs),('interaction_results',interactions)):
        write(output/f'{name}.csv',values)
    np.savez_compressed(output/'core_oof_predictions.npz',labels=y,users=u,trials=trials,**probabilities)
    (output/'split_trial_ids.json').write_text(json.dumps(splits,indent=2))
    (output/'run_manifest.json').write_text(json.dumps({'dataset':'epn612','phase':'source_nested_oof','source_run':source.name,
        'source_oof_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'families':list(CORE),'specs':SPECS,
        'target_users_opened':False,'classifier_or_family_fit':False,'probability_calibration':'existing per-outer-fold inner-source OOF temperatures',
        'scope':'predictive conditional proxy under fixed uniform multiview late fusion; not concat-classifier retraining or mutual information',
        'interaction_selection':'only X1/frequency and X1/CSP; prespecified theoretical pairs with prior source OOF disagreement0.5613/0.3573; no all-pairs search'},indent=2))
    print(json.dumps({'status':'ok','models':len(probabilities),'increment_rows':len(increments),'interaction_rows':len(interactions)}))


def replay(source,output):
    manifest=json.loads((output/'run_manifest.json').read_text())
    if hashlib.sha256((source/'oof_predictions.npz').read_bytes()).hexdigest()!=manifest['source_oof_sha256']:
        raise AssertionError('Changed source OOF provenance')
    with np.load(source/'oof_predictions.npz',allow_pickle=False) as source_saved,np.load(output/'core_oof_predictions.npz',allow_pickle=False) as saved:
        error=0.
        for key in ('labels','users','trials'):np.testing.assert_array_equal(source_saved[key],saved[key])
        for name,members in SPECS.items():
            p=sum(source_saved[f'calibrated_{n}'] for n in members)/len(members)
            np.testing.assert_allclose(p,saved[name],atol=1e-10,rtol=1e-9)
            error=max(error,float(np.abs(p-saved[name]).max()))
    audit={'status':'ok','probability_arrays_checked':len(SPECS),'max_absolute_probability_error':error,
        'classifier_or_family_fit':False,'target_users_opened':False}
    (output/'replay_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);p.add_argument('--replay',action='store_true')
    a=p.parse_args();replay(a.source,a.output) if a.replay else run(a.source,a.output)
