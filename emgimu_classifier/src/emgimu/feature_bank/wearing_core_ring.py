"""Frozen actual-concatenation wearing Core versus document F3c increment."""
import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from emgimu.datasets.electrode_shift import PATH_RE
from .electrode_shift_study import _metrics
from .families import LocalDetailFamily, ScalePatternFamily, CspSpatialFamily
from .force_full_fusion import aggregate
from .force_nested_oof import fit_temperature, temperature_probability
from .ring_covariance import RawRingCovarianceFamily
from .wearing_full_fusion import load

SPECS = {'Core': ('F0','F1_reference','F2b_CSP'),
         'Core_plus_F3c_raw': ('F0','F1_reference','F2b_CSP','F3c_raw')}
FACTORIES = {'F0': LocalDetailFamily, 'F1_reference': ScalePatternFamily,
             'F2b_CSP': CspSpatialFamily,
             'F3c_raw': lambda: RawRingCovarianceFamily(ring_topology=True)}


def features(data, families):
    result = {}
    for name, family in families.items():
        result[name], y, _, trials = aggregate(family.transform(data.batch), data)
    return result, y, trials


def predict(data, state):
    families, models = state
    values, y, trials = features(data, families)
    result = {}
    for name, members in SPECS.items():
        scaler, model = models[name]
        result[name] = model.predict_proba(scaler.transform(np.concatenate([values[k] for k in members],axis=1)))
    return result, y, trials


def fit(source):
    families = {name: factory().fit(source.batch, source.labels) for name,factory in FACTORIES.items()}
    values, y, _ = features(source, families)
    models = {}
    for name,members in SPECS.items():
        x = np.concatenate([values[k] for k in members],axis=1)
        scaler = StandardScaler().fit(x)
        model = LogisticRegression(C=1, class_weight='balanced',max_iter=1000, random_state=20260915)
        model.fit(scaler.transform(x), y)
        models[name] = (scaler, model)
    return families, models


def write(path, rows):
    with path.open('w', newline='',encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def run(archive, output, phase):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    states, saved, temperatures, splits = {}, {}, {}, {}
    family_rows, incremental_rows, complement_rows = [], [], []
    for user in ((15,16,17) if phase=='validation' else (18,19,20)):
        print(f'subject {user}: five source-only repetition folds, frozen Core/F3c',flush=True)
        source = load(archive,user,('training',))
        # Establish OOF fitting before opening any after-wearing target.
        _, sy, st = features(source, {'F0': LocalDetailFamily().fit(source.batch,source.labels)})
        reps = np.array([int(PATH_RE.fullmatch(t)['rep']) for t in source.trials])
        oof = {name: np.zeros((len(st),5)) for name in SPECS}
        folds = []
        for rep in sorted(set(reps)):
            a = source.take(np.flatnonzero(reps!=rep)); b = source.take(np.flatnonzero(reps==rep))
            state = fit(a); states[(user,'fold',int(rep))] = state
            p, y, trials = predict(b,state)
            index = np.array([np.flatnonzero(st==trial).item() for trial in trials])
            np.testing.assert_array_equal(sy[index],y)
            for name in SPECS:
                oof[name][index] = p[name]
                saved[f'{user}_fold{rep}_{name}'] = p[name]
            folds.append({'train':sorted(set(a.trials)), 'validation': sorted(set(b.trials))})
        states[(user,'full')] = fit(source)
        for name in SPECS:
            temperatures[f'{user}_{name}'] = fit_temperature(oof[name],sy)
            saved[f'{user}_oof_{name}'] = oof[name]
        saved[f'{user}_source_labels'] = sy; saved[f'{user}_source_trials'] = st
        target = load(archive,user,('trial_1','trial_2','trial_3','trial_4'))
        raw, y, trials = predict(target,states[(user,'full')])
        probability = {name:temperature_probability(raw[name],temperatures[f'{user}_{name}']) for name in SPECS}
        saved[f'{user}_labels'] = y; saved[f'{user}_trials'] = trials
        for name,p in probability.items(): saved[f'{user}_target_{name}'] = p
        splits[str(user)] = {'train':st.tolist(),'test':trials.tolist(),'oof':folds}
        domains = np.array([PATH_RE.fullmatch(t)['domain'] for t in trials])
        for condition in ('ALL',*sorted(set(domains))):
            mask = np.ones(len(y),bool) if condition=='ALL' else domains==condition
            base = {'dataset':'libemg_electrode_shift','phase':phase,'subject':user,
                    'condition':condition,'calibration_budget':0,'evaluation_unit':'whole_native_trial_mean',
                    'evaluation_trials':int(mask.sum()),'scope':'actual concatenation; personal Before-source; source OOF temperatures'}
            m = {name:_metrics(y[mask],p[mask]) for name,p in probability.items()}
            for name in SPECS:
                family_rows.append({**base,'feature_family':'+'.join(SPECS[name]),
                    'model':name,'feature_dimension':sum(len(states[(user,'full')][0][k].feature_names) for k in SPECS[name]),**m[name]})
            a,b = m['Core'],m['Core_plus_F3c_raw']
            incremental_rows.append({**base,'core':'F0+F1_reference+F2b_CSP','added_family':'F3c_raw',
                'delta_log_loss':a['log_loss']-b['log_loss'],'delta_brier':a['brier']-b['brier'],
                'delta_macro_f1':b['macro_f1']-a['macro_f1'],
                'core_per_class_f1_json':a['per_class_f1_json'],'increment_per_class_f1_json':b['per_class_f1_json']})
            ea = probability['Core'][mask].argmax(1)!=y[mask]; eb = probability['Core_plus_F3c_raw'][mask].argmax(1)!=y[mask]
            corr = float(np.corrcoef(ea,eb)[0,1]) if ea.std()>0 and eb.std()>0 else None
            complement_rows.append({**base,'family_a':'Core','family_b':'Core_plus_F3c_raw',
                'comparison_type':'core versus conditional increment; not standalone-family pair',
                'error_correlation':corr,'both_wrong':int((ea&eb).sum()),
                'core_wrong_increment_correct':int((ea&~eb).sum()),
                'core_correct_increment_wrong':int((~ea&eb).sum()),
                'disagreement_fraction':float(np.mean(probability['Core'][mask].argmax(1)!=probability['Core_plus_F3c_raw'][mask].argmax(1)))})
    for filename,rows in [('feature_family_results.csv',family_rows),('conditional_incremental.csv',incremental_rows),('error_complementarity.csv',complement_rows)]:
        write(output/filename,rows)
    (output/'fitted_states.pkl').write_bytes(pickle.dumps(states))
    np.savez_compressed(output/'heldout_predictions.npz',**saved)
    (output/'split_trial_ids.json').write_text(json.dumps(splits,indent=2),encoding='utf-8')
    (output/'run_manifest.json').write_text(json.dumps({'phase':phase,'specifications':SPECS,
        'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(), 'temperatures':temperatures,
        'source_protocol':'five repetition-held-out Before source folds; families/scalers/classifiers refit within each fold',
        'candidate_definition':'document F3c raw centered shrunk trace-normalized covariance; four lags times six statistics',
        'reference_boundary':'F1 reference is not validated historical X1-H; F3c is not historical RLCS/CES',
        'freeze':'C=1; CSP one component each tail/class; lambda=.05; 200ms native windows; no target selection',
        'limitations':'three personal users/four correlated wearing domains; five native classes no pinch; trial aggregation is not streaming'},indent=2),encoding='utf-8')
    replay(archive,output)


def replay(archive,output):
    manifest = json.loads((output/'run_manifest.json').read_text(encoding='utf-8'))
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=manifest['archive_sha256']: raise AssertionError('archive changed')
    states = pickle.loads((output/'fitted_states.pkl').read_bytes()); before = pickle.dumps(states)
    splits = json.loads((output/'split_trial_ids.json').read_text(encoding='utf-8'))
    count, max_error = 0,0.
    with np.load(output/'heldout_predictions.npz',allow_pickle=False) as saved:
        for user_key,split in splits.items():
            user = int(user_key); source = load(archive,user,('training',))
            target = load(archive,user,('trial_1','trial_2','trial_3','trial_4'))
            if set(split['train'])&set(split['test']): raise AssertionError('target overlap')
            reps = np.array([int(PATH_RE.fullmatch(t)['rep']) for t in source.trials])
            st = saved[f'{user}_source_trials']; sy = saved[f'{user}_source_labels']
            oof = {name:np.zeros((len(st),5)) for name in SPECS}
            for rep in sorted(set(reps)):
                b = source.take(np.flatnonzero(reps==rep))
                fold = split['oof'][sorted(set(reps)).index(rep)]
                if set(fold['train'])&set(fold['validation']) or set(fold['train'])&set(split['test']): raise AssertionError('fold leakage')
                p,y,trials = predict(b,states[(user,'fold',int(rep))])
                index = np.array([np.flatnonzero(st==t).item() for t in trials]); np.testing.assert_array_equal(sy[index],y)
                for name in SPECS:
                    oof[name][index] = p[name]
                    np.testing.assert_allclose(p[name],saved[f'{user}_fold{rep}_{name}'],rtol=0,atol=1e-12)
                    count+=1
            p,y,trials = predict(target,states[(user,'full')])
            np.testing.assert_array_equal(y,saved[f'{user}_labels']); np.testing.assert_array_equal(trials,saved[f'{user}_trials'])
            for name in SPECS:
                np.testing.assert_allclose(oof[name],saved[f'{user}_oof_{name}'],rtol=0,atol=1e-12); count+=1
                temp = fit_temperature(oof[name],sy)
                np.testing.assert_allclose(temp,manifest['temperatures'][f'{user}_{name}'],rtol=0,atol=1e-12)
                final = temperature_probability(p[name],temp)
                err = float(np.max(np.abs(final-saved[f'{user}_target_{name}'])))
                max_error = max(max_error,err)
                np.testing.assert_allclose(final,saved[f'{user}_target_{name}'],rtol=0,atol=1e-12); count+=1
    if before!=pickle.dumps(states): raise AssertionError('source state mutated')
    audit = {'status':'ok','arrays_recomputed':count,'maximum_error':max_error,'source_states_immutable':True,
             'scope':'saved-fold/native-data prediction replay; not exhaustive formula or document completion'}
    (output/'replay_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=('validation','final')); parser.add_argument('--replay',action='store_true')
    args = parser.parse_args()
    if args.replay: replay(args.archive,args.output)
    elif args.phase: run(args.archive,args.output,args.phase)
    else: parser.error('--phase is required for training')


if __name__=='__main__': main()
