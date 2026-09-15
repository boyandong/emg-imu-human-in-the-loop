"""Matched personal-source G5/DTW errors on complete labelled gesture bouts.

Oracle bout boundaries are available offline only. This experiment is not a
short-window detector or a claim about unsegmented streaming recognition.
"""
from pathlib import Path
import argparse
import hashlib
import json
import pickle
import numpy as np
from emgimu.datasets.benchmark import load_benchmark_trial
from emgimu.datasets.unibo_baseline import HAND_NAMES
from .core import FeatureBatch
from .validated_unibo import ValidatedUniBoFamily
from .temporal import TemporalTemplateFamily, CompleteSequenceBatch
from .unibo_full_fusion import classifier
from .unibo_study import _metrics
from .force_nested_oof import fit_temperature, temperature_probability
from .unibo_temporal_complementarity import complementarity
from .core_incremental_oof import write
from .screening import SEED


def contiguous_bouts(labels, minimum_samples=200):
    """Never join separated repetitions of the same gesture."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or minimum_samples < 200:
        raise ValueError('Require a one-dimensional timeline and at least one second')
    edges = np.r_[0, np.flatnonzero(np.diff(labels) != 0) + 1, len(labels)]
    return [(int(a), int(b), int(labels[a])) for a, b in zip(edges[:-1], edges[1:])
            if b-a >= minimum_samples and int(labels[a]) in range(4)]


def envelope_path(signal, frames=32):
    """RMS of the entire bout in equal-duration bins, not sparse crops."""
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 2 or signal.shape[1] != 4 or len(signal) < 200:
        raise ValueError('A complete native four-muscle bout of >=1 second is required')
    if frames < 2 or frames > len(signal) or not np.isfinite(signal).all():
        raise ValueError('Invalid full-bout envelope')
    return np.stack([np.sqrt(np.mean(part**2, axis=0)) for part in np.array_split(signal, frames)])


def load_bouts(root, days=range(1,7)):
    requested={int(day) for day in days}
    if not requested or requested-set(range(1,9)):
        raise ValueError('Explicit days must be a nonempty subset of 1..8')
    bouts = []
    for path in sorted((root/'trials').rglob('*.npz')):
        relative = path.relative_to(root/'trials').parts
        if len(relative) < 2 or relative[1].lower() not in {f'd{day:02}' for day in requested}:
            continue
        trial = load_benchmark_trial(path, expected_channels=4, expected_rate_hz=200.)
        if not trial.benchmark_eligible:
            continue
        trial_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        for start, end, label in contiguous_bouts(trial.hand_label):
            signal = trial.emg[start:end]
            # Existing G5 still uses its established 200-ms windows. Average
            # every contiguous nonoverlapping complete window within this bout.
            windows = signal[:len(signal)//40*40].reshape(-1, 40, 4)
            bouts.append(dict(id=f'{trial.trial_id}:{start}:{end}', trial=trial.trial_id,
                user=trial.subject_id, day=int(trial.session_id[1:]), posture=trial.posture_label,
                label=label, start=start, end=end, duration_seconds=(end-start)/200.,
                windows=windows.astype(np.float32), path=envelope_path(signal),
                raw_sha256=trial_hash))
    if not bouts:
        raise ValueError('No complete eligible labelled bouts')
    return bouts


def bout_weights(bouts):
    """Equal users, then source trials, then observed bout labels, then bouts."""
    weights = np.empty(len(bouts))
    users = sorted({b['user'] for b in bouts})
    for user in users:
        trials = {b['trial'] for b in bouts if b['user'] == user}
        for trial in trials:
            labels = {b['label'] for b in bouts if b['trial'] == trial}
            for label in labels:
                selected = [i for i,b in enumerate(bouts) if b['trial'] == trial and b['label'] == label]
                weights[selected] = 1/(len(users)*len(trials)*len(labels)*len(selected))
    return weights * len(bouts)


def g5_features(family, bouts):
    lengths = [len(b['windows']) for b in bouts]
    values = family.transform(FeatureBatch(np.concatenate([b['windows'] for b in bouts]), 200.))
    return np.stack([part.mean(0) for part in np.split(values, np.cumsum(lengths)[:-1])])


def complete_paths(bouts):
    return CompleteSequenceBatch(np.stack([b['path'] for b in bouts]), 1.,
        durations_seconds=np.array([b['duration_seconds'] for b in bouts]), full_coverage=True)


def fit_pair(bouts):
    if {b['label'] for b in bouts} != set(range(4)):
        raise ValueError('Matched four-class source is required')
    w = bout_weights(bouts)
    windows = np.concatenate([b['windows'] for b in bouts])
    labels = np.concatenate([np.full(len(b['windows']), b['label']) for b in bouts])
    window_weights = np.concatenate([np.full(len(b['windows']), v/len(b['windows'])) for b,v in zip(bouts,w)])
    family = ValidatedUniBoFamily('G5').fit(FeatureBatch(windows,200.), labels, window_weights)
    scaler, model = classifier(g5_features(family,bouts), np.array([b['label'] for b in bouts]), w)
    # Bounded deterministic candidate selection is source-only. Retain every
    # chosen bout ID; a medoid of this subset is not a medoid of all source data.
    rng = np.random.default_rng(SEED)
    selected = np.concatenate([rng.permutation([i for i,b in enumerate(bouts) if b['label']==h])[:5] for h in range(4)])
    paths = complete_paths([bouts[i] for i in selected])
    template = TemporalTemplateFamily(.1).fit(paths, np.array([bouts[i]['label'] for i in selected]))
    distances = template.transform(paths)
    distance_scale = max(float(np.median(distances)), 1e-6)
    return dict(g5=(family,scaler,model), dtw=(template,distance_scale),
        candidate_ids=[bouts[i]['id'] for i in selected])


def predict_pair(state,bouts):
    before = pickle.dumps(state)
    family,scaler,model = state['g5']
    g5 = model.predict_proba(scaler.transform(g5_features(family,bouts)))
    template,scale = state['dtw']
    distances = template.transform(complete_paths(bouts))
    logits = -distances.astype(float)/scale
    logits -= logits.max(1,keepdims=True)
    dtw = np.exp(logits);dtw /= dtw.sum(1,keepdims=True)
    if before != pickle.dumps(state):
        raise AssertionError('Prediction mutated source state')
    return dict(G5=g5, DTW=dtw)


def run(root,output):
    if output.exists():
        raise FileExistsError(output)
    print('[1/3] complete native bouts; only Days 1-6 opened',flush=True)
    bouts = load_bouts(root)
    states, temperatures, predictions, sources, splits = {}, {}, {}, {}, []
    target = [b for b in bouts if b['day']==6]
    users = sorted({b['user'] for b in target})
    for user in users:
        print(f'[2/3] user {user}: source 1-4, temperature 5, refit 1-5; validate 6',flush=True)
        inner = [b for b in bouts if b['user']==user and b['day']<=4]
        cal = [b for b in bouts if b['user']==user and b['day']==5]
        train = inner + cal
        ev = [b for b in target if b['user']==user]
        trial_sets = [{b['trial'] for b in x} for x in (inner,cal,ev)]
        if any(a & b for i,a in enumerate(trial_sets) for b in trial_sets[i+1:]):
            raise AssertionError('Source/calibration/evaluation trial leakage')
        inner_state = fit_pair(inner)
        source_prob = predict_pair(inner_state,cal)
        t = {name:fit_temperature(p,np.array([b['label'] for b in cal])) for name,p in source_prob.items()}
        full_state = fit_pair(train)
        raw = predict_pair(full_state,ev)
        states[user] = (inner_state,full_state)
        temperatures[user] = t
        sources[user] = source_prob
        predictions[user] = {name:temperature_probability(p,t[name]) for name,p in raw.items()}
        splits.append(dict(user=user,train=sorted(trial_sets[0]),calibration=sorted(trial_sets[1]),
            evaluation=sorted(trial_sets[2]),source_template_candidates=full_state['candidate_ids']))
    ordered = [b for user in users for b in target if b['user']==user]
    y = np.array([b['label'] for b in ordered]);u = np.array([b['user'] for b in ordered])
    posture = np.array([b['posture'] for b in ordered]);w = bout_weights(ordered)
    probs = {name:np.concatenate([predictions[user][name] for user in users]) for name in ('G5','DTW')}
    cells = [('ALL','ALL',np.ones(len(y),bool))]
    cells += [(user,'ALL',u==user) for user in users]
    cells += [('ALL',f'posture_{p}',posture==p) for p in sorted(set(posture))]
    cells += [('ALL',f'class_{HAND_NAMES[h]}',y==h) for h in range(4)]
    scores, errors = [], []
    for user,condition,mask in cells:
        if not mask.any():
            continue
        common = dict(dataset='unibo_inail',phase='validation',subject=user,condition=condition,
            calibration_budget=0,protocol='matched personal source Days1-5; complete oracle-labelled bouts',
            evaluation_bouts=int(mask.sum()),evaluation_trials=len({b['trial'] for b,m in zip(ordered,mask) if m}))
        for name in probs:
            scores.append({**common,'feature_family':name,'method':name,
                'feature_dimension':len(states[users[0]][1]['g5'][0].feature_names) if name=='G5' else 4,
                **_metrics(y[mask],probs[name][mask],w[mask])})
        errors.append({**common,'family_a':'validated_G5_bout_mean','family_b':'full_bout_DTW',
            **complementarity(y[mask],probs['G5'][mask],probs['DTW'][mask],w[mask])})
    print('[3/3] saving source states and independent bout evidence',flush=True)
    output.mkdir(parents=True)
    write(output/'feature_family_results.csv',scores)
    write(output/'error_complementarity.csv',errors)
    (output/'split_trial_ids.json').write_text(json.dumps(splits,indent=2))
    with (output/'fitted_states.pkl').open('wb') as handle:
        pickle.dump((states,temperatures),handle)
    np.savez_compressed(output/'heldout_predictions.npz',**probs,labels=y,subjects=u,posture=posture,
        bout_ids=np.array([b['id'] for b in ordered]),weights=w)
    source_arrays = {}
    for user in users:
        cal = [b for b in bouts if b['user']==user and b['day']==5]
        source_arrays[f'{user}_labels'] = np.array([b['label'] for b in cal])
        source_arrays[f'{user}_bout_ids'] = np.array([b['id'] for b in cal])
        for name in ('G5','DTW'):
            source_arrays[f'{user}_{name}'] = sources[user][name]
    np.savez_compressed(output/'source_calibration_predictions.npz',**source_arrays)
    metadata = [{k:v for k,v in b.items() if k not in ('path','windows')} for b in bouts]
    (output/'bout_metadata.json').write_text(json.dumps(metadata,indent=2))
    manifest = dict(dataset='unibo_inail',phase='validation',seed=SEED,train_days=[1,2,3,4,5],
        target_days=[6],final_days_opened=False,temperatures=temperatures,
        full_bout_minimum_seconds=1.,envelope_frames=32,DTW_band_fraction=.1,
        medoid_candidates_per_class=5,G5='existing validated 40-sample G5, averaged over contiguous bout windows',
        limitation='Oracle labelled bout boundaries; phase-normalized complete trajectories. Both models have identical personal historical source data. Not comparable to prior short-window F1 and not a streaming detector. Candidate-subset medoids are not full-population medoids.',
        source_sha256={b['trial']:b['raw_sha256'] for b in bouts if b['day']<=5},
        target_sha256={b['trial']:b['raw_sha256'] for b in target})
    (output/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    # Independently replay from the saved source models and raw retained bouts.
    loaded_states,loaded_t = pickle.loads((output/'fitted_states.pkl').read_bytes())
    replayed = 0
    for user in users:
        cal = [b for b in bouts if b['user']==user and b['day']==5]
        cp = predict_pair(loaded_states[user][0],cal)
        for name in cp:
            np.testing.assert_allclose(cp[name],sources[user][name],atol=1e-12,rtol=0)
            np.testing.assert_allclose(fit_temperature(cp[name],np.array([b['label'] for b in cal])),
                loaded_t[user][name],atol=1e-12,rtol=0)
        ev = [b for b in ordered if b['user']==user]
        raw = predict_pair(loaded_states[user][1],ev)
        for name in raw:
            np.testing.assert_allclose(temperature_probability(raw[name],loaded_t[user][name]),
                predictions[user][name],atol=1e-12,rtol=0)
            replayed += 1
    (output/'replay_audit.json').write_text(json.dumps(dict(status='ok',prediction_arrays=replayed,
        source_state_immutable=True,target_bouts=len(y),score_rows=len(scores),pair_rows=len(errors)),indent=2))
    print(json.dumps({'status':'ok','target_bouts':len(y),'pair_rows':len(errors),'replayed_arrays':replayed}))


def replay(root,output):
    """Reload native data and saved source states; never retrain a classifier."""
    bouts = load_bouts(root)
    metadata = json.loads((output/'bout_metadata.json').read_text())
    current = [{k:v for k,v in b.items() if k not in ('path','windows')} for b in bouts]
    if current != metadata:
        raise AssertionError('Native data, bout boundaries or ordering changed')
    states,temperatures = pickle.loads((output/'fitted_states.pkl').read_bytes())
    source_arrays = {}
    target_arrays = {'G5':[], 'DTW':[]}
    target_order = []
    for user in sorted(states):
        cal = [b for b in bouts if b['user']==user and b['day']==5]
        ev = [b for b in bouts if b['user']==user and b['day']==6]
        cp = predict_pair(states[user][0],cal)
        raw = predict_pair(states[user][1],ev)
        source_arrays[f'{user}_labels'] = np.array([b['label'] for b in cal])
        source_arrays[f'{user}_bout_ids'] = np.array([b['id'] for b in cal])
        for name in cp:
            np.testing.assert_allclose(fit_temperature(cp[name],source_arrays[f'{user}_labels']),
                temperatures[user][name],atol=1e-12,rtol=0)
            source_arrays[f'{user}_{name}'] = cp[name]
            target_arrays[name].append(temperature_probability(raw[name],temperatures[user][name]))
        target_order.extend(b['id'] for b in ev)
    with np.load(output/'heldout_predictions.npz',allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['bout_ids'],target_order)
        for name in target_arrays:
            np.testing.assert_allclose(np.concatenate(target_arrays[name]),saved[name],atol=1e-12,rtol=0)
    source_path = output/'source_calibration_predictions.npz'
    if source_path.exists():
        with np.load(source_path,allow_pickle=False) as saved:
            if set(saved.files) != set(source_arrays):
                raise AssertionError('Source probability coverage changed')
            for name,values in source_arrays.items():
                np.testing.assert_array_equal(saved[name],values)
    else:
        # Recover inspectable probabilities from already saved inner source
        # states, without repeating training or changing an experimental rule.
        np.savez_compressed(source_path,**source_arrays)
    manifest_path = output/'run_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['source_sha256'] = {b['trial']:b['raw_sha256'] for b in bouts if b['day']<=5}
    manifest_path.write_text(json.dumps(manifest,indent=2))
    result = dict(status='ok',prediction_arrays=2*len(states),source_probability_arrays=2*len(states),
        source_temperatures_recomputed=2*len(states),native_data_reloaded=True,
        source_state_immutable=True,classifier_retrained=False,target_bouts=len(target_order),
        scope='All source/target model predictions and source temperatures replayed from saved states and reloaded native bouts; not streaming accuracy')
    (output/'replay_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('dataset',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--replay',action='store_true')
    args=parser.parse_args();replay(args.dataset,args.output) if args.replay else run(args.dataset,args.output)
