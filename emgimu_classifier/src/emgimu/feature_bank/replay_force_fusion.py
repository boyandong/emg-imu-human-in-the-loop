"""Replay every force-bank variant using trusted local fitted states only."""
from pathlib import Path
import argparse
import json
import pickle
import numpy as np
from emgimu.datasets.libemg_force import load_libemg_force_windows
from .force_full_fusion import IDS, aggregate
from .calibration import ReliabilityWeights, late_fusion


def replay(root: Path, run_root: Path) -> dict:
    states, anchors = pickle.loads((run_root/'fitted_states.pkl').read_bytes())
    before = pickle.dumps((states, anchors))
    manifest = json.loads((run_root/'run_manifest.json').read_text())
    splits = json.loads((run_root/'split_trial_ids.json').read_text())
    probability_calibration=run_root/'probability_calibration.json'
    temperatures=json.loads(probability_calibration.read_text())['temperatures'] if probability_calibration.exists() else {}
    users = sorted({s['user'] for s in splits})
    calibration = load_libemg_force_windows(root, subjects=users, conditions=('Ramp',))
    target = load_libemg_force_windows(root, subjects=users, conditions=manifest['evaluation_force'])
    features = {}; raw = {}; cal_features = {}
    for name in IDS:
        family, scaler, model = states[name]
        c, cy, cu, cal_trials = aggregate(family.transform(calibration.batch), calibration)
        b, y, u, trials = aggregate(family.transform(target.batch), target)
        cal_features[name] = scaler.transform(c)
        features[name] = scaler.transform(b)
        raw[name] = model.predict_proba(features[name])
        if temperatures:
            from .force_nested_oof import temperature_probability
            raw[name]=temperature_probability(raw[name],temperatures[name])
    population = np.ones(len(IDS))/len(IDS)
    reliability = ReliabilityWeights(tuple(range(7)), IDS, population, n0=manifest['n0'])
    checked = set(); error = 0.
    with np.load(run_root/'heldout_predictions.npz', allow_pickle=False) as saved:
        for key, actual in (('labels', y), ('users', u), ('trials', trials)):
            np.testing.assert_array_equal(saved[key], actual)
        for split in splits:
            user, shots = split['user'], split['shots']
            cal = np.flatnonzero(np.isin(cal_trials, split['calibration']))
            ev = np.flatnonzero(np.isin(trials, split['evaluation']))
            if len(cal) != shots*7 or set(cal_trials[cal]) & set(trials[ev]):
                raise AssertionError('invalid calibration partition')
            if set(cal_trials[cal]) != set(split['calibration']) or set(trials[ev]) != set(split['evaluation']):
                raise AssertionError('missing trial identifiers')
            w = population if not shots else reliability.personal({n:(cal_features[n][cal], cy[cal]) for n in IDS})
            personal = {n:raw[n][ev] for n in IDS}
            if shots:
                for n in IDS:
                    anchor, temp = anchors[(user, shots, n)]
                    expected_temp = max(float(np.median(anchor.transform(cal_features[n][cal])[:,:7])), 1e-10)
                    np.testing.assert_allclose(temp, expected_temp)
                    logits = -anchor.transform(features[n][ev])[:,:7].astype(float)/temp
                    logits -= logits.max(1, keepdims=True)
                    p = np.exp(logits); p /= p.sum(1, keepdims=True)
                    alpha = shots/(shots+2)
                    personal[n] = (1-alpha)*personal[n]+alpha*p
            variants = {'full':(personal,w), 'without_F7_anchor':({n:raw[n][ev] for n in IDS},w),
                'uniform_population':({n:raw[n][ev] for n in IDS},population),
                'baseline_F0':({'F0':raw['F0'][ev]},population),
                **{f'without_{n}':({f:p for f,p in personal.items() if f!=n},w) for n in IDS}}
            for method, (providers, weights) in variants.items():
                key = f'{user}_{shots}_{method}'
                p = late_fusion(providers, IDS, weights)
                np.testing.assert_allclose(p, saved[key], rtol=1e-7, atol=1e-8, err_msg=key)
                error = max(error, float(np.abs(p-saved[key]).max())); checked.add(key)
        if checked != set(saved.files)-{'labels','trials','users'}:
            raise AssertionError('not every saved variant was replayed')
    if pickle.dumps((states, anchors)) != before:
        raise AssertionError('replay mutated fitted states')
    result = {'status':'ok', 'run_id':run_root.name, 'prediction_arrays_checked':len(checked),
        'max_absolute_probability_error':error, 'classifier_or_family_fit':False,
        'temperature_verified_calibration_only':True,'source_oof_probability_calibration':bool(temperatures)}
    (run_root/'replay_audit.json').write_text(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root', type=Path); p.add_argument('run_root', type=Path)
    a = p.parse_args(); print(json.dumps(replay(a.root, a.run_root)))
