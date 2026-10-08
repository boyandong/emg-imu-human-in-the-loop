"""Replay frozen native probabilities through the opt-in availability interface."""
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
from emgimu.feature_bank.available_bank_fusion_v1 import AvailableBankFusionPolicyV1

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'feature_bank/AVAILABLE_BANK_FUSION_ACCEPTANCE_V1.json'
PROVIDERS = ('TD24', 'pattern', 'SPD', 'log_bands')


def run():
    source = HERE / 'F8_CALIBRATED_MANUS_V2_RESULTS.json'
    native = json.loads(source.read_text(encoding='utf8'))
    rows = []; invalid = 0
    for block in native['blocks']:
        policy = AvailableBankFusionPolicyV1(tuple(range(6)), PROVIDERS, (.25,) * 4,
            1., 1., 'imported_native_f8:' + native['protocol_sha256'], tuple(sorted(set(block['source_trials']))))
        state = policy.from_fitted_weights(block['weights'], calibration_trials=tuple(block['calibration_trials']))
        probabilities = dict(zip(PROVIDERS, np.asarray(block['providers'])))
        trials = tuple(block['evaluation_trials']); before = pickle.dumps((policy, state, probabilities))
        def call(values, ids=trials, axes=None, provider_ids=None):
            return policy.predict(state, values, evaluation_trials=ids,
                provider_trial_ids={name:trials for name in values} if provider_ids is None else provider_ids,
                provider_classes={name:tuple(range(6)) for name in values} if axes is None else axes)
        full = call(probabilities)
        error = float(np.max(abs(full['probabilities'] - np.array(block['probabilities']['F8']))))
        if error > 1e-12: raise ValueError('Frozen native fusion changed')
        dropped = []
        for missing in PROVIDERS:
            values = {name:q for name,q in probabilities.items() if name != missing}
            result = call(values)
            selected = [i for i,name in enumerate(PROVIDERS) if name != missing]
            expected = sum(block['weights'][i] * np.array(block['providers'][i]) for i in selected) / sum(block['weights'][i] for i in selected)
            delta = float(np.max(abs(result['probabilities'] - expected)))
            if delta > 1e-12 or result['missing_at_prediction'] != (missing,): raise ValueError('Missing-provider normalization failed')
            dropped.append({'missing_provider':missing, 'remaining_providers':list(result['provider_names']),
                            'renormalized_weights':list(result['weights']), 'max_probability_error':delta})
        for kwargs in (
            {'ids':(state.calibration_trials[0],) + trials[1:]},
            {'ids':(state.source_trials[0],) + trials[1:]},
            {'axes':{name:tuple(reversed(range(6))) for name in probabilities}},
            {'provider_ids':{name:tuple(reversed(trials)) for name in probabilities}}):
            try: call(probabilities, **kwargs)
            except ValueError: invalid += 1
            else: raise ValueError('Invalid native replay accepted')
        if pickle.dumps((policy, state, probabilities)) != before: raise ValueError('Inference mutated source state')
        rows.append({'phase':block['phase'], 'user':block['user'], 'shots_per_class':block['shots'],
                     'evaluation_trials':len(trials), 'full_max_probability_error':error,
                     'missing_provider_cases':dropped, 'state_immutable':True})
    paths = [source, Path(__file__), ROOT/'src/emgimu/feature_bank/available_bank_fusion_v1.py',
             ROOT/'src/emgimu/feature_bank/document_reliability_v2.py',
             ROOT/'tests/test_available_bank_fusion_v1.py', ROOT/'tests/test_available_bank_fusion_v1_delivery.py']
    payload = {'schema':'available_bank_fusion_acceptance_v1',
        'source_sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'records':rows, 'native_full_cases':len(rows), 'native_missing_provider_cases':4*len(rows),
        'rejected_invalid_native_calls':invalid,
        'software_scope':'Arbitrary named providers, source-only zero-shot population weights, document-exact calibration reliability with trial-mass shrinkage, missing-calibration skip, runtime missing-provider renormalization, strict per-provider trial/class axes and source/calibration/evaluation isolation.',
        'native_scope':'Frozen MANUS four-provider imported weights replay only. n0/temperature are unused placeholders in imported mode. This does not prove V1 fitted those weights, missing-sensor fault efficacy, own-device recognition, all F0-F9 integration or population-policy selection.',
        'default_promoted':False, 'physical_validation_proven':False, 'completion_proven':False}
    OUT.write_text(json.dumps(payload,indent=2)+'\n',encoding='utf8',newline='\n')
    print(f'Verified {len(rows)} frozen native fusion cases, {4*len(rows)} provider omissions and {invalid} invalid calls; no refit')


if __name__=='__main__': run()
