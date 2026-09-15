"""Read trusted local fitted wearing states; do not refit or change old manifests."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

RUNS = tuple(f'feature_bank_wearing_{kind}_{phase}_20260916'
             for kind in ('core_raw_ring', 'document_core') for phase in ('validation', 'final'))


def describe(value):
    if isinstance(value, np.ndarray):
        return {'shape': list(value.shape), 'dtype': str(value.dtype),
                'sha256_contiguous_values': hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (tuple, list)):
        return [describe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): describe(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f'Unreviewed fitted attribute type: {type(value)}')


def audit(root, output):
    workspace = Path(__file__).resolve().parents[3]
    expected = workspace / 'work' / 'benchmark_runs'
    if root.resolve() != expected.resolve() or not output.resolve().is_relative_to(workspace):
        raise ValueError('Only known trusted local project runs and workspace output supported')
    results = []
    for run in RUNS:
        parent = root / run
        path = parent / 'fitted_states.pkl'
        original = path.read_bytes()
        states = pickle.loads(original)  # Explicitly limited to our four local outputs.
        before = pickle.dumps(states)
        records = []
        for key, (families, models) in sorted(states.items(), key=lambda item: str(item[0])):
            for name, family in families.items():
                records.append({'state_key': list(key), 'family': name,
                    'type': type(family).__module__ + '.' + type(family).__name__,
                    'saved_attributes': describe(vars(family))})
            for name, (scaler, model) in models.items():
                records.append({'state_key': list(key), 'composition': name,
                    'scaler_type': type(scaler).__name__, 'scaler_parameters': describe(scaler.get_params()),
                    'scaler_fitted_attributes': describe(vars(scaler)),
                    'classifier_type': type(model).__name__, 'classifier_parameters': describe(model.get_params()),
                    'classifier_fitted_attributes': describe(vars(model))})
        if before != pickle.dumps(states) or path.read_bytes() != original:
            raise AssertionError('Saved source state changed during inspection')
        results.append({'run_id': run, 'saved_state_sha256': hashlib.sha256(original).hexdigest(),
            'run_manifest_sha256': hashlib.sha256((parent / 'run_manifest.json').read_bytes()).hexdigest(),
            'split_trial_ids_sha256': hashlib.sha256((parent / 'split_trial_ids.json').read_bytes()).hexdigest(),
            'source_states_immutable': True, 'fitted_states': len(states), 'records': records})
    payload = {'status': 'saved_parameters_inspected_no_refit', 'completion_proven': False,
        'runs': results, 'total_fitted_states': sum(run['fitted_states'] for run in results),
        'limitations': ['Saved fitted attributes prove these object values only',
            'Current inspection is not a reconstruction of the original Python environment',
            'Array hashes describe state but do not prove the historical fitting data',
            'Raw prediction replay and source trial identities remain separate evidence',
            'Historical manifests are unchanged; no new scores or model choices']}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps({'status': payload['status'], 'runs': len(results),
        'total_fitted_states': payload['total_fitted_states'],
        'configuration_records': sum(len(run['records']) for run in results)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    audit(args.root, args.output)
