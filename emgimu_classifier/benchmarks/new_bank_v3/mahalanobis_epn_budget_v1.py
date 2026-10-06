"""Native independent-trial Mahalanobis budgets, with insufficient budgets N/A."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
from scipy.special import softmax
from sklearn.preprocessing import StandardScaler
from emgimu.datasets.epn612 import load_epn612_windows
from emgimu.feature_bank.epn_study import aggregate_trials, _metrics
from emgimu.feature_bank.document_scale_v3 import DocumentScalePatternV3
from emgimu.feature_bank.spec_spatial_v3 import SpecSpdTangentV3
from emgimu.feature_bank.families import LocalDetailFamily
from emgimu.feature_bank.calibration import DocumentPersonalAnchorV2
from emgimu.feature_bank.trial_mahalanobis_v1 import TrialMahalanobisAnchorV1

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROTOCOL = HERE/'MAHALANOBIS_EPN_BUDGET_V1_PROTOCOL.json'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def prepare():
    p = {'version': 'mahalanobis-epn-budget-v1',
         'archive': 'D:/emg-imu-benchmarks/data/raw/epn612/EMG-EPN612-Dataset.zip',
         'archive_sha256': '4ee8db037385e7bee1e6ac6f9e9eea4f0869e25f7825f5eb7e5be0dff4f93c21',
         'source_users': list(range(1, 16)), 'target_users': list(range(22, 32)),
         'feature_dimensions': {'pattern': 8, 'TD24': 24, 'SPD': 36},
         'budgets': [1, 2, 5, 10, 20], 'reserved_calibration_per_class': 20,
         'selection': 'Within each target user/class, order native trial IDs by SHA256(20261006|trial); reserve first20 once. Prefixes define nested budgets; all remaining trials are identical evaluation at every budget.',
         'features': 'Source-only native eight-channel/200Hz feature state. Four windows averaged once per independent native trial; source-trial-only StandardScaler. TD24 is RMS/MAV/WL, not Rest-threshold six-block F0.',
         'metric': 'Per-class mean and covariance, .2 isotropic shrinkage plus max(trace/d*1e-8,1e-10) ridge. Mahalanobis eligible only with dimension+2 independent native calibration trials per class.',
         'probability': 'softmax(-distance/source-of-personal-fit calibration median distance), same calibration-only rule for Euclidean and Mahalanobis; descriptive distance readout, no source-CV probability tuning.',
         'scope': 'Previously inspected EPN users22-31; retrospective native budget experiment. High-dimensional 24/36 cases must remain N/A if trial budgets are insufficient. No dimensionality reduction, window-count inflation, device or blind holdout claim.'}
    if PROTOCOL.exists():
        raise FileExistsError('Refuse to replace frozen protocol')
    PROTOCOL.write_text(json.dumps(p, indent=2)+'\n', encoding='utf8')
    print('Frozen 8/24/36-coordinate independent-trial budget protocol', flush=True)


def extract(families, data):
    values = {}
    for name, family in families.items():
        x = family.transform(data.batch)
        if name == 'TD24':
            x = x[:, :24]
        a, y, users, ids, weights = aggregate_trials(x, data)
        values[name] = a
    return values, y, users, ids


def run():
    p = json.loads(PROTOCOL.read_text()); archive = Path(p['archive'])
    if sha(archive) != p['archive_sha256']:
        raise ValueError('Native archive hash differs')
    print('1/3: fitting source-only feature state and equal-trial scalers', flush=True)
    source = load_epn612_windows(archive, users=p['source_users'])
    families = {'pattern': DocumentScalePatternV3().fit(source.batch),
                'TD24': LocalDetailFamily().fit(source.batch), 'SPD': SpecSpdTangentV3().fit(source.batch)}
    source_x, source_y, source_user, source_ids = extract(families, source)
    scalers = {name: StandardScaler().fit(x) for name, x in source_x.items()}
    frozen = pickle.dumps((families, scalers))
    target = load_epn612_windows(archive, users=p['target_users'])
    raw_x, y, users, ids = extract(families, target)
    x = {name: scalers[name].transform(v) for name, v in raw_x.items()}
    if set(ids) & set(source_ids):
        raise ValueError('Source/target native trial overlap')
    blocks = []; selections = []; metrics = {}; arrays = {}; truth = {}
    for user in p['target_users']:
        selected = {}; eval_idx = []
        for label in range(6):
            rows = np.flatnonzero((users == user) & (y == label))
            ordered = sorted(rows.tolist(), key=lambda i: hashlib.sha256(f'20261006|{ids[i]}'.encode()).digest())
            if len(ordered) <= 20:
                raise ValueError('Twenty calibration and separate evaluation trials required')
            selected[label] = ordered[:20]; eval_idx += ordered[20:]
        evaluation = np.array(sorted(eval_idx), dtype=int)
        selections.append({'user': user, 'reserved_calibration': {str(h): ids[idx].tolist() for h, idx in selected.items()},
                           'evaluation': ids[evaluation].tolist()})
        print(f'2/3: user {user}, five budgets and three source feature spaces', flush=True)
        for shots in p['budgets']:
            cal = np.array([i for h in range(6) for i in selected[h][:shots]], dtype=int)
            for name in p['feature_dimensions']:
                a = x[name]; dimension = a.shape[1]
                assert dimension == p['feature_dimensions'][name]
                base = DocumentPersonalAnchorV2(metric='euclidean').fit(a[cal], y[cal])
                base_state = pickle.dumps(base)
                distances = base.transform(a[evaluation])[:, :6].astype(float)
                euclidean = softmax(-distances/base.similarity_scale_, axis=1)
                assert pickle.dumps(base) == base_state
                eligible = shots >= dimension+2
                block = {'user': user, 'shots': shots, 'family': name, 'dimension': dimension,
                         'independent_calibration_per_class': shots, 'required_trials_per_class': dimension+2,
                         'mahalanobis_eligible': eligible, 'calibration_trials': ids[cal].tolist(),
                         'evaluation_trials': ids[evaluation].tolist(), 'labels': y[evaluation].tolist(),
                         'euclidean_probability': euclidean.tolist()}
                key = f'{name}_{shots}shot_euclidean'
                arrays.setdefault(key, []).append(euclidean); truth.setdefault(key, []).append(y[evaluation])
                if eligible:
                    model = TrialMahalanobisAnchorV1(.2).fit(a[cal], y[cal], trial_ids=ids[cal])
                    state = pickle.dumps(model)
                    d = model.transform(a[evaluation], trial_ids=ids[evaluation])[:, :6].astype(float)
                    probability = softmax(-d/model.anchor_.similarity_scale_, axis=1)
                    assert pickle.dumps(model) == state
                    block.update(mahalanobis_probability=probability.tolist(), distances=d.tolist(),
                                 calibration_features=a[cal].tolist(), calibration_labels=y[cal].tolist(),
                                 evaluation_features=a[evaluation].tolist(),
                                 similarity_scale=model.anchor_.similarity_scale_, trial_counts=model.trial_counts_)
                    key = f'{name}_{shots}shot_mahalanobis'
                    arrays.setdefault(key, []).append(probability); truth.setdefault(key, []).append(y[evaluation])
                else:
                    # Exercise the actual public API guard on native data, not just a metadata comparison.
                    try:
                        TrialMahalanobisAnchorV1(.2).fit(a[cal], y[cal], trial_ids=ids[cal])
                    except ValueError as error:
                        if 'independent trials' not in str(error):
                            raise
                        block['ineligible_reason'] = str(error)
                    else:
                        raise AssertionError('Insufficient native budget was silently accepted')
                blocks.append(block)
    for key, parts in arrays.items():
        scores_y = np.concatenate(truth[key]); probabilities = np.concatenate(parts)
        metrics[key] = _metrics(scores_y, probabilities, np.ones(len(scores_y)))
    assert frozen == pickle.dumps((families, scalers))
    evidence = ['benchmarks/new_bank_v3/mahalanobis_epn_budget_v1.py', 'src/emgimu/feature_bank/trial_mahalanobis_v1.py',
                'src/emgimu/feature_bank/calibration.py', 'src/emgimu/feature_bank/document_scale_v3.py',
                'src/emgimu/feature_bank/spec_spatial_v3.py', 'src/emgimu/feature_bank/families.py', 'src/emgimu/datasets/epn612.py']
    result = {'protocol_sha256': sha(PROTOCOL), 'source_hashes': {n: sha(ROOT/n) for n in evidence},
              'source_trials': len(source_ids), 'source_state_immutable': True, 'selections': selections,
              'blocks': blocks, 'scores': metrics, 'scope': p['scope'], 'default_promoted': False}
    (HERE/'MAHALANOBIS_EPN_BUDGET_V1_RESULTS.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
    print('3/3: eligible Mahalanobis blocks', sum(b['mahalanobis_eligible'] for b in blocks), '/', len(blocks), flush=True)
    print(json.dumps({k: {m: v[m] for m in ('macro_f1', 'log_loss')} for k, v in metrics.items() if 'pattern_10' in k or 'pattern_20' in k}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args(); prepare() if args.prepare else run()
