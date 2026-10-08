"""Frozen new-cohort confirmation of independent-trial personal covariance."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
from scipy.special import softmax
from sklearn.preprocessing import StandardScaler
from benchmarks.new_bank_v3.mahalanobis_epn_budget_v1 import extract, sha
from emgimu.datasets.epn612 import load_epn612_windows
from emgimu.feature_bank.document_scale_v3 import DocumentScalePatternV3
from emgimu.feature_bank.calibration import DocumentPersonalAnchorV2
from emgimu.feature_bank.trial_mahalanobis_v1 import TrialMahalanobisAnchorV1
from emgimu.feature_bank.epn_study import _metrics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROTOCOL = HERE / 'MAHALANOBIS_EPN_HOLDOUT_V2_PROTOCOL.json'
RESULT = HERE / 'MAHALANOBIS_EPN_HOLDOUT_V2_RESULTS.json'
DEPENDENCIES = ['benchmarks/new_bank_v3/mahalanobis_epn_holdout_v2.py',
                'benchmarks/new_bank_v3/mahalanobis_epn_budget_v1.py',
                'src/emgimu/feature_bank/document_scale_v3.py',
                'src/emgimu/feature_bank/trial_mahalanobis_v1.py',
                'src/emgimu/feature_bank/calibration.py', 'src/emgimu/datasets/epn612.py',
                'src/emgimu/feature_bank/epn_study.py']


def prepare():
    parent = json.loads((HERE / 'MAHALANOBIS_EPN_BUDGET_V1_PROTOCOL.json').read_text())
    protocol = {
        'version': 'mahalanobis-epn-holdout-v2', 'archive': parent['archive'],
        'archive_sha256': parent['archive_sha256'], 'source_users': parent['source_users'],
        'target_users': list(range(32, 42)), 'budgets': [10, 20], 'dimension': 8,
        'reserved_calibration_per_class': 20, 'covariance_shrinkage': .2,
        'selection_seed': '20261006',
        'selection': 'Same SHA256(seed|native_trial_id) order as V1. First20 reserved per class; prefixes10/20 used for calibration, remaining trials fixed for evaluation.',
        'feature': 'Source-only DocumentScalePatternV3 eight coordinates, trial means and source-only StandardScaler. No source or target dimensionality search.',
        'probability': 'Fixed V1 softmax(-distance/calibration median distance) per metric; no target evaluation temperature fitting or source-OOF calibration claim.',
        'prior_results_sha256': sha(HERE / 'MAHALANOBIS_EPN_BUDGET_V1_RESULTS.json'),
        'source_sha256': {name: sha(ROOT / name) for name in DEPENDENCIES},
        'scope': 'New precommitted evaluation on EPN users32-41, disjoint from checked source/validation/final cohorts1-31 and earlier public user107 sensor inspection. Not proof of every historical file-access event, full 612-user efficacy, own-device or low calibration burden. Protocol/source frozen before target signal loading; no default promotion.'}
    if PROTOCOL.exists():
        raise FileExistsError('Frozen protocol already exists')
    PROTOCOL.write_text(json.dumps(protocol, indent=2) + '\n', encoding='utf8')
    print('Frozen new-cohort 10/20-shot, eight-coordinate protocol', flush=True)


def run():
    if RESULT.exists():
        raise FileExistsError('Refuse to replace holdout results')
    p = json.loads(PROTOCOL.read_text())
    if any(sha(ROOT / name) != expected for name, expected in p['source_sha256'].items()):
        raise ValueError('Precommitted experiment implementation changed')
    if sha(Path(p['archive'])) != p['archive_sha256']:
        raise ValueError('Native archive changed')
    print('1/3 Source feature/scaler fit', flush=True)
    source = load_epn612_windows(p['archive'], users=p['source_users'])
    family = DocumentScalePatternV3().fit(source.batch)
    sx, _, _, source_ids = extract({'pattern': family}, source)
    scaler = StandardScaler().fit(sx['pattern'])
    frozen = pickle.dumps((family, scaler))
    target = load_epn612_windows(p['archive'], users=p['target_users'])
    values, labels, users, ids = extract({'pattern': family}, target)
    x = scaler.transform(values['pattern'])
    if set(source_ids) & set(ids) or x.shape[1] != p['dimension']:
        raise ValueError('Native source overlap or feature dimension changed')
    blocks = []
    for user in p['target_users']:
        selections = {}
        evaluation = []
        for label in range(6):
            candidates = np.flatnonzero((users == user) & (labels == label))
            ordered = sorted(candidates, key=lambda i: hashlib.sha256(
                f'{p["selection_seed"]}|{ids[i]}'.encode()).digest())
            if len(ordered) <= p['reserved_calibration_per_class']:
                raise ValueError('Independent calibration plus evaluation trials required')
            selections[label] = ordered[:20]
            evaluation.extend(ordered[20:])
        evaluation = np.array(sorted(evaluation), dtype=int)
        for shots in p['budgets']:
            calibration = np.array([i for h in range(6) for i in selections[h][:shots]], dtype=int)
            euclidean = DocumentPersonalAnchorV2(metric='euclidean').fit(x[calibration], labels[calibration])
            mahal = TrialMahalanobisAnchorV1(p['covariance_shrinkage']).fit(
                x[calibration], labels[calibration], trial_ids=ids[calibration])
            fitted = pickle.dumps((euclidean, mahal))
            distances = {'euclidean': euclidean.transform(x[evaluation])[:, :6].astype(float),
                         'mahalanobis': mahal.transform(x[evaluation], trial_ids=ids[evaluation])[:, :6].astype(float)}
            scales = {'euclidean': euclidean.similarity_scale_, 'mahalanobis': mahal.anchor_.similarity_scale_}
            probability = {name: softmax(-d/scales[name], axis=1) for name, d in distances.items()}
            if fitted != pickle.dumps((euclidean, mahal)):
                raise ValueError('Evaluation mutated calibrated state')
            blocks.append({'user': user, 'shots': shots, 'calibration_ids': ids[calibration].tolist(),
                           'evaluation_ids': ids[evaluation].tolist(), 'labels': labels[evaluation].tolist(),
                           'calibration_features': x[calibration].tolist(), 'calibration_labels': labels[calibration].tolist(),
                           'evaluation_features': x[evaluation].tolist(), 'scales': scales,
                           'distances': {name: d.tolist() for name, d in distances.items()},
                           'probabilities': {name: v.tolist() for name, v in probability.items()},
                           'scores': {name: _metrics(labels[evaluation], v, np.ones(len(evaluation))) for name, v in probability.items()}})
        print(f'2/3 User {user}: complete', flush=True)
    scores = {}
    for shots in p['budgets']:
        chosen = [b for b in blocks if b['shots'] == shots]
        truth = np.concatenate([b['labels'] for b in chosen])
        scores[str(shots)] = {name: _metrics(truth, np.concatenate([b['probabilities'][name] for b in chosen]), np.ones(len(truth)))
                             for name in ('euclidean', 'mahalanobis')}
    if frozen != pickle.dumps((family, scaler)):
        raise ValueError('Target processing mutated source state')
    result = {'schema': 'mahalanobis_epn_holdout_v2', 'protocol_sha256': sha(PROTOCOL),
              'source_trial_ids': source_ids.tolist(), 'source_state_immutable': True,
              'blocks': blocks, 'scores': scores, 'default_promoted': False, 'scope': p['scope']}
    RESULT.write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
    print('3/3 Holdout result saved', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    prepare() if args.prepare else run()
