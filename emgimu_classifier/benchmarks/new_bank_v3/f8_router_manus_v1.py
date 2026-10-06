"""Paired F8 V3 routing screen on frozen native MANUS calibration/eval splits."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from emgimu.datasets.semg_manus import load_semg_manus_windows
from emgimu.feature_bank.manus_study import GESTURES, _aggregate, _metrics
from emgimu.feature_bank.families import LocalDetailFamily
from emgimu.feature_bank.document_scale_v3 import DocumentScalePatternV3
from emgimu.feature_bank.spec_spatial_v3 import SpecSpdTangentV3
from emgimu.feature_bank.relative_spectrum import LogBandEnergyFamily
from emgimu.feature_bank.document_session_v3 import DocumentSessionDescriptorV3
from emgimu.feature_bank.session_router_v1 import DocumentSessionRouterV1

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROTOCOL = HERE/'F8_ROUTER_MANUS_V1_PROTOCOL.json'
DESCRIPTORS = HERE/'F8_DOCUMENT_MANUS_RESULTS.json'
ARCHIVE = Path('D:/emg-imu-benchmarks/data/raw/semg_manus/semg-manus-dataset-v1.zip')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def prepare():
    parent = json.loads(DESCRIPTORS.read_text())
    p = {'version': 'f8-router-manus-v1', 'archive': str(ARCHIVE),
         'archive_sha256': parent['archive_sha256'], 'descriptor_sha256': sha(DESCRIPTORS),
         'providers': list(DocumentSessionRouterV1.providers), 'users': parent['users'],
         'source_session': 1, 'validation_session': 2, 'descriptive_final_session': 3,
         'shots': [1, 2], 'classifier_C': 1., 'max_iter': 5000, 'seed': 20261006,
         'baseline': 'TD24 = channel RMS/MAV/WL only. MANUS has no recorded Rest for the requested Rest-noise six-block F0; source-active MAD threshold blocks are omitted, not claimed as document F0.',
         'reference': 'Each provider fits source-session features and balanced logistic regression. Trial mean coordinates, equal trial mass. Existing frozen calibration/evaluation IDs; all providers share identical trials.',
         'routing': 'Mean class F8 shifts / source mean pairwise class geometry; TD24 uses absolute log activation shift plus relative channel-quality score shift. Degenerate source geometry is inactive. Weight=max(exp(-clip(risk,0,5)),.05), normalized. Equal .25 prior; no evaluation-dependent tuning.',
         'scope': 'New versioned paired retrospective native 8-channel session routing experiment. Previously inspected sessions are validation/descriptive-final, not a fresh confirmatory holdout. Relative quality shifts are not physical-fault proof. Opt-in, no device claim.'}
    if PROTOCOL.exists():
        raise FileExistsError('Refuse to replace frozen protocol')
    PROTOCOL.write_text(json.dumps(p, indent=2)+'\n', encoding='utf8')
    print('Frozen four-provider routing protocol with 24 calibration blocks', flush=True)


def extract(families, data):
    values = {}
    for name, family in families.items():
        x = family.transform(data.batch)
        if name == 'TD24':
            x = x[:, :24]
        features, labels, users, sessions, speeds, trials = _aggregate(x, data)
        values[name] = features
    return values, labels, users, trials


def run():
    p = json.loads(PROTOCOL.read_text()); archive = Path(p['archive'])
    if sha(archive) != p['archive_sha256'] or sha(DESCRIPTORS) != p['descriptor_sha256']:
        raise ValueError('Frozen native archive or descriptor changed')
    parent = json.loads(DESCRIPTORS.read_text())
    print('1/3: fitting source-session-only providers and reference geometry', flush=True)
    source = load_semg_manus_windows(archive, users=p['users'], sessions=(1,), gestures=GESTURES)
    families = {'TD24': LocalDetailFamily().fit(source.batch),
                'pattern': DocumentScalePatternV3().fit(source.batch),
                'SPD': SpecSpdTangentV3().fit(source.batch),
                'log_bands': LogBandEnergyFamily().fit(source.batch)}
    features, labels, users, trials = extract(families, source)
    models = {}
    for name, values in features.items():
        models[name] = make_pipeline(StandardScaler(), LogisticRegression(C=p['classifier_C'],
            class_weight='balanced', max_iter=p['max_iter'], random_state=p['seed'])).fit(values, labels)
        assert np.array_equal(models[name][-1].classes_, np.arange(6))
        if np.max(models[name][-1].n_iter_) >= p['max_iter']:
            raise RuntimeError('Source provider failed convergence')
    summaries = {}; routers = {}; source_ids = {}
    for user in p['users']:
        idx = np.flatnonzero(source.users == user)
        summaries[user] = DocumentSessionDescriptorV3().fit_long_term(source.batch.take(idx),
            source.labels[idx], source.trials[idx], user_id=str(user), session_ids=['1']*len(idx), ring_topology=False)
        routers[user] = DocumentSessionRouterV1().fit_source(summaries[user])
        source_ids[user] = sorted(set(source.trials[idx]))
    frozen = pickle.dumps((families, models, summaries, routers))
    blocks = []; scores = {}
    for phase, session in [('validation', 2), ('descriptive_final', 3)]:
        print(f'2/3: extracting {phase} providers, before separate frozen calibration selection', flush=True)
        target = load_semg_manus_windows(archive, users=p['users'], sessions=(session,), gestures=GESTURES)
        values, y, target_users, trial_ids = extract(families, target)
        probabilities = {name: models[name].predict_proba(x) for name, x in values.items()}
        for shots in p['shots']:
            all_y = []; collected = {'TD24': [], 'uniform': [], 'F8': []}
            for record in parent['records']:
                if record['phase'] != phase or record['shots'] != shots:
                    continue
                user = record['user']; cal = set(record['calibration_trials']); ev = set(record['evaluation_trials'])
                if cal & ev or (cal | ev) & set(source_ids[user]):
                    raise ValueError('Source/calibration/evaluation trial overlap')
                idx = np.flatnonzero((target.users == user) & np.isin(target.trials, list(cal)))
                if set(target.trials[idx]) != cal:
                    raise ValueError('Frozen calibration coverage differs')
                descriptor = summaries[user].from_calibration(target.batch.take(idx), target.labels[idx],
                    target.trials[idx], user_id=str(user), session_ids=[str(session)]*len(idx))
                np.testing.assert_allclose(descriptor['phi_session'], record['descriptor']['phi_session'], rtol=1e-8, atol=1e-9)
                evaluation = np.flatnonzero((target_users == user) & np.isin(trial_ids, list(ev)))
                if set(trial_ids[evaluation]) != ev:
                    raise ValueError('Frozen evaluation coverage differs')
                experts = np.stack([probabilities[name][evaluation] for name in p['providers']])
                routed, weight, risks = routers[user].fuse(experts, descriptor)
                arms = {'TD24': experts[0], 'uniform': experts.mean(axis=0), 'F8': routed}
                for name, probability in arms.items():
                    collected[name].append(probability)
                all_y.append(y[evaluation])
                blocks.append({'phase': phase, 'shots': shots, 'user': user, 'source_trials': source_ids[user],
                    'calibration_trials': sorted(cal), 'evaluation_trials': trial_ids[evaluation].tolist(),
                    'labels': y[evaluation].tolist(), 'providers': experts.tolist(), 'weights': weight.tolist(),
                    'risks': risks.tolist(), 'source_normalizers': routers[user].scales_,
                    'probabilities': {name: probability.tolist() for name, probability in arms.items()}})
            scores[f'{phase}_{shots}shot'] = {name: _metrics(np.concatenate(all_y), np.concatenate(parts))
                                              for name, parts in collected.items()}
            print(f'{phase} {shots}shot: uniform/F8 loss '
                  f"{scores[f'{phase}_{shots}shot']['uniform']['log_loss']:.4f}/"
                  f"{scores[f'{phase}_{shots}shot']['F8']['log_loss']:.4f}", flush=True)
    assert frozen == pickle.dumps((families, models, summaries, routers))
    evidence = ['benchmarks/new_bank_v3/f8_router_manus_v1.py', 'src/emgimu/feature_bank/session_router_v1.py',
                'src/emgimu/feature_bank/document_session_v3.py', 'src/emgimu/feature_bank/document_scale_v3.py',
                'src/emgimu/feature_bank/spec_spatial_v3.py', 'src/emgimu/feature_bank/families.py',
                'src/emgimu/feature_bank/relative_spectrum.py']
    result = {'protocol_sha256': sha(PROTOCOL), 'source_hashes': {name: sha(ROOT/name) for name in evidence},
              'source_state_sha256': hashlib.sha256(frozen).hexdigest(), 'source_state_immutable': True,
              'provider_dimensions': {name: x.shape[1] for name, x in features.items()},
              'source_trials': len(trials), 'blocks': blocks, 'scores': scores,
              'scope': p['scope'], 'default_promoted': False}
    (HERE/'F8_ROUTER_MANUS_V1_RESULTS.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
    print('3/3: wrote paired native trial probabilities and routing weights', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args(); prepare() if args.prepare else run()
