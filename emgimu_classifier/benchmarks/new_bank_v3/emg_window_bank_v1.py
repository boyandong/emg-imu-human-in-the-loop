"""Predeclared source-only EMG window bank, additions and leave-group-out fits."""
import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from emgimu.datasets.epn612 import load_epn612_windows
from emgimu.feature_bank.new_bank_v2 import RestNoiseDetailV2
from emgimu.feature_bank.document_scale_v3 import DocumentScalePatternV3
from emgimu.feature_bank.spec_spatial_v3 import SpecTraceCovarianceV3, SpecSpdTangentV3
from emgimu.feature_bank.document_ces_v3 import DocumentCesFamilyV3
from emgimu.feature_bank.document_spectral_v3 import DocumentSpectralStateV3
from emgimu.feature_bank.document_temporal_v3 import DocumentTemporalFormV3
from emgimu.feature_bank.epn_study import aggregate_trials, _metrics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROTOCOL = HERE / 'EMG_WINDOW_BANK_V1_PROTOCOL.json'
RESULT = HERE / 'EMG_WINDOW_BANK_V1_RESULTS.json'
TABLE = HERE / 'EMG_WINDOW_BANK_V1_PREDICTIONS.csv'
GROUPS = ('F0', 'F1', 'F2ac', 'F3b', 'F4abc', 'F5window')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def compositions():
    result = {'F0': ('F0',)}
    for group in GROUPS[1:]: result['F0_plus_' + group] = ('F0', group)
    result['window_bank'] = GROUPS
    for group in GROUPS: result['window_bank_minus_' + group] = tuple(g for g in GROUPS if g != group)
    return result


def concatenate(features, groups):
    if not groups or len(set(groups)) != len(groups): raise ValueError('Unique nonempty groups required')
    arrays = [np.asarray(features[g]) for g in groups]
    if any(a.ndim != 2 or len(a) != len(arrays[0]) or not np.isfinite(a).all() for a in arrays):
        raise ValueError('Aligned finite trial features required')
    return np.concatenate(arrays, axis=1)


def factories():
    return {'F0': [RestNoiseDetailV2(rest_label=0)], 'F1': [DocumentScalePatternV3()],
            'F2ac': [SpecTraceCovarianceV3(.05), SpecSpdTangentV3(.05)],
            'F3b': [DocumentCesFamilyV3(25.)], 'F4abc': [DocumentSpectralStateV3(4, 4)],
            'F5window': [DocumentTemporalFormV3(25.)]}


def prepare():
    if PROTOCOL.exists(): raise FileExistsError('Existing frozen protocol')
    parent = json.loads((HERE / 'EMG_F0_F7_BANK_V1_PROTOCOL.json').read_text(encoding='utf8'))
    sources = ['benchmarks/new_bank_v3/emg_window_bank_v1.py', 'src/emgimu/datasets/epn612.py',
               'src/emgimu/feature_bank/core.py', 'src/emgimu/feature_bank/families.py',
               'src/emgimu/feature_bank/new_bank_v1.py', 'src/emgimu/feature_bank/new_bank_v2.py',
               'src/emgimu/feature_bank/document_scale_v3.py', 'src/emgimu/feature_bank/spec_spatial_v3.py',
               'src/emgimu/feature_bank/document_ces_v3.py', 'src/emgimu/feature_bank/document_spectral_v3.py',
               'src/emgimu/feature_bank/document_temporal_v3.py', 'src/emgimu/feature_bank/epn_study.py',
               'src/emgimu/feature_bank/screening.py']
    p = {'schema': 'emg_window_bank_v1', 'archive': parent['archive'], 'archive_sha256': parent['archive_sha256'],
         'source_users': list(range(1, 16)), 'target_users': list(range(52, 62)),
         'known_previous_target_users': list(range(16, 52)), 'source_model_seed': 20261008,
         'groups': list(GROUPS), 'compositions': compositions(),
         'source_model': 'Each composition independently fits source-only StandardScaler and balanced LogisticRegression C1 max_iter2000. Equal native-trial mass; arithmetic mean of four200ms windows per trial. Fixed family parameters; no target tuning, calibration, routing or gating.',
         'primary': 'Fixed six-group window_bank versus F0: lower pooled loss/Brier, nonworse macro-F1 and at least7/10 user loss wins. All additions/removals reported irrespective of sign; no composition selected using target results.',
         'scope': 'All six declared EMG window groups, not all document subfamilies or the document-wide F0-F9 bank. F2b CSP, ring-specific F3a/c, session F4d, full-bout/template F5b/c, anatomically calibrated F6, personal F7, F8 routing and F9 gating are absent. EPN52-61 public native cue-window trial classification; not hardware or own-user efficacy. No proof about every historical access event. Zero target calibration trials, full150 trials/user retained.',
         'source_sha256': {s: sha(ROOT / s) for s in sources}, 'default_promoted': False}
    PROTOCOL.write_text(json.dumps(p, indent=2) + '\n', encoding='utf8', newline='\n')
    print('Frozen six-group EMG window-bank protocol; no target signals loaded', flush=True)


def run():
    if RESULT.exists() or TABLE.exists(): raise FileExistsError('Refuse completed experiment overwrite')
    p = json.loads(PROTOCOL.read_text(encoding='utf8'))
    for path, digest in p['source_sha256'].items():
        if sha(ROOT / path) != digest: raise ValueError('Frozen implementation changed')
    if sha(Path(p['archive'])) != p['archive_sha256']: raise ValueError('Raw archive changed')
    print('1/3 Source trial features and independent composition fits', flush=True)
    source = load_epn612_windows(p['archive'], users=p['source_users'])
    fitted = factories(); features = {}; group_names = {}
    source_ids = y = weights = None
    for group, families in fitted.items():
        columns = []
        for family in families:
            family.fit(source.batch, source.labels)
            x, labels, users, ids, mass = aggregate_trials(family.transform(source.batch), source)
            if source_ids is not None and (not np.array_equal(ids, source_ids) or not np.array_equal(labels, y)):
                raise ValueError('Source family trial identities differ')
            source_ids, y, weights = ids, labels, mass
            columns.append(x)
        features[group] = np.concatenate(columns, axis=1)
        group_names[group] = [name for family in families for name in family.feature_names]
    models = {}; saved = {}
    for arm, groups in compositions().items():
        x = concatenate(features, groups)
        scaler = StandardScaler().fit(x, sample_weight=weights)
        model = LogisticRegression(C=1, class_weight='balanced', max_iter=2000, random_state=p['source_model_seed'])
        model.fit(scaler.transform(x), y, sample_weight=weights)
        if model.n_iter_.max() >= 2000 or not np.array_equal(model.classes_, range(6)):
            raise ValueError('Source convergence/class contract failed')
        models[arm] = (scaler, model)
        saved[arm] = {'groups': list(groups), 'dimension': x.shape[1], 'feature_dtype': str(x.dtype),
                      'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
                      'coef': model.coef_.tolist(), 'intercept': model.intercept_.tolist(),
                      'classes': model.classes_.tolist(), 'iterations': model.n_iter_.tolist()}
        print(f'Source fit {arm}: {x.shape[1]} coordinates', flush=True)
    state = pickle.dumps((fitted, models)); rows = []; blocks = []
    print('2/3 Untuned target composition readouts', flush=True)
    for user in p['target_users']:
        target = load_epn612_windows(p['archive'], users=[user]); tf = {}; ids = labels = None
        for group, families in fitted.items():
            columns = []
            for family in families:
                x, ly, lu, ti, mass = aggregate_trials(family.transform(target.batch), target)
                if ids is not None and (not np.array_equal(ids, ti) or not np.array_equal(labels, ly)):
                    raise ValueError('Target group trial identities differ')
                ids, labels = ti, ly; columns.append(x)
            tf[group] = np.concatenate(columns, axis=1)
        if len(ids) != 150 or len(set(ids)) != 150 or set(ids) & set(source_ids):
            raise ValueError('Independent held-out trial coverage failed')
        probabilities = {}
        for arm, groups in compositions().items():
            scaler, model = models[arm]
            q = model.predict_proba(scaler.transform(concatenate(tf, groups)))
            probabilities[arm] = q.tolist()
            for trial, label, probability in zip(ids, labels, q):
                rows.append({'user': user, 'arm': arm, 'trial_id': trial, 'label': int(label),
                             **{f'p_{c}': float(probability[c]) for c in range(6)}})
        blocks.append({'user': user, 'evaluation_ids': ids.tolist(), 'labels': labels.tolist(),
                       'features': {g: x.tolist() for g, x in tf.items()}, 'probabilities': probabilities})
        print(f'User{user}: all13 fixed compositions complete', flush=True)
    if state != pickle.dumps((fitted, models)): raise ValueError('Target changed source state')
    with TABLE.open('w', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    scores = {}
    for arm in compositions():
        per_user = {str(b['user']): _metrics(np.array(b['labels']), np.array(b['probabilities'][arm]), np.ones(len(b['labels']))) for b in blocks}
        labels = np.concatenate([b['labels'] for b in blocks]); q = np.concatenate([b['probabilities'][arm] for b in blocks])
        scores[arm] = {'pooled': _metrics(labels, q, np.ones(len(labels))), 'per_user': per_user,
                       'minimum_user_macro_f1': min(s['macro_f1'] for s in per_user.values())}
    base = scores['F0']['pooled']; full = scores['window_bank']['pooled']
    wins = sum(scores['window_bank']['per_user'][u]['log_loss'] < s['log_loss'] - 1e-12 for u, s in scores['F0']['per_user'].items())
    criteria = {'lower_log_loss': full['log_loss'] < base['log_loss'], 'lower_brier': full['brier'] < base['brier'],
                'nonworse_macro_f1': full['macro_f1'] >= base['macro_f1'], 'at_least_seven_user_loss_wins': wins >= 7}
    result = {'schema': 'emg_window_bank_v1', 'protocol_sha256': sha(PROTOCOL), 'prediction_sha256': sha(TABLE),
              'source_trial_ids': source_ids.tolist(), 'source_trial_count': len(y),
              'group_feature_names': group_names, 'source_models': saved, 'source_state_immutable': True,
              'blocks': blocks, 'scores': scores, 'primary': {'criteria': criteria, 'passed': all(criteria.values()),
              'user_logloss_wins': wins, 'delta_logloss': base['log_loss'] - full['log_loss'],
              'delta_macro_f1': full['macro_f1'] - base['macro_f1'], 'delta_brier': base['brier'] - full['brier']},
              'actual_target_calibration_trials': 0, 'scope': p['scope'], 'default_promoted': False,
              'physical_validation_proven': False, 'completion_proven': False}
    RESULT.write_text(json.dumps(result, indent=2) + '\n', encoding='utf8', newline='\n')
    print('3/3 Saved native six-group bank, additions and removals', flush=True)
    print(json.dumps(result['primary']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true'); args = parser.parse_args()
    prepare() if args.prepare else run()
