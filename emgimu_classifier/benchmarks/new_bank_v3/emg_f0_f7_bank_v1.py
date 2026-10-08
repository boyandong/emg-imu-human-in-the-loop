"""Predeclared EMG-only two-provider bank and paired provider ablations."""
import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from emgimu.datasets.epn612 import load_epn612_windows
from emgimu.feature_bank.new_bank_v2 import RestNoiseDetailV2
from emgimu.feature_bank.affine_spd_anchor import AffineSpdPrototypeAnchor, document_spd_matrices, affine_spd_distance
from emgimu.feature_bank.epn_study import aggregate_trials, _metrics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROTOCOL = HERE/'EMG_F0_F7_BANK_V1_PROTOCOL.json'
RESULT = HERE/'EMG_F0_F7_BANK_V1_RESULTS.json'
TABLE = HERE/'EMG_F0_F7_BANK_V1_PREDICTIONS.csv'
ARMS = ('F0', 'F7', 'F0_F7', 'F0_uniform')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def select_trials(ids, labels, seed, reserved=5):
    ids, labels = np.asarray(ids), np.asarray(labels)
    if ids.ndim != 1 or labels.shape != ids.shape or len(set(ids.tolist())) != len(ids):
        raise ValueError('Unique aligned independent trial IDs required')
    if not np.array_equal(np.unique(labels), np.arange(6)): raise ValueError('All six native classes required')
    selected = {}; evaluation = []
    for label in range(6):
        candidates = np.flatnonzero(labels == label)
        ordered = sorted(candidates, key=lambda i: hashlib.sha256(f'{seed}|{ids[i]}'.encode()).digest())
        if len(ordered) <= reserved: raise ValueError('Independent calibration and evaluation trials required')
        selected[label] = ordered[:reserved]; evaluation.extend(ordered[reserved:])
    return selected, np.array(sorted(evaluation), dtype=int)


def paired_arms(base, anchor):
    a, b = np.asarray(base), np.asarray(anchor)
    if a.ndim != 2 or a.shape != b.shape or a.shape[1] != 6 or not len(a): raise ValueError('Matching six-class probabilities required')
    for p in (a, b):
        if not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1), 1, atol=1e-9):
            raise ValueError('Invalid probabilities')
    return {'F0':a.copy(), 'F7':b.copy(), 'F0_F7':(a+b)/2, 'F0_uniform':(a+1/6)/2}


def prepare():
    if PROTOCOL.exists(): raise FileExistsError('Protocol already frozen')
    prior = json.loads((HERE/'MAHALANOBIS_EPN_HOLDOUT_V2_PROTOCOL.json').read_text(encoding='utf8'))
    paths = ['benchmarks/new_bank_v3/emg_f0_f7_bank_v1.py', 'src/emgimu/datasets/epn612.py',
             'src/emgimu/feature_bank/new_bank_v2.py','src/emgimu/feature_bank/new_bank_v1.py',
             'src/emgimu/feature_bank/core.py','src/emgimu/feature_bank/families.py',
             'src/emgimu/feature_bank/affine_spd_anchor.py','src/emgimu/feature_bank/spec_spatial_v3.py',
             'src/emgimu/feature_bank/epn_study.py']
    p = {'schema':'emg_f0_f7_bank_v1','archive':prior['archive'],'archive_sha256':prior['archive_sha256'],
         'source_users':list(range(1,16)), 'target_users':list(range(42,52)),
         'known_prior_target_users':list(range(16,42)), 'budgets':[0,1,2,5], 'reserved_per_class':5,
         'selection_seed':'20261008_emg_only_bank', 'source_model_seed':20261008,
         'source_model':'Source-only RestNoiseDetailV2 six metrics/8 channels; 48-dimensional equal-window trial mean. StandardScaler then balanced LogisticRegression C1,max_iter2000; no target temperature or representation selection.',
         'target_selection':'SHA256(seed|trial_id) per class; reserve first5 for nested1/2/5 calibration; identical remaining trial set even at0 shots. Each shot is a distinct native trial.',
         'F7':'Document centered trace-normalized SPD with fixed .05 shrinkage and SPD ridge; equal-trial arithmetic prototypes, exact affine distance. Softmax temperature is median pairwise calibration-prototype distance, floored1e-10.',
         'bank':'Fixed0.5 F0 +0.5 F7 probabilities; minusF7=F0, minusF0=F7. F0_uniform controls uniform softening; not feature concatenation or mutual-information estimation.',
         'primary':'Five-shot pooled comparison: lower loss/Brier, nonworse macroF1 thanF0, lower loss than uniform and at least7/10 user loss wins. Report each criterion; not a significance test or automatic default promotion.',
         'scope':'New precommitted EMG-only two-provider bank on EPN42-51, disjoint from known earlier1-41 cohorts. No claim about every historical file access or all612 users. Offline native cue-window trial classification; not continuous, whole-document full bank, strict F6 body-frame, own-device or low-burden efficacy. IMU arrays required by the legacy loader are not feature inputs.',
         'source_sha256':{path:sha(ROOT/path) for path in paths}, 'default_promoted':False}
    PROTOCOL.write_text(json.dumps(p,indent=2)+'\n',encoding='utf8',newline='\n')
    print('Frozen EMG-only bank protocol; target signals not loaded',flush=True)


def run():
    if RESULT.exists() or TABLE.exists(): raise FileExistsError('Refuse completed experiment overwrite')
    p = json.loads(PROTOCOL.read_text(encoding='utf8'))
    for path, digest in p['source_sha256'].items():
        if sha(ROOT/path) != digest: raise ValueError('Frozen implementation changed')
    if sha(Path(p['archive'])) != p['archive_sha256']: raise ValueError('Archive changed')
    print('1/3 Fit source-only EMG model',flush=True)
    source = load_epn612_windows(p['archive'],users=p['source_users'])
    family = RestNoiseDetailV2(rest_label=0).fit(source.batch,source.labels)
    sx, sy, su, source_ids, weights = aggregate_trials(family.transform(source.batch),source)
    scaler = StandardScaler().fit(sx,sample_weight=weights)
    model = LogisticRegression(C=1,class_weight='balanced',max_iter=2000,random_state=p['source_model_seed'])
    model.fit(scaler.transform(sx),sy,sample_weight=weights)
    if model.n_iter_.max() >= 2000 or not np.array_equal(model.classes_,range(6)): raise ValueError('Source convergence/class contract failed')
    source_state = pickle.dumps((family,scaler,model)); blocks = []; rows = []
    source_fit = {'thresholds':family.thresholds_.tolist(),'rest_windows':family.rest_windows_,
                  'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),
                  'coef':model.coef_.tolist(),'intercept':model.intercept_.tolist(),'classes':model.classes_.tolist(),
                  'feature_dimension':sx.shape[1],'source_trials':len(sy),
                  'class_trial_counts':{str(c):int(sum(sy==c)) for c in range(6)}}
    print('2/3 New target cohort: fixed source and nested calibration',flush=True)
    for user in p['target_users']:
        target = load_epn612_windows(p['archive'],users=[user])
        x,y,users,ids,_ = aggregate_trials(family.transform(target.batch),target)
        if set(source_ids)&set(ids) or set(users)!={user}: raise ValueError('Source/target identity contract failed')
        selected,evaluation = select_trials(ids,y,p['selection_seed'],p['reserved_per_class'])
        matrices = document_spd_matrices(target.batch.emg)
        all_matrices = np.stack([matrices[target.trials==t].mean(0) for t in ids])
        base = model.predict_proba(scaler.transform(x[evaluation]))
        block = {'user':user,'evaluation_ids':ids[evaluation].tolist(),'labels':y[evaluation].tolist(),
                 'reserved_ids':{str(c):ids[index].tolist() for c,index in selected.items()},
                 'evaluation_features':x[evaluation].tolist(),'evaluation_matrices':all_matrices[evaluation].tolist(),
                 'F0_probabilities':base.tolist(),'calibrations':[]}
        for budget in p['budgets']:
            if budget == 0:
                arms = {'F0':base}; calibration_ids = []
            else:
                calibration = np.array([i for c in range(6) for i in selected[c][:budget]])
                anchor = AffineSpdPrototypeAnchor().fit(all_matrices[calibration],y[calibration],ids[calibration])
                before = pickle.dumps(anchor)
                returned, distances = anchor.transform(all_matrices[evaluation],ids[evaluation])
                if not np.array_equal(returned,ids[evaluation]): raise ValueError('Anchor trial order differs')
                between = [affine_spd_distance(anchor.prototypes_[i],anchor.prototypes_[j]) for i in range(6) for j in range(i+1,6)]
                temperature = max(float(np.median(between)),1e-10)
                probability = softmax(-distances/temperature,axis=1)
                if before != pickle.dumps(anchor): raise ValueError('Evaluation changed anchor')
                arms = paired_arms(base,probability); calibration_ids = ids[calibration].tolist()
                block['calibrations'].append({'shots':budget,'ids':calibration_ids,'labels':y[calibration].tolist(),
                    'calibration_matrices':all_matrices[calibration].tolist(),'prototypes':anchor.prototypes_.tolist(),
                    'temperature':temperature,'F7_probabilities':probability.tolist()})
            for arm, probability in arms.items():
                for trial,label,q in zip(ids[evaluation],y[evaluation],probability):
                    rows.append({'user':user,'shots_per_class':budget,'arm':arm,'trial_id':trial,'label':int(label),
                                 **{f'p_{c}':float(q[c]) for c in range(6)}})
        blocks.append(block)
        print(f'User{user}: all budgets/paired ablations complete',flush=True)
    if source_state != pickle.dumps((family,scaler,model)): raise ValueError('Target data changed source state')
    scores = {}
    for budget in p['budgets']:
        scores[str(budget)] = {}
        for arm in (('F0',) if budget==0 else ARMS):
            group = [r for r in rows if r['shots_per_class']==budget and r['arm']==arm]
            y = np.array([r['label'] for r in group]); q = np.array([[r[f'p_{c}'] for c in range(6)] for r in group])
            per_user = {}
            for user in p['target_users']:
                mask = np.array([r['user']==user for r in group])
                per_user[str(user)] = _metrics(y[mask],q[mask],np.ones(mask.sum()))
            scores[str(budget)][arm] = {'pooled':_metrics(y,q,np.ones(len(y))),'per_user':per_user,
                'minimum_user_macro_f1':min(v['macro_f1'] for v in per_user.values())}
    primary = scores['5']; core,full,uniform = [primary[a]['pooled'] for a in ('F0','F0_F7','F0_uniform')]
    wins = sum(primary['F0_F7']['per_user'][str(u)]['log_loss'] < primary['F0']['per_user'][str(u)]['log_loss'] for u in p['target_users'])
    criteria = {'lower_log_loss':full['log_loss']<core['log_loss'], 'nonworse_macro_f1':full['macro_f1']>=core['macro_f1'],
                'lower_brier':full['brier']<core['brier'], 'lower_loss_than_uniform':full['log_loss']<uniform['log_loss'],
                'at_least7_user_loss_wins':wins>=7}
    with TABLE.open('w',encoding='utf8',newline='') as out:
        writer = csv.DictWriter(out,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    result = {'schema':p['schema'],'protocol_sha256':sha(PROTOCOL),'prediction_sha256':sha(TABLE),
              'source_trial_ids':source_ids.tolist(),'source_fit':source_fit,'source_state_immutable':True,
              'blocks':blocks,'scores':scores,'primary_five_shot':{'criteria':criteria,'conjunction_passed':all(criteria.values()),
                    'user_logloss_wins':wins,'delta_logloss':core['log_loss']-full['log_loss'],
                    'delta_macro_f1':full['macro_f1']-core['macro_f1'],'delta_brier':core['brier']-full['brier'],
                    'delta_logloss_over_uniform':uniform['log_loss']-full['log_loss']},
              'default_promoted':False,'physical_validation_proven':False,'completion_proven':False,'scope':p['scope']}
    RESULT.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8',newline='\n')
    print('3/3 Saved fixed EMG-only bank and provider removals',flush=True)
    print(json.dumps(result['primary_five_shot']),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    prepare() if args.prepare else run()
