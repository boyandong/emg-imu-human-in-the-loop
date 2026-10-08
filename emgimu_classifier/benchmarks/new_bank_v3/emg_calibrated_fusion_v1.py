"""Predeclared six-provider OOF-calibrated late fusion and provider removals."""
import argparse
import csv
import json
import pickle
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from emgimu.datasets.epn612 import load_epn612_windows
from emgimu.feature_bank.epn_study import aggregate_trials, _metrics
from emgimu.feature_bank.force_nested_oof import fit_temperature, temperature_probability
from emgimu.feature_bank.available_bank_fusion_v1 import AvailableBankFusionPolicyV1
from benchmarks.new_bank_v3.emg_window_bank_v1 import factories, GROUPS, sha
from benchmarks.new_bank_v3.emg_f0_f7_bank_v1 import select_trials

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROTOCOL = HERE/'EMG_CALIBRATED_FUSION_V1_PROTOCOL.json'
RESULT = HERE/'EMG_CALIBRATED_FUSION_V1_RESULTS.json'
TABLE = HERE/'EMG_CALIBRATED_FUSION_V1_PREDICTIONS.csv'


def features(data, fitted):
    vectors = {}; canonical = None
    for group, families in fitted.items():
        columns = []
        for family in families:
            x, y, users, ids, mass = aggregate_trials(family.transform(data.batch), data)
            if canonical is not None:
                if any(not np.array_equal(a, b) for a, b in zip(canonical, (y, users, ids, mass))):
                    raise ValueError('Provider trial identity mismatch')
            else: canonical = (y, users, ids, mass)
            columns.append(x)
        vectors[group] = np.concatenate(columns, axis=1)
    return vectors, canonical


def fit_source(data, seed):
    fitted = factories()
    for families in fitted.values():
        for family in families: family.fit(data.batch, data.labels)
    x, (y, users, ids, mass) = features(data, fitted)
    models = {}; saved = {}
    for group in GROUPS:
        scaler = StandardScaler().fit(x[group], sample_weight=mass)
        model = LogisticRegression(C=1, class_weight='balanced', max_iter=2000, random_state=seed)
        model.fit(scaler.transform(x[group]), y, sample_weight=mass)
        if model.n_iter_.max() >= 2000 or not np.array_equal(model.classes_, range(6)):
            raise ValueError('Source convergence/class contract failed')
        models[group] = (scaler, model)
        saved[group] = {'mean':scaler.mean_.tolist(), 'scale':scaler.scale_.tolist(),
            'coef':model.coef_.tolist(), 'intercept':model.intercept_.tolist(), 'classes':model.classes_.tolist(),
            'dimension':x[group].shape[1], 'dtype':str(x[group].dtype), 'iterations':model.n_iter_.tolist()}
    return fitted, models, saved, ids


def prepare():
    if PROTOCOL.exists(): raise FileExistsError('Existing frozen protocol')
    parent = json.loads((HERE/'EMG_WINDOW_BANK_V1_PROTOCOL.json').read_text(encoding='utf8'))
    paths = list(parent['source_sha256']) + ['benchmarks/new_bank_v3/emg_calibrated_fusion_v1.py',
        'benchmarks/new_bank_v3/emg_f0_f7_bank_v1.py', 'src/emgimu/feature_bank/force_nested_oof.py',
        'src/emgimu/feature_bank/available_bank_fusion_v1.py', 'src/emgimu/feature_bank/document_reliability_v2.py']
    p = {'schema':'emg_calibrated_fusion_v1', 'archive':parent['archive'], 'archive_sha256':parent['archive_sha256'],
         'source_users':list(range(1,16)), 'target_users':list(range(62,72)),
         'known_prior_target_users':list(range(16,62)), 'source_folds':[list(range(1,6)),list(range(6,11)),list(range(11,16))],
         'source_model_seed':20261008, 'selection_seed':'20261008_six_provider_late_fusion',
         'budgets':[0,1,2,5], 'reserved_per_class':5, 'providers':list(GROUPS),
         'source_model':'Six independent fixed LogisticRegression C1/max2000 classifiers on trial means. Each source-user OOF fold refits every representation including Rest thresholds and tangent reference, then scaler/classifier using only the other10 source users. Source-OOF raw probabilities fit six bounded .25..4 temperatures, with temperature1 always a candidate. These temperature-fitting losses are not held-out performance.',
         'weight_policy':'Uniform population over six providers, predeclared n0=12 and log-reliability temperature1. Calibration-only D/E on source-standardized trial vectors; alpha=12/(12+6*shots). No target-selected policy or probability recalibration. Provider classifiers remain source-only; no personal prototypes/normalization/anchor updates.',
         'target_protocol':'Each class reserves5 native trials by SHA256(seed|trial_id), nested1/2/5 prefixes. All budgets evaluate identical120 trials/user. Source-only provider and uniform controls use0 calibration trials. Reliability and its removals use6*shots; zero shots equals uniform.',
         'primary':'Five-shot reliability versus source-calibrated F0 and uniform: lower pooled loss/Brier and nonworse macro-F1 than F0, lower loss than uniform, at least7/10 user loss wins versus F0. Report all guards; no target-dependent provider shortlist, alpha, temperature or default promotion.',
         'scope':'New versioned EPN62-71 public six-provider EMG window fusion, 1200 held-out trials. Provider omission renormalizes frozen calibrated weights, without refitting or recomputing reliability. Not all document F0-F9/subfamilies, strict F6, F7 personal anchor, F8 session signature, F9 physical fault gating, online segmentation or own-device efficacy. No claim about every historical access event.',
         'source_sha256':{name:sha(ROOT/name) for name in paths}, 'default_promoted':False}
    PROTOCOL.write_text(json.dumps(p,indent=2)+'\n',encoding='utf8',newline='\n')
    print('Frozen OOF-calibrated six-provider fusion protocol; target signals not loaded',flush=True)


def run():
    if RESULT.exists() or TABLE.exists(): raise FileExistsError('Refuse completed experiment overwrite')
    p = json.loads(PROTOCOL.read_text(encoding='utf8'))
    for name,digest in p['source_sha256'].items():
        if sha(ROOT/name) != digest: raise ValueError('Frozen implementation changed')
    if sha(Path(p['archive'])) != p['archive_sha256']: raise ValueError('Raw archive changed')
    source = load_epn612_windows(p['archive'],users=p['source_users'])
    _, source_labels, source_users, source_ids, _ = aggregate_trials(np.zeros((source.batch.windows,1)),source)
    indices = {trial:i for i,trial in enumerate(source_ids)}
    oof = {name:np.zeros((len(source_ids),6)) for name in GROUPS}; fold_ids=np.full(len(source_ids),-1); folds=[]
    print('1/3 Source-user OOF representations, classifiers and temperatures',flush=True)
    for fold,held_users in enumerate(p['source_folds']):
        train = source.take(np.flatnonzero(~np.isin(source.users,held_users)))
        held = source.take(np.flatnonzero(np.isin(source.users,held_users)))
        fitted, models, _, train_ids = fit_source(train,p['source_model_seed'])
        state=pickle.dumps((fitted,models)); hx,(hy,hu,hi,_) = features(held,fitted)
        if set(train_ids)&set(hi) or set(train.users)&set(held.users): raise ValueError('Source fold leakage')
        positions=np.array([indices[t] for t in hi]); fold_ids[positions]=fold
        for name,(scaler,model) in models.items(): oof[name][positions]=model.predict_proba(scaler.transform(hx[name]))
        if pickle.dumps((fitted,models))!=state: raise ValueError('OOF inference changed source state')
        folds.append({'fold':fold,'train_users':sorted(set(train.users.tolist())),'held_users':held_users,
                      'train_ids':train_ids.tolist(),'held_ids':hi.tolist(),'source_state_immutable':True})
        print(f'Source fold{fold+1}/3 complete',flush=True)
    if np.any(fold_ids < 0): raise ValueError('Missing source OOF rows')
    temperatures={name:fit_temperature(q,source_labels) for name,q in oof.items()}
    fitted,models,source_fit,full_ids=fit_source(source,p['source_model_seed'])
    if not np.array_equal(full_ids,source_ids): raise ValueError('Final source identities differ')
    source_state=pickle.dumps((fitted,models)); rows=[]; blocks=[]
    policy=AvailableBankFusionPolicyV1(tuple(range(6)),GROUPS,(1/6,)*6,12.,1.,sha(PROTOCOL),tuple(source_ids))
    print('2/3 New target users, nested calibration and frozen-weight removals',flush=True)
    for user in p['target_users']:
        data=load_epn612_windows(p['archive'],users=[user]); x,(y,users,ids,_)=features(data,fitted)
        chosen,evaluation=select_trials(ids,y,p['selection_seed'],p['reserved_per_class'])
        if len(evaluation)!=120 or set(ids)&set(source_ids): raise ValueError('Target independent trial coverage failed')
        standardized={name:models[name][0].transform(values) for name,values in x.items()}
        raw={name:models[name][1].predict_proba(standardized[name]) for name in GROUPS}
        probabilities={name:temperature_probability(q,temperatures[name]) for name,q in raw.items()}
        eval_ids=tuple(ids[evaluation]); eval_y=y[evaluation]
        values={name:q[evaluation] for name,q in probabilities.items()}
        axes={name:tuple(range(6)) for name in GROUPS}; trial_axes={name:eval_ids for name in GROUPS}
        block={'user':user,'evaluation_ids':list(eval_ids),'labels':eval_y.tolist(),
               'evaluation_features':{name:x[name][evaluation].tolist() for name in GROUPS},
               'raw_provider_probabilities':{name:q[evaluation].tolist() for name,q in raw.items()},
               'provider_probabilities':{name:q.tolist() for name,q in values.items()},
               'reserved_ids':{str(c):ids[positions].tolist() for c,positions in chosen.items()},'calibrations':[]}
        for shots in p['budgets']:
            cal=np.array([i for c in range(6) for i in chosen[c][:shots]],dtype=int)
            calibration={name:(z[cal],y[cal],ids[cal]) for name,z in standardized.items()} if shots else {}
            state=policy.calibrate(calibration,forbidden_evaluation_trials=eval_ids)
            arms={name:q for name,q in values.items()}
            uniform=policy.predict(policy.calibrate({}),values,evaluation_trials=eval_ids,provider_trial_ids=trial_axes,provider_classes=axes)
            reliability=policy.predict(state,values,evaluation_trials=eval_ids,provider_trial_ids=trial_axes,provider_classes=axes)
            arms['uniform_bank']=uniform['probabilities']; arms['reliability_bank']=reliability['probabilities']
            for missing in GROUPS:
                subset={name:q for name,q in values.items() if name!=missing}
                result=policy.predict(state,subset,evaluation_trials=eval_ids,
                    provider_trial_ids={name:eval_ids for name in subset},provider_classes={name:tuple(range(6)) for name in subset})
                arms['reliability_minus_'+missing]=result['probabilities']
            block['calibrations'].append({'shots':shots,'ids':ids[cal].tolist(),'labels':y[cal].tolist(),
                'source_standardized_features':{name:z[cal].tolist() for name,z in standardized.items()},
                'weights':list(state.weights),'alpha':state.alpha,'n_cal_trials':state.n_cal_trials,
                'probabilities':{name:q.tolist() for name,q in arms.items()}})
            for arm,q in arms.items():
                for trial,label,probability in zip(eval_ids,eval_y,q):
                    rows.append({'user':user,'shots_per_class':shots,'arm':arm,'trial_id':trial,'label':int(label),
                                 **{f'p_{c}':float(probability[c]) for c in range(6)}})
        blocks.append(block);print(f'User{user}: nested budgets and6 removals complete',flush=True)
    if source_state!=pickle.dumps((fitted,models)): raise ValueError('Target changed source state')
    with TABLE.open('w',encoding='utf8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    scores={}
    for shots in p['budgets']:
        scores[str(shots)]={}
        for arm in blocks[0]['calibrations'][0]['probabilities']:
            per_user={};truth=[];probabilities=[]
            for block in blocks:
                entry=next(c for c in block['calibrations'] if c['shots']==shots)
                q=np.array(entry['probabilities'][arm]);y=np.array(block['labels'])
                per_user[str(block['user'])]=_metrics(y,q,np.ones(len(y)));truth.extend(y);probabilities.extend(q)
            scores[str(shots)][arm]={'pooled':_metrics(np.array(truth),np.array(probabilities),np.ones(len(truth))),
                                    'per_user':per_user,'minimum_user_macro_f1':min(s['macro_f1'] for s in per_user.values())}
    b=scores['5']['F0']; f=scores['5']['reliability_bank']; u=scores['5']['uniform_bank']
    wins=sum(f['per_user'][user]['log_loss']<v['log_loss']-1e-12 for user,v in b['per_user'].items())
    guards={'lower_log_loss':f['pooled']['log_loss']<b['pooled']['log_loss'],
            'lower_brier':f['pooled']['brier']<b['pooled']['brier'],
            'nonworse_macro_f1':f['pooled']['macro_f1']>=b['pooled']['macro_f1'],
            'lower_loss_than_uniform':f['pooled']['log_loss']<u['pooled']['log_loss'],
            'at_least_seven_user_loss_wins':wins>=7}
    payload={'schema':'emg_calibrated_fusion_v1','protocol_sha256':sha(PROTOCOL),'prediction_sha256':sha(TABLE),
        'source_trial_ids':source_ids.tolist(),'source_labels':source_labels.tolist(),'source_users':source_users.tolist(),
        'source_oof':{'fold_ids':fold_ids.tolist(),'folds':folds,'raw_probabilities':{name:q.tolist() for name,q in oof.items()}},
        'temperatures':temperatures,'source_models':source_fit,'source_state_immutable':True,
        'blocks':blocks,'scores':scores,'primary_five_shot':{'criteria':guards,'passed':all(guards.values()),
            'user_logloss_wins':wins,'delta_logloss':b['pooled']['log_loss']-f['pooled']['log_loss'],
            'delta_macro_f1':f['pooled']['macro_f1']-b['pooled']['macro_f1'],
            'delta_logloss_over_uniform':u['pooled']['log_loss']-f['pooled']['log_loss']},
        'scope':p['scope'],'default_promoted':False,'physical_validation_proven':False,'completion_proven':False}
    RESULT.write_text(json.dumps(payload,indent=2)+'\n',encoding='utf8',newline='\n')
    print('3/3 Saved OOF-calibrated late-fusion experiment',flush=True);print(json.dumps(payload['primary_five_shot']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    prepare() if args.prepare else run()
