import csv
import hashlib
import json
from itertools import combinations
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'benchmarks/new_bank_v3'


def load(): return json.loads((HERE/'EMG_CALIBRATED_FUSION_V1_RESULTS.json').read_text(encoding='utf8'))


def temper(p,t):
    z=np.log(np.clip(p,1e-15,None))/t;z-=z.max(1,keepdims=True)
    q=np.exp(z);return q/q.sum(1,keepdims=True)


def metrics(y,q):
    prediction=q.argmax(1);f1=[]
    for c in range(6):
        denominator=sum(y==c)+sum(prediction==c)
        f1.append(2*sum((y==c)&(prediction==c))/denominator if denominator else 0)
    return {'macro_f1':sum(f1)/6,'accuracy':np.mean(y==prediction),
        'log_loss':-np.mean(np.log(np.clip(q[np.arange(len(y)),y],np.finfo(float).eps,1))),
        'brier':np.mean((q-np.eye(6)[y])**2)}


def test_source_fold_isolation_unique_oof_coverage_and_temperature_fit_only_source():
    r=load();path=HERE/'EMG_CALIBRATED_FUSION_V1_PROTOCOL.json'
    p=json.loads(path.read_text(encoding='utf8'))
    assert hashlib.sha256(path.read_bytes()).hexdigest()==r['protocol_sha256']
    for name,digest in p['source_sha256'].items(): assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    assert p['source_users']==list(range(1,16)) and p['target_users']==list(range(62,72))
    assert not set(p['target_users']) & set(p['source_users']+p['known_prior_target_users'])
    ids=r['source_trial_ids'];index={t:i for i,t in enumerate(ids)};users=np.array(r['source_users']);y=np.array(r['source_labels'])
    assert len(ids)==len(index)==2250
    held=[]
    for fold in r['source_oof']['folds']:
        assert not set(fold['train_ids']) & set(fold['held_ids'])
        assert set(fold['train_ids']) | set(fold['held_ids'])==set(ids)
        assert set(fold['train_users'])==set(p['source_users'])-set(fold['held_users'])
        assert set(users[[index[t] for t in fold['held_ids']]])==set(fold['held_users'])
        assert set(users[[index[t] for t in fold['train_ids']]])==set(fold['train_users'])
        assert fold['source_state_immutable'] and len(fold['train_ids'])==1500 and len(fold['held_ids'])==750
        assert all(r['source_oof']['fold_ids'][index[t]]==fold['fold'] for t in fold['held_ids'])
        held+=fold['held_ids']
    assert len(held)==len(set(held))==len(ids)
    for name,values in r['source_oof']['raw_probabilities'].items():
        q=np.array(values);assert q.shape==(2250,6) and np.isfinite(q).all()
        np.testing.assert_allclose(q.sum(1),1.,atol=1e-12)
        def loss(log_t): return -np.log(np.clip(temper(q,np.exp(log_t))[np.arange(len(y)),y],1e-15,None)).mean()
        optimum=minimize_scalar(loss,bounds=(np.log(.25),np.log(4.)),method='bounded')
        expected=np.exp(min((0.,optimum.x),key=loss))
        assert abs(r['temperatures'][name]-expected)<1e-12
        assert loss(np.log(r['temperatures'][name]))<=loss(0.)+1e-12
    assert r['source_state_immutable'] and not r['default_promoted'] and not r['physical_validation_proven'] and not r['completion_proven']


def test_independent_model_probability_replay_calibration_arithmetic_and_removals():
    r=load();provider_names=list(r['source_models']);all_ids=[]
    for block in r['blocks']:
        ids=block['evaluation_ids'];all_ids+=ids
        assert len(ids)==len(set(ids))==120 and not set(ids)&set(r['source_trial_ids'])
        np.testing.assert_array_equal(np.bincount(block['labels']),[20]*6)
        reserved={t for values in block['reserved_ids'].values() for t in values}
        assert len(reserved)==30 and not set(ids)&reserved
        previous=set()
        for name,fit in r['source_models'].items():
            x=np.array(block['evaluation_features'][name],dtype=fit['dtype'])
            assert x.shape==(120,fit['dimension'])
            x-=np.array(fit['mean'],dtype=x.dtype);x/=np.array(fit['scale'],dtype=x.dtype)
            z=x@np.array(fit['coef']).T+np.array(fit['intercept']);raw=np.exp(z-z.max(1,keepdims=True));raw/=raw.sum(1,keepdims=True)
            np.testing.assert_allclose(raw,block['raw_provider_probabilities'][name],atol=1e-12,rtol=1e-12)
            np.testing.assert_allclose(temper(raw,r['temperatures'][name]),block['provider_probabilities'][name],atol=1e-12,rtol=1e-12)
        for c in block['calibrations']:
            shots=c['shots'];assert c['n_cal_trials']==len(c['ids'])==6*shots
            assert previous<=set(c['ids'])<=reserved and not set(ids)&set(c['ids']);previous=set(c['ids'])
            for label in range(6):
                assert [t for t,y in zip(c['ids'],c['labels']) if y==label]==block['reserved_ids'][str(label)][:shots]
            assert c['alpha']==12/(12+6*shots)
            if shots:
                y=np.array(c['labels']);between=[];within=[]
                for name in provider_names:
                    x=np.array(c['source_standardized_features'][name]);means=[x[y==label].mean(0) for label in range(6)]
                    between.append(np.mean([np.linalg.norm(means[a]-means[b]) for a,b in combinations(range(6),2)]))
                    within.append(np.mean([np.mean(np.linalg.norm(x[y==label]-means[label],axis=1)) for label in range(6)]))
                reliability=np.array(between)/(np.array(within)+1e-10)
                z=np.log(reliability+1e-10);personal=np.exp(z-max(z));personal/=sum(personal)
                expected=c['alpha']*np.full(6,1/6)+(1-c['alpha'])*personal
            else: expected=np.full(6,1/6)
            np.testing.assert_allclose(c['weights'],expected,rtol=0,atol=1e-12)
            arrays=np.array([block['provider_probabilities'][name] for name in provider_names])
            np.testing.assert_allclose(c['probabilities']['uniform_bank'],arrays.mean(0),rtol=0,atol=1e-12)
            np.testing.assert_allclose(c['probabilities']['reliability_bank'],np.einsum('k,ktc->tc',expected,arrays),rtol=0,atol=1e-12)
            for index,name in enumerate(provider_names):
                np.testing.assert_array_equal(c['probabilities'][name],block['provider_probabilities'][name])
                keep=[i for i in range(6) if i!=index]
                wanted=np.einsum('k,ktc->tc',expected[keep]/sum(expected[keep]),arrays[keep])
                np.testing.assert_allclose(c['probabilities']['reliability_minus_'+name],wanted,rtol=0,atol=1e-12)
    assert len(all_ids)==len(set(all_ids))==1200


def test_independent_scores_primary_and_complete_fixed_trial_predictions():
    r=load();path=HERE/'EMG_CALIBRATED_FUSION_V1_PREDICTIONS.csv'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==r['prediction_sha256']
    with path.open(encoding='utf8',newline='') as stream:rows=list(csv.DictReader(stream))
    lookup={(int(row['user']),int(row['shots_per_class']),row['arm'],row['trial_id']):row for row in rows}
    assert len(rows)==len(lookup)==67200
    for shots,scores in r['scores'].items():
        for arm,s in scores.items():
            labels=[];probabilities=[]
            for block in r['blocks']:
                entry=next(c for c in block['calibrations'] if c['shots']==int(shots))
                y=np.array(block['labels']);q=np.array(entry['probabilities'][arm]);labels.extend(y);probabilities.extend(q)
                for key,value in metrics(y,q).items(): assert abs(value-s['per_user'][str(block['user'])][key])<1e-12
                table_rows=[lookup[block['user'],int(shots),arm,t] for t in block['evaluation_ids']]
                np.testing.assert_array_equal([int(row['label']) for row in table_rows],y)
                np.testing.assert_array_equal([[float(row[f'p_{c}']) for c in range(6)] for row in table_rows],q)
            for key,value in metrics(np.array(labels),np.array(probabilities)).items(): assert abs(value-s['pooled'][key])<1e-12
            assert s['minimum_user_macro_f1']==min(v['macro_f1'] for v in s['per_user'].values())
    b=r['scores']['5']['F0'];f=r['scores']['5']['reliability_bank'];u=r['scores']['5']['uniform_bank'];p=r['primary_five_shot']
    wins=sum(f['per_user'][user]['log_loss']<v['log_loss']-1e-12 for user,v in b['per_user'].items())
    guards={'lower_log_loss':f['pooled']['log_loss']<b['pooled']['log_loss'],'lower_brier':f['pooled']['brier']<b['pooled']['brier'],
            'nonworse_macro_f1':f['pooled']['macro_f1']>=b['pooled']['macro_f1'],
            'lower_loss_than_uniform':f['pooled']['log_loss']<u['pooled']['log_loss'],'at_least_seven_user_loss_wins':wins>=7}
    assert p['criteria']==guards and p['passed']==all(guards.values()) and p['user_logloss_wins']==wins
    assert abs(p['delta_logloss']-(b['pooled']['log_loss']-f['pooled']['log_loss']))<1e-12
    assert abs(p['delta_logloss_over_uniform']-(u['pooled']['log_loss']-f['pooled']['log_loss']))<1e-12


def test_canonical_fusion_ablation_retains_frozen_weights_and_actual_trial_cost():
    r=load(); folder=ROOT/'feature_bank/delivery/new_bank_v3'
    with (folder/'ablation_full_bank.csv').open(encoding='utf8',newline='') as stream:
        rows=[x for x in csv.DictReader(stream) if x['run_id']=='emg_calibrated_fusion_v1']
    assert len(rows)==264
    assert len({(x['subject'],x['calibration_budget'],x['removed_provider']) for x in rows})==264
    for row in rows:
        shots=row['calibration_budget'];user=row['subject'];arm=row['remaining_bank']
        assert arm=='reliability_minus_'+row['removed_provider'] and row['full_bank']=='reliability_bank'
        a=r['scores'][shots]['reliability_bank']; b=r['scores'][shots][arm]
        a=a['pooled'] if user=='ALL' else a['per_user'][user]
        b=b['pooled'] if user=='ALL' else b['per_user'][user]
        assert int(row['evaluation_trials'])==(1200 if user=='ALL' else 120)
        assert int(row['full_target_calibration_trials_per_user'])==int(row['remaining_target_calibration_trials_per_user'])==6*int(shots)
        for key in ('macro_f1','log_loss','brier','accuracy'):
            assert abs(float(row['full_'+key])-a[key])<1e-12
            assert abs(float(row['remaining_'+key])-b[key])<1e-12
        assert abs(float(row['delta_logloss'])-(b['log_loss']-a['log_loss']))<1e-12
        assert abs(float(row['delta_macro_f1'])-(a['macro_f1']-b['macro_f1']))<1e-12
        notes=json.loads(row['metadata_notes_json']);costs=notes['actual_target_calibration_trials_per_user']
        assert all(costs[name]==0 for name in list(r['source_models'])+['uniform_bank'])
        assert costs['reliability_bank']==costs[arm]==6*int(shots)
        assert notes['calibration_updates_only_weights_not_provider_models']
        assert notes['source_user_OOF_probability_temperatures'] and notes['no_IMU_features']
