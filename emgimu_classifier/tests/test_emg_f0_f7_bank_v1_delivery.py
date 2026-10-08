import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.linalg import eigvalsh

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT/'benchmarks/new_bank_v3'


def load(path): return json.loads(path.read_text(encoding='utf8'))


def normalized_exp(logits):
    exp=np.exp(logits-logits.max(axis=1,keepdims=True))
    return exp/exp.sum(axis=1,keepdims=True)


def distance(a,b): return np.linalg.norm(np.log(eigvalsh(a,b)))


def metrics(y,q):
    prediction=q.argmax(1);f=[]
    for c in range(6):
        denominator=np.sum(y==c)+np.sum(prediction==c)
        f.append(2*np.sum((y==c)&(prediction==c))/denominator if denominator else 0)
    return {'macro_f1':sum(f)/6,'accuracy':np.mean(y==prediction),
            'log_loss':-np.mean(np.log(np.clip(q[np.arange(len(y)),y],np.finfo(float).eps,1))),
            'brier':np.mean((q-np.eye(6)[y])**2)}


def test_frozen_source_readout_independent_affine_geometry_and_paired_trial_coverage():
    p=load(HERE/'EMG_F0_F7_BANK_V1_PROTOCOL.json');r=load(HERE/'EMG_F0_F7_BANK_V1_RESULTS.json')
    assert hashlib.sha256((HERE/'EMG_F0_F7_BANK_V1_PROTOCOL.json').read_bytes()).hexdigest()==r['protocol_sha256']
    assert hashlib.sha256((HERE/'EMG_F0_F7_BANK_V1_PREDICTIONS.csv').read_bytes()).hexdigest()==r['prediction_sha256']
    for path,digest in p['source_sha256'].items(): assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
    assert p['source_users']==list(range(1,16)) and p['target_users']==list(range(42,52))
    assert set(p['target_users']).isdisjoint(p['source_users']+p['known_prior_target_users'])
    assert r['source_state_immutable'] and not r['default_promoted'] and not r['physical_validation_proven'] and not r['completion_proven']
    with (HERE/'EMG_F0_F7_BANK_V1_PREDICTIONS.csv').open(encoding='utf8',newline='') as f:rows=list(csv.DictReader(f))
    assert len({(v['user'],v['shots_per_class'],v['arm'],v['trial_id']) for v in rows})==len(rows)
    fit=r['source_fit'];mean,scale,coef,intercept=[np.array(fit[k]) for k in ('mean','scale','coef','intercept')]
    assert fit['feature_dimension']==48 and coef.shape==(6,48) and np.all(scale>0)
    assert fit['source_trials']==len(r['source_trial_ids'])==sum(fit['class_trial_counts'].values())
    assert fit['rest_windows']>0 and len(fit['thresholds'])==8
    for block in r['blocks']:
        u=block['user'];ids=block['evaluation_ids'];matrices=np.array(block['evaluation_matrices'])
        assert not set(ids)&set(r['source_trial_ids']) and len(ids)==len(set(ids))
        reserved=set(t for values in block['reserved_ids'].values() for t in values)
        assert len(reserved)==30 and not reserved&set(ids)
        # F0 returns float32. This sklearn scaler casts its fitted parameters
        # to float32 before in-place subtraction/division and float64 projection.
        # Preserve those steps rather than weakening the probability tolerance.
        standardized=np.array(block['evaluation_features'],dtype=np.float32)
        standardized-=mean.astype(np.float32);standardized/=scale.astype(np.float32)
        expected_base=normalized_exp(standardized@coef.T+intercept)
        np.testing.assert_allclose(expected_base,block['F0_probabilities'],atol=1e-12,rtol=0)
        calibrations={c['shots']:c for c in block['calibrations']}
        assert set(calibrations)=={1,2,5}
        for shots,cal in calibrations.items():
            assert len(cal['ids'])==len(set(cal['ids']))==6*shots
            assert set(cal['ids'])<=reserved and not set(cal['ids'])&set(ids)
            assert not set(cal['ids'])&set(r['source_trial_ids'])
            assert set(calibrations[1]['ids'])<=set(calibrations[2]['ids'])<=set(calibrations[5]['ids'])
            cm=np.array(cal['calibration_matrices']);labels=np.array(cal['labels'])
            expected_prototypes=np.stack([cm[labels==c].mean(0) for c in range(6)])
            np.testing.assert_allclose(expected_prototypes,cal['prototypes'],atol=1e-12,rtol=0)
            between=[distance(expected_prototypes[i],expected_prototypes[j]) for i in range(6) for j in range(i+1,6)]
            temp=max(float(np.median(between)),1e-10)
            assert abs(temp-cal['temperature'])<1e-9
            distances=np.array([[distance(m,prototype) for prototype in expected_prototypes] for m in matrices])
            probability=normalized_exp(-distances/temp)
            np.testing.assert_allclose(probability,cal['F7_probabilities'],atol=1e-9,rtol=0)
            for arm in ('F0','F7','F0_F7','F0_uniform'):
                selected=[v for v in rows if int(v['user'])==u and int(v['shots_per_class'])==shots and v['arm']==arm]
                assert [v['trial_id'] for v in selected]==ids
                q=np.array([[float(v[f'p_{c}']) for c in range(6)] for v in selected])
                expected={'F0':expected_base,'F7':probability,'F0_F7':(expected_base+probability)/2,'F0_uniform':expected_base/2+1/12}[arm]
                np.testing.assert_allclose(q,expected,atol=1e-9,rtol=0)
        zero=[v for v in rows if int(v['user'])==u and int(v['shots_per_class'])==0]
        assert [v['trial_id'] for v in zero]==ids and all(v['arm']=='F0' for v in zero)


def test_independent_scores_worst_users_and_primary_guard():
    r=load(HERE/'EMG_F0_F7_BANK_V1_RESULTS.json')
    with (HERE/'EMG_F0_F7_BANK_V1_PREDICTIONS.csv').open(encoding='utf8',newline='') as f:rows=list(csv.DictReader(f))
    for budget,arms in r['scores'].items():
        for arm,stored in arms.items():
            group=[v for v in rows if v['shots_per_class']==budget and v['arm']==arm]
            for user,expected in [('ALL',stored['pooled']),*stored['per_user'].items()]:
                values=group if user=='ALL' else [v for v in group if v['user']==user]
                y=np.array([int(v['label']) for v in values]);q=np.array([[float(v[f'p_{c}']) for c in range(6)] for v in values])
                for key,value in metrics(y,q).items(): assert abs(expected[key]-value)<1e-12
            assert stored['minimum_user_macro_f1']==min(v['macro_f1'] for v in stored['per_user'].values())
    scores=r['scores']['5'];a,b,z=[scores[v]['pooled'] for v in ('F0','F0_F7','F0_uniform')]
    primary=r['primary_five_shot']
    wins=sum(scores['F0_F7']['per_user'][u]['log_loss']<v['log_loss'] for u,v in scores['F0']['per_user'].items())
    criteria={'lower_log_loss':b['log_loss']<a['log_loss'],'nonworse_macro_f1':b['macro_f1']>=a['macro_f1'],
              'lower_brier':b['brier']<a['brier'],'lower_loss_than_uniform':b['log_loss']<z['log_loss'],'at_least7_user_loss_wins':wins>=7}
    assert primary['criteria']==criteria and primary['conjunction_passed']==all(criteria.values())
    assert primary['user_logloss_wins']==wins
    assert abs(primary['delta_logloss']-(a['log_loss']-b['log_loss']))<1e-12
    assert abs(primary['delta_macro_f1']-(b['macro_f1']-a['macro_f1']))<1e-12
