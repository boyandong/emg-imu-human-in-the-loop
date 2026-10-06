import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT/'benchmarks/new_bank_v3'


def probability(raw, temperature):
    p = np.maximum(np.array(raw),1e-15)**(1/temperature)
    return p/p.sum(axis=1,keepdims=True)


def test_source_oof_calibration_folds_temperature_and_fixed_target_probabilities():
    p = json.loads((HERE/'F8_CALIBRATED_MANUS_V2_PROTOCOL.json').read_text())
    r = json.loads((HERE/'F8_CALIBRATED_MANUS_V2_RESULTS.json').read_text())
    parent = json.loads((HERE/'F8_ROUTER_MANUS_V1_RESULTS.json').read_text())
    assert r['protocol_sha256'] == hashlib.sha256((HERE/'F8_CALIBRATED_MANUS_V2_PROTOCOL.json').read_bytes()).hexdigest()
    assert r['parent_sha256'] == p['parent_sha256'] == hashlib.sha256((HERE/'F8_ROUTER_MANUS_V1_RESULTS.json').read_bytes()).hexdigest()
    for name,digest in r['source_hashes'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest
    folds = r['source_folds']; oof = r['source_oof']; source_ids = {x['trial'] for x in oof}
    assert len(source_ids) == len(oof)
    assert {f['held_user'] for f in folds} == set(p['users'])
    assert sum(len(f['held_trials']) for f in folds) == len(oof)
    for f in folds:
        assert set(f['train_users']) == set(p['users'])-{f['held_user']}
        assert not set(f['train_trials']) & set(f['held_trials'])
        assert set(f['train_trials']) | set(f['held_trials']) == source_ids
        assert set(f['held_trials']) == {x['trial'] for x in oof if x['user'] == f['held_user']}
    target_ids = {t for b in parent['blocks'] for t in b['calibration_trials']+b['evaluation_trials']}
    assert not source_ids & target_ids
    y = np.array([x['label'] for x in oof])
    for name in p['providers']:
        raw = np.array([x['raw_probability'][name] for x in oof]); candidates = []
        for t,row in zip(p['temperature_candidates'],r['temperature_selection'][name]):
            loss = -np.log(np.maximum(probability(raw,t)[np.arange(len(y)),y],1e-15)).mean()
            assert row['temperature'] == t and np.isclose(loss,row['log_loss'])
            candidates.append((float(loss),abs(float(np.log(t))),t))
        assert min(candidates)[2] == r['temperatures'][name]
    assert len(r['blocks']) == len(parent['blocks']) == 24
    for b,old in zip(r['blocks'],parent['blocks']):
        for key in ('phase','shots','user','source_trials','calibration_trials','evaluation_trials','labels','weights','risks'):
            assert b[key] == old[key]
        assert b['uncalibrated_probabilities'] == old['probabilities']
        experts = np.stack([probability(old['providers'][i],r['temperatures'][n]) for i,n in enumerate(p['providers'])])
        np.testing.assert_allclose(b['providers'],experts,atol=1e-12)
        np.testing.assert_allclose(b['probabilities']['TD24'],experts[0],atol=1e-12)
        np.testing.assert_allclose(b['probabilities']['uniform'],experts.mean(axis=0),atol=1e-12)
        np.testing.assert_allclose(b['probabilities']['F8'],np.einsum('p,ptc->tc',old['weights'],experts),atol=1e-12)
    for phase in ('validation','descriptive_final'):
        for shots in (1,2):
            group = [b for b in r['blocks'] if b['phase'] == phase and b['shots'] == shots]
            labels = np.concatenate([b['labels'] for b in group])
            for name in ('TD24','uniform','F8'):
                prob = np.concatenate([b['probabilities'][name] for b in group]); metric = r['scores'][f'{phase}_{shots}shot'][name]
                assert np.isclose(metric['log_loss'],log_loss(labels,prob,labels=range(6)))
                assert np.isclose(metric['macro_f1'],f1_score(labels,prob.argmax(axis=1),labels=range(6),average='macro',zero_division=0))
    assert not r['default_promoted'] and not r['completion_proven']
