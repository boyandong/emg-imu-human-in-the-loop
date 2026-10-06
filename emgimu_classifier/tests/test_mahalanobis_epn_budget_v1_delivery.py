import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import softmax
from sklearn.metrics import f1_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT/'benchmarks/new_bank_v3'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_native_budget_provenance_covariance_oracle_and_probability_metrics():
    path = HERE/'MAHALANOBIS_EPN_BUDGET_V1_PROTOCOL.json'
    p = json.loads(path.read_text())
    r = json.loads((HERE/'MAHALANOBIS_EPN_BUDGET_V1_RESULTS.json').read_text())
    assert r['protocol_sha256'] == sha(path)
    assert all(sha(ROOT/n) == digest for n, digest in r['source_hashes'].items())
    assert not set(p['source_users']) & set(p['target_users'])
    assert r['source_state_immutable'] and not r['default_promoted']
    selections = {s['user']: s for s in r['selections']}
    assert len(r['blocks']) == 150
    assert sum(b['mahalanobis_eligible'] for b in r['blocks']) == 20
    collected = {}
    for b in r['blocks']:
        selection = selections[b['user']]
        assert b['dimension'] == p['feature_dimensions'][b['family']]
        expected_ids = [i for h in range(6) for i in selection['reserved_calibration'][str(h)][:b['shots']]]
        assert b['calibration_trials'] == expected_ids
        assert b['evaluation_trials'] == selection['evaluation']
        assert len(set(expected_ids)) == 6*b['shots']
        assert not set(expected_ids) & set(selection['evaluation'])
        assert b['mahalanobis_eligible'] == (b['shots'] >= b['dimension']+2)
        if not b['mahalanobis_eligible']:
            assert 'independent trials' in b['ineligible_reason']
            assert 'mahalanobis_probability' not in b
        else:
            assert b['family'] == 'pattern' and b['shots'] in (10, 20)
            assert all(n == b['shots'] for n in b['trial_counts'].values())
            x = np.array(b['calibration_features']); labels = np.array(b['calibration_labels'])
            query = np.array(b['evaluation_features']); d = b['dimension']; columns = []
            for h in range(6):
                group = x[labels == h]; mu = group.mean(axis=0)
                cov = np.cov(group, rowvar=False); iso = np.trace(cov)/d
                reg = .8*cov+.2*iso*np.eye(d)+max(iso*1e-8, 1e-10)*np.eye(d)
                delta = query-mu
                columns.append(np.sqrt(np.maximum(np.sum(delta*np.linalg.solve(reg, delta.T).T, axis=1), 0.)))
            np.testing.assert_allclose(b['distances'], np.column_stack(columns), rtol=2e-6, atol=1e-5)
            np.testing.assert_allclose(b['mahalanobis_probability'],
                                       softmax(-np.array(b['distances'])/b['similarity_scale'], axis=1), atol=1e-12)
        for metric in ['euclidean']+(['mahalanobis'] if b['mahalanobis_eligible'] else []):
            key = f"{b['family']}_{b['shots']}shot_{metric}"
            collected.setdefault(key, []).append((b['labels'], b[f'{metric}_probability']))
    assert set(collected) == set(r['scores'])
    for key, parts in collected.items():
        y = np.concatenate([v[0] for v in parts]); probability = np.concatenate([v[1] for v in parts])
        np.testing.assert_allclose(probability.sum(axis=1), 1., atol=1e-12)
        score = r['scores'][key]
        assert abs(score['macro_f1']-f1_score(y, probability.argmax(axis=1), labels=np.arange(6), average='macro')) < 1e-12
        assert abs(score['log_loss']-log_loss(y, probability, labels=np.arange(6))) < 1e-12
