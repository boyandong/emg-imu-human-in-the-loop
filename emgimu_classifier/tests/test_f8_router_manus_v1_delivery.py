import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT/'benchmarks/new_bank_v3'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_paired_router_native_ids_weights_probabilities_and_metrics():
    path = HERE/'F8_ROUTER_MANUS_V1_PROTOCOL.json'
    p = json.loads(path.read_text())
    r = json.loads((HERE/'F8_ROUTER_MANUS_V1_RESULTS.json').read_text())
    assert r['protocol_sha256'] == sha(path)
    assert p['descriptor_sha256'] == sha(HERE/'F8_DOCUMENT_MANUS_RESULTS.json')
    assert all(sha(ROOT/n) == digest for n, digest in r['source_hashes'].items())
    assert r['source_state_immutable'] and not r['default_promoted']
    assert r['provider_dimensions'] == {'TD24': 24, 'pattern': 8, 'SPD': 36, 'log_bands': 32}
    assert len(r['blocks']) == 24
    parent = json.loads((HERE/'F8_DOCUMENT_MANUS_RESULTS.json').read_text())
    lookup = {(b['phase'], b['shots'], b['user']): b for b in parent['records']}
    for block in r['blocks']:
        original = lookup[(block['phase'], block['shots'], block['user'])]
        source = set(block['source_trials']); cal = set(block['calibration_trials']); ev = set(block['evaluation_trials'])
        assert source == set(original['source_trials'])
        assert cal == set(original['calibration_trials']) and ev == set(original['evaluation_trials'])
        assert not source & (cal | ev) and not cal & ev
        raw = np.maximum(np.exp(-np.clip(block['risks'], 0., 5.)), .05)
        weights = raw/raw.sum()
        np.testing.assert_allclose(block['weights'], weights, atol=1e-14)
        experts = np.asarray(block['providers'])
        assert experts.shape == (4, len(ev), 6)
        np.testing.assert_allclose(experts.sum(axis=2), 1., atol=1e-10)
        np.testing.assert_allclose(block['probabilities']['TD24'], experts[0], atol=1e-14)
        np.testing.assert_allclose(block['probabilities']['uniform'], experts.mean(axis=0), atol=1e-14)
        np.testing.assert_allclose(block['probabilities']['F8'], np.einsum('k,knh->nh', weights, experts), atol=1e-14)
    for phase in ('validation', 'descriptive_final'):
        for shots in (1, 2):
            blocks = [b for b in r['blocks'] if b['phase'] == phase and b['shots'] == shots]
            assert {b['user'] for b in blocks} == set(p['users'])
            y = np.concatenate([b['labels'] for b in blocks])
            for name in ('TD24', 'uniform', 'F8'):
                probability = np.concatenate([b['probabilities'][name] for b in blocks])
                score = r['scores'][f'{phase}_{shots}shot'][name]
                assert abs(score['macro_f1']-f1_score(y, probability.argmax(axis=1), labels=np.arange(6), average='macro')) < 1e-12
                assert abs(score['log_loss']-log_loss(y, probability, labels=np.arange(6))) < 1e-12
