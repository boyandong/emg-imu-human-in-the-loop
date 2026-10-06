import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT/'benchmarks/new_bank_v3'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_existing_g5_source_identity_and_same_event_paired_metrics():
    protocol_path = HERE/'DETECTED_G5_UNIBO_V1_PROTOCOL.json'
    p = json.loads(protocol_path.read_text())
    r = json.loads((HERE/'DETECTED_G5_UNIBO_V1_RESULTS.json').read_text())
    assert r['protocol_sha256'] == sha(protocol_path)
    assert all(sha(ROOT/n) == digest for n, digest in r['source_hashes'].items())
    assert p['dtw_chain_sha256'] == sha(HERE/'DETECTED_DTW_UNIBO_V1_RESULTS.json')
    assert p['boundary_protocol_sha256'] == sha(HERE/'AUTONOMOUS_CONTINUOUS_UNIBO_V1_PROTOCOL.json')
    assert p['source_protocol_sha256'] == sha(HERE/'F5_PATH_UNIBO_PROTOCOL.json')
    source = json.loads((HERE/'F5_PATH_UNIBO_PROTOCOL.json').read_text())
    assert p['source_state_sha256'] == source['frozen_files']['source/fitted_states.pkl']
    assert p['source_metadata_sha256'] == source['frozen_files']['source/bout_metadata.json']
    parent = json.loads((HERE/'DETECTED_DTW_UNIBO_V1_RESULTS.json').read_text())
    old = {(e['member'], e['event_index']): e for e in parent['events']}
    assert len(r['events']) == len(old) == 1257
    assert {(e['member'], e['event_index']) for e in r['events']} == set(old)
    detected_correct = oracle_correct = count = 0
    for e in r['events']:
        previous = old[(e['member'], e['event_index'])]
        for key, value in previous.items():
            assert e[key] == value
        probability = np.array(e['g5_probability'])
        assert probability.shape == (4,) and np.isfinite(probability).all() and np.min(probability) >= 0
        np.testing.assert_allclose(probability.sum(), 1., atol=1e-12)
        assert e['g5_prediction'] == int(probability.argmax())
        assert e['complete_windows'] == (e['end']-e['start'])//40
        assert e['unrepresented_tail_samples'] == (e['end']-e['start'])%40
        if e['reference_label'] is None:
            assert e['g5_oracle_prediction'] is None
            continue
        oracle = np.array(e['g5_oracle_probability'])
        np.testing.assert_allclose(oracle.sum(), 1., atol=1e-12)
        assert e['g5_oracle_prediction'] == int(oracle.argmax())
        count += 1
        detected_correct += int(e['g5_prediction'] == e['reference_label'])
        oracle_correct += int(e['g5_oracle_prediction'] == e['reference_label'])
    for key, value in parent['totals'].items():
        assert r['totals'][key] == value
    t = r['totals']
    assert t['matched_supported'] == count
    assert (t['g5_detected_correct'], t['g5_oracle_correct']) == (detected_correct, oracle_correct)
    assert t['g5_matched_detection_accuracy'] == detected_correct/count
    assert t['g5_matched_oracle_accuracy'] == oracle_correct/count
    assert t['g5_end_to_end_reference_success'] == detected_correct/t['supported_references']
    supported = [e for e in r['events'] if e['reference_label'] is not None]
    diag = r['diagnostics']
    for label in (1, 2, 3):
        group = [e for e in supported if e['reference_label'] == label]
        observed = diag['per_class'][str(label)]
        assert observed['matched'] == len(group)
        assert observed['correct'] == sum(e['g5_prediction'] == label for e in group)
        assert sum(observed['predicted_counts'].values()) == len(group)
    assert sum(d['matched'] for d in diag['per_user'].values()) == count
    assert sum(d['g5_correct'] for d in diag['per_user'].values()) == detected_correct
    assert sum(diag['paired_correctness'].values()) == count
    assert diag['paired_correctness']['g5_1_dtw_0']+diag['paired_correctness']['g5_1_dtw_1'] == detected_correct
    truth = np.array([e['reference_label'] for e in supported])
    for arm, field in [('detected', 'g5_probability'), ('matched_oracle', 'g5_oracle_probability')]:
        probability = np.array([e[field] for e in supported]); score = r['scores'][arm]
        assert abs(score['active_macro_f1']-f1_score(truth, probability.argmax(axis=1), labels=[1, 2, 3], average='macro')) < 1e-12
        assert abs(score['log_loss']-log_loss(truth, probability, labels=np.arange(4))) < 1e-12
        assert abs(score['brier']-np.mean((probability-np.eye(4)[truth])**2)) < 1e-12
        assert 0 <= score['ece'] <= 1
    assert {s['user'] for s in r['source_selections']} == {f'u{i:02}' for i in range(1, 8)}
    for selection in r['source_selections']:
        assert selection['source_temperature'] > 0
        assert all('-d06-' not in trial and '-d07-' not in trial and '-d08-' not in trial
                   for trial in selection['source_recordings'])
    assert r['source_state_immutable'] and not r['classifier_refitted'] and not r['default_promoted']
