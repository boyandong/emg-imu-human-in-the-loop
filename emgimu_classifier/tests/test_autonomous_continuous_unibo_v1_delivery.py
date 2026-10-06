import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT/'benchmarks/new_bank_v3'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_continuous_boundary_results_geometry_scores_and_source_disjointness():
    protocol_path = HERE/'AUTONOMOUS_CONTINUOUS_UNIBO_V1_PROTOCOL.json'
    p = json.loads(protocol_path.read_text())
    r = json.loads((HERE/'AUTONOMOUS_CONTINUOUS_UNIBO_V1_RESULTS.json').read_text())
    assert r['protocol_sha256'] == sha(protocol_path)
    assert all(sha(ROOT/path) == digest for path, digest in r['source_hashes'].items())
    source = {n for n, v in p['members'].items() if v['day'] == 1}
    evaluation = {n for n, v in p['members'].items() if v['day'] == 6}
    assert len(source) == len(evaluation) == 28 and not source & evaluation
    assert {n for state in r['source_states'] for n in state['source_member_sha256']} == source
    assert {row['member'] for row in r['recordings']} == evaluation
    assert len(r['recordings']) == 28
    errors = []; n_ref = n_det = n_match = 0
    for row in r['recordings']:
        refs = row['reference_bounds']; events = row['detected_bounds']; matches = row['matches']
        n_ref += len(refs); n_det += len(events); n_match += len(matches)
        assert len({m['event_index'] for m in matches}) == len(matches)
        assert len({m['reference_index'] for m in matches}) == len(matches)
        for m in matches:
            label, a, b = refs[m['reference_index']]; c, d = events[m['event_index']]
            iou = max(0, min(b, d)-max(a, c))/(max(b, d)-min(a, c))
            assert m['iou'] == iou and iou >= p['match_iou']
            assert m['label'] == label and label in range(2, 7)
            assert m['onset_error_s'] == (c-a)/200.
            assert m['offset_error_s'] == (d-b)/200.
            errors.append([abs(m['onset_error_s']), abs(m['offset_error_s'])])
    t = r['totals']
    assert (t['references'], t['detections'], t['matched']) == (n_ref, n_det, n_match)
    assert t['precision'] == n_match/n_det and t['recall'] == n_match/n_ref
    np.testing.assert_allclose([t['onset_mae_s'], t['offset_mae_s']], np.mean(errors, axis=0), atol=1e-12)
    assert not r['default_promoted']
