"""Saved frozen boundary screen is reproducible, including a negative outcome."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT/'benchmarks/new_bank_v3'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_autonomous_source_only_protocol_and_metric_readback():
    protocol_path = HERE/'AUTONOMOUS_UNIBO_V1_PROTOCOL.json'
    protocol = json.loads(protocol_path.read_text())
    result = json.loads((HERE/'AUTONOMOUS_UNIBO_V1_RESULTS.json').read_text())
    assert result['protocol_sha256'] == sha(protocol_path)
    assert protocol['parent_protocol_sha256'] == sha(HERE/'F5_PATH_UNIBO_PROTOCOL.json')
    assert all(sha(ROOT/path) == digest for path, digest in result['source_hashes'].items())
    assert len(result['source_states']) == len(protocol['by_user']) == 7
    for state in result['source_states']:
        split = protocol['by_user'][state['user']]
        assert set(state['source_raw_sha256']) == set(split['source_trials'])
        assert not set(split['source_trials']) & set(split['evaluation_trials'])
        assert state['on'] > state['off'] > 0
        assert state['source_rest_samples'] >= 200
    counts = {'references': 0, 'detections': 0, 'matched': 0, 'censored_trials': 0}
    expected = {(user, trial) for user, split in protocol['by_user'].items()
                for trial in split['evaluation_trials']}
    assert {(r['user'], r['trial']) for r in result['recordings']} == expected
    assert len(result['recordings']) == len(expected)
    for row in result['recordings']:
        counts['references'] += len(row['reference_bounds'])
        counts['detections'] += len(row['detected_bounds'])
        counts['matched'] += len(row['matches'])
        counts['censored_trials'] += int(row['censored_end'])
        for label, start, end in row['reference_bounds']:
            assert label in (1, 2, 3) and end-start >= 200
    for name, value in counts.items():
        assert result['totals'][name] == value
    for name, denominator in [('precision', 'detections'), ('recall', 'references')]:
        assert result['totals'][name] == counts['matched']/max(counts[denominator], 1)
    assert result['default_promoted'] is False
