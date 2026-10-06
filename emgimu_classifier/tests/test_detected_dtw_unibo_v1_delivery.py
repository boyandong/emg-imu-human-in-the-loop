import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT/'benchmarks/new_bank_v3'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_detected_chain_identities_controls_and_metrics():
    path = HERE/'DETECTED_DTW_UNIBO_V1_PROTOCOL.json'
    p = json.loads(path.read_text())
    r = json.loads((HERE/'DETECTED_DTW_UNIBO_V1_RESULTS.json').read_text())
    assert r['protocol_sha256'] == sha(path)
    for filename, key in [('AUTONOMOUS_CONTINUOUS_UNIBO_V1_RESULTS.json', 'boundary_result_sha256'),
                          ('AUTONOMOUS_CONTINUOUS_UNIBO_V1_PROTOCOL.json', 'boundary_protocol_sha256'),
                          ('F5_PATH_UNIBO_PROTOCOL.json', 'template_protocol_sha256')]:
        assert sha(HERE/filename) == p[key]
    assert all(sha(ROOT/n) == digest for n, digest in r['source_hashes'].items())
    parent = json.loads((HERE/'AUTONOMOUS_CONTINUOUS_UNIBO_V1_RESULTS.json').read_text())
    lookup = {record['member']: record for record in parent['recordings']}
    expected = {(record['member'], i) for record in parent['recordings'] for i in range(len(record['detected_bounds']))}
    assert {(e['member'], e['event_index']) for e in r['events']} == expected
    assert len(r['events']) == len(expected)
    mapping = {2: 2, 3: 1, 6: 3}
    totals = {'supported_references': sum(ref[0] in mapping for record in parent['recordings'] for ref in record['reference_bounds']),
              'matched_supported': 0, 'detected_correct': 0, 'oracle_correct_on_matched': 0,
              'unsupported_matched': 0, 'unmatched_detections': 0}
    for event in r['events']:
        record = lookup[event['member']]
        assert [event['start'], event['end']] == record['detected_bounds'][event['event_index']]
        d = np.array(event['distances'])
        assert d.shape == (4,) and np.isfinite(d).all() and np.min(d) >= 0
        assert event['prediction'] == int(d.argmin())
        match = next((m for m in record['matches'] if m['event_index'] == event['event_index']), None)
        if match is None:
            totals['unmatched_detections'] += 1
            assert event['reference_label'] is None
            continue
        label, start, end = record['reference_bounds'][match['reference_index']]
        if label not in mapping:
            totals['unsupported_matched'] += 1
            assert event['reference_label'] is None
            continue
        assert event['reference_label'] == mapping[label]
        assert (event['reference_start'], event['reference_end']) == (start, end)
        assert event['oracle_prediction'] == int(np.argmin(event['oracle_distances']))
        totals['matched_supported'] += 1
        totals['detected_correct'] += int(event['prediction'] == event['reference_label'])
        totals['oracle_correct_on_matched'] += int(event['oracle_prediction'] == event['reference_label'])
    for key, value in totals.items():
        assert r['totals'][key] == value
    assert r['totals']['matched_detection_accuracy'] == totals['detected_correct']/totals['matched_supported']
    assert r['totals']['matched_oracle_accuracy'] == totals['oracle_correct_on_matched']/totals['matched_supported']
    assert r['totals']['end_to_end_reference_success'] == totals['detected_correct']/totals['supported_references']
    old = json.loads((HERE/'F5_PATH_UNIBO_RESULTS.json').read_text())
    selections = {s['user']: s for s in old['source_selection']}
    for selection in r['source_selections']:
        expected_selection = selections[selection['user']]
        assert selection['candidate_ids'] == expected_selection['candidate_ids']
        assert selection['medoid_ids'] == expected_selection['medoid_ids']
    assert r['source_state_immutable'] and not r['default_promoted']
