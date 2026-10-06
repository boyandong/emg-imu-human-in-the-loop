import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT/'benchmarks/new_bank_v3'


def test_all_references_and_events_accounted_including_missed_and_unsupported():
    r = json.loads((HERE/'DETECTED_G5_OUTCOME_V1_RESULTS.json').read_text())
    for path, digest in r['source_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest
    assert r['generator_sha256'] == hashlib.sha256((HERE/'detected_g5_outcome_audit.py').read_bytes()).hexdigest()
    boundary = json.loads((HERE/'AUTONOMOUS_CONTINUOUS_UNIBO_V1_RESULTS.json').read_text())
    source = json.loads((HERE/'DETECTED_G5_UNIBO_V1_RESULTS.json').read_text())
    expected = {(f['member'], j): (label, a, b) for f in boundary['recordings']
                for j, (label, a, b) in enumerate(f['reference_bounds'])}
    actual = {(e['member'], e['reference_index']): (e['native_label'], e['start'], e['end']) for e in r['references']}
    assert actual == expected and len(actual) == len(r['references']) == 1411
    matched = [e for e in r['references'] if e['event_index'] is not None]
    identities = [(e['member'], e['event_index']) for e in matched+r['unmatched_detections']]
    assert len(identities) == len(set(identities)) == 1257
    lookup = {(e['member'], e['event_index']): e for e in source['events']}
    assert set(identities) == set(lookup)
    for e in matched+r['unmatched_detections']:
        assert e['prediction'] == lookup[(e['member'], e['event_index'])]['g5_prediction']
    for label in (1, 2, 3):
        group = [e for e in r['references'] if e['label'] == label]
        stats = r['per_class'][str(label)]
        correct = sum(e['prediction'] == label for e in group)
        missed = sum(e['prediction'] is None for e in group)
        assert stats['references'] == len(group)
        assert stats['correct'] == correct and stats['missed'] == missed
        assert stats['wrong'] == len(group)-correct-missed
        assert stats['end_to_end_recall'] == correct/len(group)
        assert sum(stats['prediction_counts'].values())+missed == len(group)
    t = r['totals']
    assert (t['correct'], t['wrong'], t['missed']) == (516, 234, 100)
    assert t['unsupported_matched']+t['unsupported_missed']+t['supported_references'] == 1411
    assert t['unmatched_detections'] == len(r['unmatched_detections']) == 21
    supported_matched = [e for e in matched if e['label'] is not None]
    for key, oracle_correct, detected_correct in [('oracle_correct_detected_wrong', True, False),
                                                  ('oracle_wrong_detected_correct', False, True)]:
        assert r['boundary_paired'][key] == sum((e['oracle_prediction'] == e['label']) == oracle_correct
            and (e['prediction'] == e['label']) == detected_correct for e in supported_matched)
    assert not r['default_promoted'] and not r['completion_proven']
